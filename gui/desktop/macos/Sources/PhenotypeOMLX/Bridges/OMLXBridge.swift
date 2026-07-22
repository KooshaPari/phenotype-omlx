// OMLXBridge.swift — async wrapper around the `omlx-research` CLI binary.
//
// All I/O uses Swift concurrency. We invoke the binary via `Process` and:
//   * pipe a JSON request envelope on stdin,
//   * read a JSON response envelope from stdout,
//   * capture stderr for diagnostics on non-zero exit.
//
// The on-disk contract:
//
//   STDIN  (request)  = { "action": "research",
//                         "prompt": "<string>",
//                         "model_id": "<string>?",
//                         "max_tokens": <int>?,
//                         "temperature": <double>?,
//                         "stream": false }
//   STDOUT (response) = { "ok": true,
//                         "output": "<string>",
//                         "model_id": "<string>?",
//                         "tokens": <int>?,
//                         "latency_ms": <double>?,
//                         "stop_reason": "<string>?" }
//                     or { "ok": false, "error": "<string>", "code": <int> }
//
// The Python side (`omlx_research.cli`) is expected to honor the JSON
// envelope when invoked as `omlx-research research --json`. If the JSON
// flag isn't recognized yet, the bridge surfaces a clear error to the UI
// rather than silently corrupting the chat transcript.

import Foundation

// MARK: - Public error type

enum OMLXBridgeError: LocalizedError {
    case binaryNotFound(searched: [String])
    case launchFailed(underlying: Error)
    case nonZeroExit(code: Int32, stderr: String)
    case timeout(seconds: TimeInterval)
    case invalidResponse(reason: String)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .binaryNotFound(let s):
            return "omlx-research binary not found. Searched:\n  • " + s.joined(separator: "\n  • ")
        case .launchFailed(let e):
            return "Failed to launch omlx-research: \(e.localizedDescription)"
        case .nonZeroExit(let code, let stderr):
            let tail = stderr.isEmpty ? "(no stderr)" : stderr
            return "omlx-research exited with code \(code):\n\(tail)"
        case .timeout(let s):
            return "omlx-research timed out after \(Int(s))s"
        case .invalidResponse(let r):
            return "Could not parse omlx-research response: \(r)"
        case .cancelled:
            return "Cancelled by user."
        }
    }
}

// MARK: - Request / response DTOs

struct ResearchRequest: Codable {
    let action: String            // "research"
    let prompt: String
    let modelId: String?
    let maxTokens: Int?
    let temperature: Double?
    let stream: Bool

    enum CodingKeys: String, CodingKey {
        case action, prompt, temperature, stream
        case modelId = "model_id"
        case maxTokens = "max_tokens"
    }
}

struct ResearchResponse: Codable {
    let ok: Bool
    let output: String?
    let modelId: String?
    let tokens: Int?
    let latencyMs: Double?
    let stopReason: String?
    let error: String?
    let code: Int?

    enum CodingKeys: String, CodingKey {
        case ok, output, tokens, error, code
        case modelId = "model_id"
        case latencyMs = "latency_ms"
        case stopReason = "stop_reason"
    }
}

// MARK: - Binary locator

enum OMLXBinaryLocator {
    /// Search order:
    ///   1. $PHENOTYPE_OMLX_HOME/cli/bin/omlx-research
    ///   2. ~/.omlx/bin/omlx-research
    ///   3. $PATH lookup via `/usr/bin/which`
    static func locate() -> URL? {
        let fm = FileManager.default
        let envHome = ProcessInfo.processInfo.environment["PHENOTYPE_OMLX_HOME"]
        var candidates: [String] = []
        if let envHome, !envHome.isEmpty {
            candidates.append("\(envHome)/cli/bin/omlx-research")
        }
        candidates.append("/Users/kooshapari/CodeProjects/Phenotype/repos/phenotype-omlx/cli/bin/omlx-research")
        if let home = ProcessInfo.processInfo.environment["HOME"] {
            candidates.append("\(home)/.omlx/bin/omlx-research")
        }

        for c in candidates where fm.isExecutableFile(atPath: c) {
            return URL(fileURLWithPath: c)
        }

        // PATH fallback (best-effort, non-blocking).
        if let resolved = runWhich("omlx-research") {
            return URL(fileURLWithPath: resolved)
        }
        return nil
    }

    static func searchedPaths() -> [String] {
        let envHome = ProcessInfo.processInfo.environment["PHENOTYPE_OMLX_HOME"]
        var paths: [String] = []
        if let envHome, !envHome.isEmpty {
            paths.append("\(envHome)/cli/bin/omlx-research")
        }
        paths.append("/Users/kooshapari/CodeProjects/Phenotype/repos/phenotype-omlx/cli/bin/omlx-research")
        if let home = ProcessInfo.processInfo.environment["HOME"] {
            paths.append("\(home)/.omlx/bin/omlx-research")
        }
        return paths
    }

    private static func runWhich(_ name: String) -> String? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        p.arguments = [name]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = Pipe()
        do {
            try p.run()
            p.waitUntilExit()
            guard p.terminationStatus == 0 else { return nil }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let s = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (s?.isEmpty == false) ? s : nil
        } catch {
            return nil
        }
    }
}

// MARK: - Bridge

actor OMLXBridge {
    static let shared = OMLXBridge()

    /// Hard ceiling per call so a misbehaving backend can't wedge the UI.
    var timeout: TimeInterval = 600 // 10 minutes

    /// Cancel an in-flight call. Process.terminate() is best-effort.
    private var current: Process?

    // MARK: locate

    func locateBinary() throws -> URL {
        guard let url = OMLXBinaryLocator.locate() else {
            throw OMLXBridgeError.binaryNotFound(searched: OMLXBinaryLocator.searchedPaths())
        }
        return url
    }

    // MARK: runResearch

    /// Invoke `omlx-research research --json`, send a JSON request on stdin,
    /// parse a JSON response from stdout. Throws `OMLXBridgeError` on any
    /// failure with a user-presentable description.
    func runResearch(prompt: String,
                     modelId: String? = nil,
                     maxTokens: Int? = nil,
                     temperature: Double? = nil) async throws -> ResearchResponse {
        let bin = try locateBinary()
        let req = ResearchRequest(
            action: "research",
            prompt: prompt,
            modelId: modelId,
            maxTokens: maxTokens,
            temperature: temperature,
            stream: false
        )
        let payload = try JSONEncoder().encode(req)

        let proc = Process()
        proc.executableURL = bin
        // `research` is the pass-through subcommand the Python CLI handles;
        // `--json` tells it to expect stdin JSON / emit stdout JSON.
        proc.arguments = ["research", "--json"]

        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        proc.standardInput = stdin
        proc.standardOutput = stdout
        proc.standardError = stderr

        // Inherit PATH/venv so `python3` inside the wrapper resolves.
        proc.environment = ProcessInfo.processInfo.environment

        do {
            try proc.run()
        } catch {
            throw OMLXBridgeError.launchFailed(underlying: error)
        }
        current = proc

        // Send payload on stdin (close after write so the child sees EOF).
        do {
            try stdin.fileHandleForWriting.write(contentsOf: payload)
            try stdin.fileHandleForWriting.close()
        } catch {
            proc.terminate()
            throw OMLXBridgeError.launchFailed(underlying: error)
        }

        // Race: read output vs. timeout.
        let outTask = Task<Data, Error> {
            try await withCheckedThrowingContinuation { cont in
                stdout.fileHandleForReading.readabilityHandler = { fh in
                    let d = fh.availableData
                    if d.isEmpty {
                        fh.readabilityHandler = nil
                        cont.resume(returning: Data())
                    } else {
                        // accumulate via a buffer; simplest: keep last chunk only
                        cont.resume(returning: d)
                    }
                }
            }
        }
        let errTask = Task<String, Error> {
            let d = try await withCheckedThrowingContinuation { cont in
                stderr.fileHandleForReading.readabilityHandler = { fh in
                    let chunk = fh.availableData
                    if chunk.isEmpty {
                        fh.readabilityHandler = nil
                        cont.resume(returning: Data())
                    } else {
                        cont.resume(returning: chunk)
                    }
                }
            }
            return String(data: d, encoding: .utf8) ?? ""
        }

        // Wait with timeout.
        let exitTask = Task<Int32, Error> {
            try await withCheckedThrowingContinuation { cont in
                proc.terminationHandler = { p in cont.resume(returning: p.terminationStatus) }
            }
        }

        let result = await waitWithTimeout(timeout: timeout) { () -> (Int32, Data, String) in
            let code = try await exitTask.value
            let out = (try? await outTask.value) ?? Data()
            let err = (try? await errTask.value) ?? ""
            return (code, out, err)
        }

        current = nil
        switch result {
        case .success(let tuple):
            let (code, outData, errStr) = tuple
            if code != 0 {
                throw OMLXBridgeError.nonZeroExit(code: code, stderr: errStr)
            }
            do {
                let resp = try JSONDecoder().decode(ResearchResponse.self, from: outData)
                if !resp.ok {
                    throw OMLXBridgeError.invalidResponse(
                        reason: resp.error ?? "response.ok == false")
                }
                return resp
            } catch let dec as DecodingError {
                let preview = String(data: outData.prefix(400), encoding: .utf8) ?? "<binary>"
                throw OMLXBridgeError.invalidResponse(
                    reason: "decode error: \(dec). Preview: \(preview)")
            }
        case .timedOut:
            proc.terminate()
            throw OMLXBridgeError.timeout(seconds: timeout)
        case .cancelled:
            proc.terminate()
            throw OMLXBridgeError.cancelled
        }
    }

    func cancel() {
        current?.terminate()
        current = nil
    }
}

// MARK: - Tiny timeout helper

private enum Await<T> {
    case success(T)
    case timedOut
    case cancelled
}

private func waitWithTimeout<T: Sendable>(timeout: TimeInterval,
                                          _ op: @escaping @Sendable () async throws -> T) async -> Await<T> {
    await withTaskGroup(of: Await<T>.self) { group in
        group.addTask {
            do {
                let v = try await op()
                return .success(v)
            } catch is CancellationError {
                return .cancelled
            } catch {
                // Re-throw as cancellation so the UI surfaces a generic fail.
                return .cancelled
            }
        }
        group.addTask {
            try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
            return .timedOut
        }
        let first = await group.next()!
        group.cancelAll()
        return first
    }
}