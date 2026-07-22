// StatusBridge.swift — parses output from `omlx-research status`.
//
// `omlx-research status` invokes `python3 -m omlx_research.cli status`.
// We don't depend on that CLI emitting JSON — the GUI is robust to the
// free-form output and extracts the fields it cares about.
//
// JSON contract (when the Python side learns to emit it):
//
//   STDOUT  = { "mlx_version": "<string>",
//               "turboquant_plus": { "enabled": <bool>,
//                                    "kv_bits": <int>,
//                                    "skip_last": <bool> },
//               "ffi_loaded": <bool>,
//               "perf_core": "<path>?",
//               "backend": "<string>" }
//   or
//   STDOUT  = "<plain text, line-oriented>"
//
// Both shapes are tolerated. If JSON decoding fails, we fall back to
// scanning the lines for `key: value` pairs (the Python CLI emits that style
// for several subcommands).

import Foundation

struct OMLXStatus: Codable, Equatable {
    var mlxVersion: String?
    var turboquantPlusEnabled: Bool?
    var turboquantKVBits: Int?
    var turboquantSkipLast: Bool?
    var ffiLoaded: Bool?
    var perfCorePath: String?
    var backend: String?
    /// Raw lines we couldn't classify — preserved verbatim for the UI.
    var rawLines: [String] = []

    enum CodingKeys: String, CodingKey {
        case mlxVersion = "mlx_version"
        case turboquantPlusEnabled = "turboquant_plus_enabled"
        case turboquantKVBits = "turboquant_kv_bits"
        case turboquantSkipLast = "turboquant_skip_last"
        case ffiLoaded = "ffi_loaded"
        case perfCorePath = "perf_core"
        case backend
        case turboquantPlus = "turboquant_plus"
    }

    init(from decoder: Decoder) throws {
        // Prefer JSON when present, but fall back to text scanning via the
        // convenience initialiser `init(text:)`.
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let nested = try? c.nestedContainer(keyedBy: TurboQuantKeys.self,
                                                forKey: .turboquantPlus) {
            self.turboquantPlusEnabled = try nested.decodeIfPresent(Bool.self, forKey: .enabled)
            self.turboquantKVBits = try nested.decodeIfPresent(Int.self, forKey: .kvBits)
            self.turboquantSkipLast = try nested.decodeIfPresent(Bool.self, forKey: .skipLast)
        } else {
            self.turboquantPlusEnabled = try c.decodeIfPresent(Bool.self, forKey: .turboquantPlusEnabled)
            self.turboquantKVBits = try c.decodeIfPresent(Int.self, forKey: .turboquantKVBits)
            self.turboquantSkipLast = try c.decodeIfPresent(Bool.self, forKey: .turboquantSkipLast)
        }
        self.mlxVersion = try c.decodeIfPresent(String.self, forKey: .mlxVersion)
        self.ffiLoaded = try c.decodeIfPresent(Bool.self, forKey: .ffiLoaded)
        self.perfCorePath = try c.decodeIfPresent(String.self, forKey: .perfCorePath)
        self.backend = try c.decodeIfPresent(String.self, forKey: .backend)
    }

    enum TurboQuantKeys: String, CodingKey {
        case enabled, kvBits = "kv_bits", skipLast = "skip_last"
    }

    init() {}
}

actor StatusBridge {
    static let shared = StatusBridge()

    /// Execute `omlx-research status` and parse the result.
    func status(timeout: TimeInterval = 30) async throws -> OMLXStatus {
        let bin: URL
        do {
            bin = try await OMLXBridge.shared.locateBinary()
        } catch {
            throw error
        }

        let proc = Process()
        proc.executableURL = bin
        proc.arguments = ["status"]
        proc.environment = ProcessInfo.processInfo.environment
        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe
        proc.standardInput = FileHandle.nullDevice

        do {
            try proc.run()
        } catch {
            throw OMLXBridgeError.launchFailed(underlying: error)
        }

        // Read everything (status output is small).
        let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
        let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()

        if proc.terminationStatus != 0 {
            let s = String(data: errData, encoding: .utf8) ?? ""
            throw OMLXBridgeError.nonZeroExit(code: proc.terminationStatus, stderr: s)
        }

        return parse(text: outData)
    }

    /// Public so views can re-parse without re-running the binary.
    func parse(text data: Data) -> OMLXStatus {
        // Try JSON first.
        if let s = try? JSONDecoder().decode(OMLXStatus.self, from: data) {
            return s
        }
        // Fall back to line scanning.
        let text = String(data: data, encoding: .utf8) ?? ""
        var st = OMLXStatus()
        st.rawLines = text.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)

        for line in st.rawLines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            // "<key>: <value>" — common shape from python -m omlx_research.cli status.
            guard let colon = trimmed.firstIndex(of: ":") else { continue }
            let key = trimmed[..<colon].lowercased().trimmingCharacters(in: .whitespaces)
            let val = trimmed[trimmed.index(after: colon)...]
                .trimmingCharacters(in: .whitespaces)
            switch key {
            case "mlx_version", "mlx version", "mlx":
                st.mlxVersion = val
            case "turboquant+", "turboquant_plus", "turboquant plus":
                st.turboquantPlusEnabled = (val.lowercased() == "enabled"
                                             || val.lowercased() == "on"
                                             || val.lowercased() == "true")
            case "turboquant_kv_bits", "turboquant kv bits":
                st.turboquantKVBits = Int(val)
            case "turboquant_skip_last":
                st.turboquantSkipLast = (val.lowercased() == "true" || val.lowercased() == "yes")
            case "ffi_loaded", "ffi loaded", "ffi":
                st.ffiLoaded = (val.lowercased() == "true"
                                || val.lowercased() == "yes"
                                || val.lowercased() == "loaded")
            case "perf_core", "perf core":
                st.perfCorePath = val
            case "backend":
                st.backend = val
            default:
                continue
            }
        }
        return st
    }
}