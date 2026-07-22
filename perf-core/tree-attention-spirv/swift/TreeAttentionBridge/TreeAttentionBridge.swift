// Swift bridge: Metal tree-attention kernel from Rust.
// API mirrors the Rust tree-attention crate exactly.

import Foundation
import Metal

public struct TreeAttnParams {
    public let B: UInt32; public let H: UInt32; public let T: UInt32
    public let W: UInt32; public let D: UInt32
    public init(B: UInt32, H: UInt32, T: UInt32, W: UInt32, D: UInt32) {
        self.B = B; self.H = H; self.T = T; self.W = W; self.D = D
    }
}

public enum TreeAttentionError: Error {
    case noMetalDevice, kernelNotFound, bufferCreationFailed
}

public final class TreeAttentionBridge {
    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue

    public init(library: MTLLibrary) throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw TreeAttentionError.noMetalDevice }
        guard let fn = library.makeFunction(name: "tree_attention_fwd") else { throw TreeAttentionError.kernelNotFound }
        guard let q = dev.makeCommandQueue() else { throw TreeAttentionError.bufferCreationFailed }
        self.device = dev
        self.pipeline = try dev.makeComputePipelineState(function: fn)
        self.queue = q
    }

    public func forward(
        Q: Data, K: Data, V: Data, treeMask: Data, params: TreeAttnParams
    ) throws -> Data {
        let BHTD = Int(params.B * params.H * params.T * params.D)
        let TW   = Int(params.T * params.W)
        guard let qBuf = device.makeBuffer(bytes: (Q as NSData).bytes, length: BHTD * 2, options: .storageModeShared),
              let kBuf = device.makeBuffer(bytes: (K as NSData).bytes, length: BHTD * 2, options: .storageModeShared),
              let vBuf = device.makeBuffer(bytes: (V as NSData).bytes, length: BHTD * 2, options: .storageModeShared),
              let mBuf = device.makeBuffer(bytes: (treeMask as NSData).bytes, length: TW * 4, options: .storageModeShared),
              let oBuf = device.makeBuffer(length: BHTD * 2, options: .storageModeShared)
        else { throw TreeAttentionError.bufferCreationFailed }

        var p = params
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { throw TreeAttentionError.bufferCreationFailed }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kBuf, offset: 0, index: 1)
        enc.setBuffer(vBuf, offset: 0, index: 2)
        enc.setBuffer(mBuf, offset: 0, index: 3)
        enc.setBuffer(oBuf, offset: 0, index: 4)
        enc.setBytes(&p, length: MemoryLayout<TreeAttnParams>.size, index: 5)
        let grid = MTLSize(width: Int(params.B), height: Int(params.H), depth: Int(params.T))
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return Data(bytesNoCopy: oBuf.contents(), count: BHTD * 2, deallocator: .none)
    }
}
