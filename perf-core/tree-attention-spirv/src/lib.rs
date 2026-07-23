// tree-attention-spirv — Rust bridge to the Swift/Metal tree-attention kernel.
//
// On macOS with the Xcode toolchain, the Swift bridge compiles the Metal
// kernel from `metal-shaders/tree_attention.metal` and links the C ABI
// produced by `swiftc -emit-library`.
//
// Without `--features spirv`, this crate compiles to a no-op stub.

#[derive(Debug, Clone)]
pub struct TreeAttnParams {
    pub batch: usize,
    pub num_heads: usize,
    pub seq_len: usize,
    pub tree_width: usize,
    pub head_dim: usize,
}

/// Run the tree-attention forward pass.
///
/// Without feature "spirv": always returns `None` (caller should fall back
/// to the pure-Rust reference in `tree-attention`).
#[cfg(feature = "spirv")]
pub fn forward(
    params: &TreeAttnParams,
    q: &[u16],
    k: &[u16],
    v: &[u16],
    mask: &[i32],
) -> Option<Vec<u16>> {
    extern "C" {
        fn tree_attention_metal_init() -> i32;
        fn tree_attention_metal_forward(
            q: *const u16,
            k: *const u16,
            v: *const u16,
            mask: *const i32,
            out: *mut u16,
            params: *const CBridgeParams,
            len: usize,
        ) -> i32;
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CBridgeParams {
        b: u32,
        h: u32,
        t: u32,
        w: u32,
        d: u32,
    }

    use std::sync::OnceLock;
    static INIT: OnceLock<i32> = OnceLock::new();
    let rc = INIT
        .get_or_init(|| unsafe { tree_attention_metal_init() })
        .clone();
    if rc != 0 {
        return None;
    }

    let len =
        params.batch * params.num_heads * params.seq_len * params.tree_width * params.head_dim;
    let mut out = vec![0u16; len];
    let p = CBridgeParams {
        b: params.batch as u32,
        h: params.num_heads as u32,
        t: params.seq_len as u32,
        w: params.tree_width as u32,
        d: params.head_dim as u32,
    };
    let rc = unsafe {
        tree_attention_metal_forward(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            mask.as_ptr(),
            out.as_mut_ptr(),
            &p,
            len,
        )
    };
    if rc == 0 {
        Some(out)
    } else {
        None
    }
}

#[cfg(not(feature = "spirv"))]
pub fn forward(
    _params: &TreeAttnParams,
    _q: &[u16],
    _k: &[u16],
    _v: &[u16],
    _mask: &[i32],
) -> Option<Vec<u16>> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_attention::tree_causal_mask;

    #[test]
    fn metal_bridge_compiles_without_panic() {
        let p = TreeAttnParams {
            batch: 1,
            num_heads: 2,
            seq_len: 4,
            tree_width: 2,
            head_dim: 8,
        };
        let q = vec![0u16; p.batch * p.num_heads * p.seq_len * p.head_dim];
        let k = q.clone();
        let v = q.clone();
        let mask_vec = tree_causal_mask(p.seq_len, p.tree_width, 1, 0);
        let mask: Vec<i32> = mask_vec.iter().flatten().map(|&b| b as i32).collect();
        let _ = forward(&p, &q, &k, &v, &mask);
        // No panic is the win — without feature "spirv" we get None, with it
        // we get a real Metal result (or None if Metal is unavailable).
    }
}
