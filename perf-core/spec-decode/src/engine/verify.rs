//! Verification helpers — deterministic verification pass and test targets.

use super::SpecDecodeEngine;
use crate::verify::{verify as verify_draft, VerifyResult};
use crate::SpecError;
impl SpecDecodeEngine {
    /// Run a deterministic verification pass against the target's logits
    /// without performing any draft step — exposed so external callers
    /// (Python, FFI) can plug in custom draft proposals.
    pub fn verify_only(
        &self,
        target_logits: &[f32],
        draft_tokens: &[u32],
        draft_probs: &[f32],
    ) -> Result<VerifyResult, SpecError> {
        verify_draft(target_logits, draft_tokens, draft_probs, &self.config)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::{BackendInfo, TargetBackend, TargetOutput};
    use crate::{SpecDecodeConfig, SpecDecodeEngine};
    use async_trait::async_trait;

    struct ConstantTarget(u32);
    #[async_trait]
    impl TargetBackend for ConstantTarget {
        async fn forward(&self, _: &[u32]) -> Result<TargetOutput, String> {
            let mut logits = vec![0.0_f32; 64];
            logits[self.0 as usize] = 10.0;
            Ok(TargetOutput {
                logits,
                hidden: None,
                finished: false,
            })
        }
        async fn verify_tree(
            &self,
            _: &[u32],
            candidates: &[Vec<u32>],
        ) -> Result<Vec<bool>, String> {
            Ok(candidates
                .iter()
                .map(|c| c.first().copied() == Some(self.0))
                .collect())
        }
        fn info(&self) -> BackendInfo {
            BackendInfo {
                engine: "test".into(),
                model_id: "constant".into(),
                device: "cpu".into(),
                dtype: "f32".into(),
                kv_cache_type: None,
            }
        }
    }

    #[test]
    fn verify_only_is_deterministic() {
        let e = SpecDecodeEngine::new(
            SpecDecodeConfig::default(),
            Box::new(ConstantTarget(5)),
            None,
        );
        let mut logits = vec![0.0_f32; 16];
        logits[5] = 10.0;
        let r = e.verify_only(&logits, &[5_u32, 5, 5], &[1.0; 16]).unwrap();
        assert_eq!(r.accepted_prefix.len(), 3);
    }
}
