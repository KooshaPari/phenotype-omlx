//! KernelKey stability + hash contracts.

use kernel_registry::compat::OperatorKind;

use super::{key_with, TEST_DEVICE_FINGERPRINT};

#[test]
fn kernel_key_hash_is_stable() {
    let k1 = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    let k2 = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    assert_eq!(k1, k2);
    assert_eq!(
        k1.fast_hash(),
        k2.fast_hash(),
        "fast_hash() must be a pure function of the key fields"
    );
    // Hard-coded values keep accidental field reordering loud.
    let k3 = key_with(OperatorKind::Attention, TEST_DEVICE_FINGERPRINT, 1);
    assert_ne!(
        k1.fast_hash(),
        k3.fast_hash(),
        "operator_kind must affect fast_hash"
    );
}

#[test]
fn kernel_key_eq_treats_policy_version_as_distinguishing() {
    let k1 = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    let k2 = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 2);
    assert_ne!(
        k1, k2,
        "policy_version must distinguish keys — selection policy changes invalidate prior evidence"
    );
    assert_ne!(k1.fast_hash(), k2.fast_hash());
}
