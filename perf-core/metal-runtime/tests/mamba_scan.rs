#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matches_chunked_mamba_reference() {
    use metal_runtime::{mamba_selective_scan_metal, ArtifactAllowlist, MetallibLoader};
    use model_kernels::recurrent::mamba_selective::{mamba_selective_scan, MambaSelectiveParams};
    use sha2::{Digest, Sha256};
    let path = std::path::PathBuf::from(std::env::var("MAMBA_SCAN_METALLIB").unwrap());
    let digest = Sha256::digest(std::fs::read(&path).unwrap()).into();
    let name = path.file_name().unwrap().to_str().unwrap().to_owned();
    let artifact = MetallibLoader::new(
        path.parent().unwrap(),
        ArtifactAllowlist::new([(name.clone(), digest)]),
    )
    .load(&name)
    .unwrap();
    let u = [0.7_f32, -0.2, 0.4, 1.1];
    let dt = [0.4_f32, 0.2, 0.3, 0.1];
    let b = [0.8_f32, 0.5, 0.7, 0.2];
    let c = [-0.6_f32, -0.2, 0.4, 0.3];
    let d = [0.2_f32, 0.1, 0.0, -0.1];
    let a_log = [0.0_f32, -0.3, 0.2];
    let mut got_state = vec![0.1_f32, -0.2, 0.3];
    let mut want_state = got_state.clone();
    let got =
        mamba_selective_scan_metal(&u, &dt, &b, &c, &d, &a_log, &mut got_state, &artifact).unwrap();
    let params = MambaSelectiveParams {
        dt: &dt,
        a_log: &a_log,
        b: &b,
        c: &c,
        d: &d,
    };
    let want = mamba_selective_scan(&params, &u, &mut want_state).unwrap();
    for (a, b) in got.iter().zip(want) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
    for (a, b) in got_state.iter().zip(want_state) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }

    let mut split_state = vec![0.1_f32, -0.2, 0.3];
    let first = mamba_selective_scan_metal(
        &u[..2],
        &dt[..2],
        &b[..2],
        &c[..2],
        &d[..2],
        &a_log,
        &mut split_state,
        &artifact,
    )
    .unwrap();
    let second = mamba_selective_scan_metal(
        &u[2..],
        &dt[2..],
        &b[2..],
        &c[2..],
        &d[2..],
        &a_log,
        &mut split_state,
        &artifact,
    )
    .unwrap();
    assert_eq!(first.len() + second.len(), got.len());
    for (a, b) in first.iter().chain(second.iter()).zip(got.iter()) {
        assert!((a - b).abs() < 1e-5, "split {a} != fused {b}");
    }
    for (a, b) in split_state.iter().zip(got_state.iter()) {
        assert!((a - b).abs() < 1e-5, "split state {a} != fused {b}");
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn parallel_channel_boundaries_match_reference() {
    use metal_runtime::{mamba_selective_scan_metal, ArtifactAllowlist, MetallibLoader};
    use model_kernels::recurrent::mamba_selective::{mamba_selective_scan, MambaSelectiveParams};
    use sha2::{Digest, Sha256};
    let path = std::path::PathBuf::from(std::env::var("MAMBA_SCAN_METALLIB").unwrap());
    let digest = Sha256::digest(std::fs::read(&path).unwrap()).into();
    let name = path.file_name().unwrap().to_str().unwrap().to_owned();
    let artifact = MetallibLoader::new(
        path.parent().unwrap(),
        ArtifactAllowlist::new([(name.clone(), digest)]),
    )
    .load(&name)
    .unwrap();
    for &state_dim in &[1_usize, 255, 256, 257] {
        let steps = 5;
        let u: Vec<f32> = (0..steps).map(|i| (i as f32 * 0.37).sin()).collect();
        let dt = vec![0.1_f32; steps];
        let b = vec![0.3_f32; steps];
        let c = vec![0.2_f32; steps];
        let d = vec![0.05_f32; steps];
        let a_log: Vec<f32> = (0..state_dim).map(|i| -0.01 * (i as f32 + 1.0)).collect();
        let mut got_state = vec![0.01_f32; state_dim];
        let mut want_state = got_state.clone();
        let got =
            mamba_selective_scan_metal(&u, &dt, &b, &c, &d, &a_log, &mut got_state, &artifact)
                .unwrap();
        let params = MambaSelectiveParams {
            dt: &dt,
            a_log: &a_log,
            b: &b,
            c: &c,
            d: &d,
        };
        let want = mamba_selective_scan(&params, &u, &mut want_state).unwrap();
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() < 1e-4, "dim={state_dim}: {a} != {b}");
        }
        for (a, b) in got_state.iter().zip(want_state) {
            assert!((a - b).abs() < 1e-4, "dim={state_dim}: {a} != {b}");
        }
    }
}

#[test]
fn malformed_scan_shapes_fail_before_dispatch() {
    use metal_runtime::{validate_scan_shapes, MambaScanError};
    assert_eq!(
        validate_scan_shapes(&[], &[], &[], &[], &[], &[0.0], &[0.0]),
        Err(MambaScanError::BadShape)
    );
    assert_eq!(
        validate_scan_shapes(&[1.0], &[], &[1.0], &[1.0], &[1.0], &[0.0], &[0.0]),
        Err(MambaScanError::BadShape)
    );
    assert_eq!(
        validate_scan_shapes(&[1.0], &[0.1], &[1.0], &[1.0], &[1.0], &[0.0], &[0.0, 0.0]),
        Err(MambaScanError::BadShape)
    );
}
