#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matches_mamba_one_step_reference() {
    use metal_runtime::{mamba_selective_step_metal, ArtifactAllowlist, MetallibLoader};
    use model_kernels::recurrent::mamba_selective::{mamba_selective_scan, MambaSelectiveParams};
    use sha2::{Digest, Sha256};
    let path = std::path::PathBuf::from(std::env::var("MAMBA_METALLIB").unwrap());
    let bytes = std::fs::read(&path).unwrap();
    let digest = Sha256::digest(&bytes).into();
    let name = path.file_name().unwrap().to_str().unwrap().to_owned();
    let art = MetallibLoader::new(
        path.parent().unwrap(),
        ArtifactAllowlist::new([(name.clone(), digest)]),
    )
    .load(&name)
    .unwrap();
    let al = [0.0f32, -0.3, 0.2];
    let mut got = [0.1f32, -0.2, 0.3];
    let mut want = got;
    let u = 0.7;
    let dt = 0.4;
    let b = 0.8;
    let c = -0.6;
    let d = 0.2;
    let y = mamba_selective_step_metal(u, dt, b, c, d, &al, &mut got, &art).unwrap();
    let p = MambaSelectiveParams {
        dt: &[dt],
        a_log: &al,
        b: &[b],
        c: &[c],
        d: &[d],
    };
    let ys = mamba_selective_scan(&p, &[u], &mut want).unwrap();
    assert!((y - ys[0]).abs() <= 1e-5);
    for (a, b) in got.iter().zip(want) {
        assert!((a - b).abs() <= 1e-5);
    }
}
