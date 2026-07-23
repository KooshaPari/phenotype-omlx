#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matches_retnet_retention_reference() {
    use metal_runtime::{retnet_retention_step_metal, ArtifactAllowlist, MetallibLoader};
    use sha2::{Digest, Sha256};
    let path = std::path::PathBuf::from(std::env::var("RETNET_METALLIB").unwrap());
    let digest = Sha256::digest(std::fs::read(&path).unwrap()).into();
    let name = path.file_name().unwrap().to_str().unwrap().to_owned();
    let artifact = MetallibLoader::new(
        path.parent().unwrap(),
        ArtifactAllowlist::new([(name.clone(), digest)]),
    )
    .load(&name)
    .unwrap();
    let q = [1.0, 2.0, -1.0];
    let k = [0.5, -0.25, 1.0];
    let v = [1.0, -0.5, 0.25];
    let mut got_state = vec![0.1; 9];
    let mut want_state = got_state.clone();
    let decay = 0.8;
    let got = retnet_retention_step_metal(&q, &k, &v, &mut got_state, decay, 3, &artifact).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            want_state[i * 3 + j] = decay * want_state[i * 3 + j] + k[i] * v[j];
        }
    }
    let want: Vec<f32> = (0..3)
        .map(|j| (0..3).map(|i| q[i] * want_state[i * 3 + j]).sum())
        .collect();
    for (a, b) in got.iter().zip(want) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
    for (a, b) in got_state.iter().zip(want_state) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
}
