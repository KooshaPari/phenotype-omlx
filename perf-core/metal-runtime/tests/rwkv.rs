#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matches_rwkv7_reference() {
    use metal_runtime::{rwkv7_time_mix_metal, ArtifactAllowlist, MetallibLoader};
    use model_kernels::recurrent::rwkv7_time_mix;
    use sha2::{Digest, Sha256};
    let path = std::path::PathBuf::from(std::env::var("RWKV_METALLIB").unwrap());
    let bytes = std::fs::read(&path).unwrap();
    let digest = Sha256::digest(&bytes).into();
    let name = path.file_name().unwrap().to_str().unwrap().to_owned();
    let artifact = MetallibLoader::new(
        path.parent().unwrap(),
        ArtifactAllowlist::new([(name.clone(), digest)]),
    )
    .load(&name)
    .unwrap();
    let xs = [
        [1.0, 0.5, -0.25, 2.0],
        [0.0, 1.0, 0.5, -0.5],
        [-1.0, 0.25, 1.0, 0.0],
    ];
    let mut got_state = [0.0; 4];
    let mut want_state = [0.0; 4];
    for x in xs {
        let got =
            rwkv7_time_mix_metal(&x, &mut got_state, 0.5, 0.25, 0.75, 0.4, 0.9, &artifact).unwrap();
        let want = rwkv7_time_mix(&x, &mut want_state, 0.5, 0.25, 0.75, 0.4, 0.9).unwrap();
        assert!((got - want).abs() <= 1e-5, "got {got}, want {want}");
        for (a, b) in got_state.iter().zip(want_state) {
            assert!((a - b).abs() <= 1e-5);
        }
    }
}
