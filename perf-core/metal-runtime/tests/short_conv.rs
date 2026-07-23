#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matches_lfm_short_conv_reference() {
    use metal_runtime::{short_conv1d_step_metal, ArtifactAllowlist, MetallibLoader};
    use model_kernels::recurrent::short_conv1d_step;
    use sha2::{Digest, Sha256};

    let path = std::path::PathBuf::from(
        std::env::var("SHORT_CONV_METALLIB").expect("SHORT_CONV_METALLIB test artifact"),
    );
    let bytes = std::fs::read(&path).expect("read short-conv metallib");
    let digest = Sha256::digest(&bytes).into();
    let name = path.file_name().unwrap().to_str().unwrap().to_owned();
    let artifact = MetallibLoader::new(
        path.parent().unwrap(),
        ArtifactAllowlist::new([(name.clone(), digest)]),
    )
    .load(&name)
    .expect("verified short-conv artifact");

    let kernel = [1.0f32, 0.5, -0.25];
    let inputs = [1.0f32, 2.0, -1.0, 3.0, -0.5];
    let mut gpu_state = vec![0.0f32; kernel.len() - 1];
    let mut cpu_state = vec![0.0f32; kernel.len() - 1];
    for x in inputs {
        let got = short_conv1d_step_metal(x, &kernel, &mut gpu_state, &artifact).unwrap();
        let want = short_conv1d_step(&[x], &kernel, &mut cpu_state).unwrap();
        assert!((got - want).abs() <= 1e-5, "got {got}, want {want}");
        assert_eq!(gpu_state, cpu_state);
    }
}
