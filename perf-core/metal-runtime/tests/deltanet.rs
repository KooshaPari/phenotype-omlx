#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matches_deltanet_reference() {
    use metal_runtime::{
        deltanet_step_metal, deltanet_step_metal_two_pass, ArtifactAllowlist, MetallibLoader,
    };
    use model_kernels::recurrent::deltanet_step;
    use sha2::{Digest, Sha256};

    let path = std::path::PathBuf::from(std::env::var("DELTANET_METALLIB").unwrap());
    let bytes = std::fs::read(&path).unwrap();
    let digest = Sha256::digest(&bytes).into();
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
    let mut metal_state = vec![0.1; 9];
    let mut reference_state = metal_state.clone();
    let got = deltanet_step_metal(&q, &k, &v, &mut metal_state, 0.5, 3, &artifact).unwrap();
    let want = deltanet_step(&q, &k, &v, &mut reference_state, 0.5, 3).unwrap();
    for (actual, expected) in got.iter().zip(want) {
        assert!((actual - expected).abs() < 1e-5);
    }
    for (actual, expected) in metal_state.iter().zip(reference_state) {
        assert!((actual - expected).abs() < 1e-5);
    }

    let mut two_pass_state = vec![0.1; 9];
    let two_pass =
        deltanet_step_metal_two_pass(&q, &k, &v, &mut two_pass_state, 0.5, 3, &artifact).unwrap();
    let mut expected_state = vec![0.1; 9];
    let expected_output = deltanet_step(&q, &k, &v, &mut expected_state, 0.5, 3).unwrap();
    for (actual, expected) in two_pass.iter().zip(expected_output) {
        assert!((actual - expected).abs() < 1e-5);
    }
    for (actual, expected) in two_pass_state.iter().zip(expected_state) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "explicit artifact-backed serial/two-pass percentile comparison"]
fn deltanet_serial_two_pass_percentiles() {
    use metal_runtime::{
        deltanet_step_metal, deltanet_step_metal_two_pass, ArtifactAllowlist, MetallibLoader,
    };
    use serde::Serialize;
    use sha2::{Digest, Sha256};
    use std::time::Instant;

    #[derive(Serialize)]
    struct Timing {
        median_us: f64,
        p95_us: f64,
    }
    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        samples: usize,
        shape: [usize; 3],
        parity_max_abs_error: f32,
        serial: Timing,
        two_pass: Timing,
        artifact_sha256: String,
    }
    fn percentile(mut samples: Vec<f64>) -> Timing {
        samples.sort_by(f64::total_cmp);
        Timing {
            median_us: samples[samples.len() / 2],
            p95_us: samples[(samples.len() - 1) * 95 / 100],
        }
    }

    let path = std::path::PathBuf::from(std::env::var("DELTANET_METALLIB").unwrap());
    let bytes = std::fs::read(&path).unwrap();
    let digest = Sha256::digest(&bytes);
    let artifact_sha256 = digest.iter().map(|b| format!("{b:02x}")).collect();
    let name = path.file_name().unwrap().to_str().unwrap().to_owned();
    let artifact = MetallibLoader::new(
        path.parent().unwrap(),
        ArtifactAllowlist::new([(name.clone(), digest.into())]),
    )
    .load(&name)
    .unwrap();

    let n = 8;
    let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17).sin()).collect();
    let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.23).cos()).collect();
    let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.31).sin()).collect();
    let samples = 21;
    for _ in 0..3 {
        let mut state = vec![0.1_f32; n * n];
        deltanet_step_metal(&q, &k, &v, &mut state, 0.5, n, &artifact).unwrap();
        let mut state = vec![0.1_f32; n * n];
        deltanet_step_metal_two_pass(&q, &k, &v, &mut state, 0.5, n, &artifact).unwrap();
    }
    let mut serial_samples = Vec::with_capacity(samples);
    let mut two_pass_samples = Vec::with_capacity(samples);
    let mut parity_max_abs_error = 0.0_f32;
    for _ in 0..samples {
        let mut serial_state = vec![0.1_f32; n * n];
        let start = Instant::now();
        let serial = deltanet_step_metal(&q, &k, &v, &mut serial_state, 0.5, n, &artifact).unwrap();
        serial_samples.push(start.elapsed().as_secs_f64() * 1e6);
        let mut two_pass_state = vec![0.1_f32; n * n];
        let start = Instant::now();
        let two_pass =
            deltanet_step_metal_two_pass(&q, &k, &v, &mut two_pass_state, 0.5, n, &artifact)
                .unwrap();
        two_pass_samples.push(start.elapsed().as_secs_f64() * 1e6);
        for (a, b) in serial.iter().zip(&two_pass) {
            parity_max_abs_error = parity_max_abs_error.max((a - b).abs());
        }
        for (a, b) in serial_state.iter().zip(two_pass_state) {
            parity_max_abs_error = parity_max_abs_error.max((a - b).abs());
        }
    }
    assert!(
        parity_max_abs_error <= 1e-5,
        "two-pass parity error {parity_max_abs_error}"
    );
    let record = Record {
        schema: "metal-runtime.deltanet-serial-two-pass.v1",
        samples,
        shape: [1, n, n],
        parity_max_abs_error,
        serial: percentile(serial_samples),
        two_pass: percentile(two_pass_samples),
        artifact_sha256,
    };
    if let Ok(output) = std::env::var("DELTANET_COMPARE_OUTPUT") {
        std::fs::write(output, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    }
    println!("{}", serde_json::to_string_pretty(&record).unwrap());
}
