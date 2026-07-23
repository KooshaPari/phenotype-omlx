#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matches_zaya_cca_reference() {
    use metal_runtime::{cca_block_attend_metal, ArtifactAllowlist, MetallibLoader};
    use model_kernels::attention::{cca_block_attend, CcaBlock};
    use sha2::{Digest, Sha256};
    let path = std::path::PathBuf::from(std::env::var("CCA_METALLIB").unwrap());
    let bytes = std::fs::read(&path).unwrap();
    let digest = Sha256::digest(&bytes).into();
    let name = path.file_name().unwrap().to_str().unwrap().to_owned();
    let artifact = MetallibLoader::new(
        path.parent().unwrap(),
        ArtifactAllowlist::new([(name.clone(), digest)]),
    )
    .load(&name)
    .unwrap();
    let q = [1.0, -0.5, 0.25];
    let summaries = vec![0.5, 0.2, -0.1, -0.4, 0.7, 0.3];
    let scales = [1.0, 0.75];
    let sizes = [2u32, 3u32];
    let blocks = vec![
        CcaBlock {
            block_summary: summaries[0..3].to_vec(),
            block_summary_scale: scales[0],
            block_indices: vec![0, 1],
        },
        CcaBlock {
            block_summary: summaries[3..6].to_vec(),
            block_summary_scale: scales[1],
            block_indices: vec![2, 3, 4],
        },
    ];
    let mut want = vec![0.0; 3];
    cca_block_attend(&q, &blocks, 3, &mut want).unwrap();
    let got = cca_block_attend_metal(&q, &summaries, &scales, &sizes, 3, &artifact).unwrap();
    for (a, b) in got.iter().zip(want) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
}
