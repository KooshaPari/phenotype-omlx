#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matches_mla_cache_reference() {
    use metal_runtime::{mla_cache_attend_metal, ArtifactAllowlist, MetallibLoader};
    use model_kernels::attention::{mla_cache_append, mla_cache_attend, MlaCacheEntry};
    use sha2::{Digest, Sha256};

    let path = std::path::PathBuf::from(std::env::var("MLA_CACHE_METALLIB").unwrap());
    let bytes = std::fs::read(&path).unwrap();
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    let name = path.file_name().unwrap().to_str().unwrap().to_owned();
    let artifact = MetallibLoader::new(
        path.parent().unwrap(),
        ArtifactAllowlist::new([(name.clone(), digest)]),
    )
    .load(&name)
    .unwrap();

    let q_latent = [0.4f32, -0.2, 0.7];
    let q_rope = [0.1f32, -0.3];
    let entries = vec![
        MlaCacheEntry::new(vec![0.2, 0.5, -0.1], vec![0.4, 0.2]).unwrap(),
        MlaCacheEntry::new(vec![-0.3, 0.1, 0.8], vec![-0.2, 0.6]).unwrap(),
        MlaCacheEntry::new(vec![0.9, -0.4, 0.2], vec![0.3, -0.5]).unwrap(),
    ];
    let compressed: Vec<f32> = entries
        .iter()
        .flat_map(|e| e.compressed_kv.iter().copied())
        .collect();
    let rope: Vec<f32> = entries
        .iter()
        .flat_map(|e| e.k_rope.iter().copied())
        .collect();
    let mut want = vec![0.0; 3];
    mla_cache_attend(&q_latent, &q_rope, &entries, 3, 2, &mut want).unwrap();
    let got =
        mla_cache_attend_metal(&q_latent, &q_rope, &compressed, &rope, 3, 2, &artifact).unwrap();
    for (a, b) in got.iter().zip(want) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
    let mut roundtrip = Vec::new();
    mla_cache_append(&mut roundtrip, &compressed[..3], &rope[..2]).unwrap();
    assert_eq!(roundtrip.len(), 1);
}
