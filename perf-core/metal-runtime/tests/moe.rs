use metal_runtime::{MoeRouter, MoeRouterError, MoeShape};
use model_kernels::moe::router_topk;

fn assert_matches_reference(logits: &[f32], shape: MoeShape, top_k: usize) {
    let router = MoeRouter::new(shape, top_k).expect("valid router");
    let output = router.route_reference(logits).expect("route");
    assert_eq!(output.expert_ids.len(), shape.tokens * top_k);
    assert_eq!(output.weights.len(), shape.tokens * top_k);

    for token in 0..shape.tokens {
        let start = token * shape.experts;
        let expected = router_topk(
            &logits[start..start + shape.experts],
            shape.experts,
            top_k,
            0,
        )
        .unwrap();
        let out_start = token * top_k;
        for (rank, expected_item) in expected.iter().enumerate().take(top_k) {
            assert_eq!(
                output.expert_ids[out_start + rank],
                expected_item.0 as u32,
                "token {token}, rank {rank}"
            );
            assert!(
                (output.weights[out_start + rank] - expected_item.1).abs() <= 1.0e-6,
                "token {token}, rank {rank}: got {}, expected {}",
                output.weights[out_start + rank],
                expected_item.1
            );
        }
    }
}

#[test]
fn reference_matches_model_kernels_for_all_supported_top_k_values() {
    for top_k in [1, 2, 4, 8] {
        let shape = MoeShape {
            tokens: 3,
            experts: 8,
        };
        let logits: Vec<f32> = (0..shape.tokens * shape.experts)
            .map(|i| ((i * 17 + 3) % 23) as f32 * 0.25 - 2.0)
            .collect();
        assert_matches_reference(&logits, shape, top_k);
    }
}

#[test]
fn stable_ties_choose_lower_expert_ids() {
    let shape = MoeShape {
        tokens: 1,
        experts: 8,
    };
    let router = MoeRouter::new(shape, 4).unwrap();
    let output = router.route_reference(&[1.0; 8]).unwrap();
    assert_eq!(output.expert_ids, [0, 1, 2, 3]);
    for weight in output.weights {
        assert!((weight - 0.25).abs() <= 1.0e-6);
    }
}

#[test]
fn rejects_bad_shapes_and_unsupported_top_k() {
    assert!(matches!(
        MoeRouter::new(
            MoeShape {
                tokens: 0,
                experts: 8
            },
            2
        ),
        Err(MoeRouterError::ZeroDimension {
            dimension: "tokens"
        })
    ));
    assert!(matches!(
        MoeRouter::new(
            MoeShape {
                tokens: 1,
                experts: 65
            },
            2
        ),
        Err(MoeRouterError::TooManyExperts { experts: 65 })
    ));
    assert!(matches!(
        MoeRouter::new(
            MoeShape {
                tokens: 1,
                experts: 8
            },
            3
        ),
        Err(MoeRouterError::UnsupportedTopK { top_k: 3 })
    ));
    assert!(matches!(
        MoeRouter::new(
            MoeShape {
                tokens: 1,
                experts: 4
            },
            8
        ),
        Err(MoeRouterError::TopKExceedsExperts {
            top_k: 8,
            experts: 4
        })
    ));
}

#[test]
fn rejects_bad_logit_length_and_nonfinite_values() {
    let router = MoeRouter::new(
        MoeShape {
            tokens: 2,
            experts: 4,
        },
        2,
    )
    .unwrap();
    assert!(matches!(
        router.route_reference(&[0.0; 7]),
        Err(MoeRouterError::BadLogitLength {
            expected: 8,
            got: 7
        })
    ));
    let mut logits = [0.0; 8];
    logits[5] = f32::NAN;
    assert!(matches!(
        router.route_reference(&logits),
        Err(MoeRouterError::NonFiniteLogit { index: 5 })
    ));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matches_reference_ids_and_weights() {
    use metal_runtime::{ArtifactAllowlist, MetallibLoader};
    use sha2::{Digest, Sha256};

    let path = std::env::var("MOE_TOPK_METALLIB").expect("MOE_TOPK_METALLIB test artifact");
    let path = std::path::PathBuf::from(path);
    let bytes = std::fs::read(&path).expect("read precompiled test metallib");
    let digest = Sha256::digest(&bytes).into();
    let root = path.parent().expect("metallib parent");
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .expect("metallib basename");
    let artifact = MetallibLoader::new(root, ArtifactAllowlist::new([(name.to_owned(), digest)]))
        .load(name)
        .expect("verified precompiled test metallib");
    let shape = MoeShape {
        tokens: 4,
        experts: 64,
    };
    let logits: Vec<f32> = (0..shape.tokens * shape.experts)
        .map(|i| ((i * 29 + 11) % 97) as f32 * 0.125 - 5.0)
        .collect();
    for top_k in [1, 2, 4, 8] {
        let router = MoeRouter::new(shape, top_k).unwrap();
        let expected = router.route_reference(&logits).unwrap();
        let actual = router.route_metal(&logits, &artifact).unwrap();
        assert_eq!(actual.expert_ids, expected.expert_ids);
        for (got, want) in actual.weights.iter().zip(&expected.weights) {
            assert!((got - want).abs() <= 2.0e-5, "got {got}, expected {want}");
        }
    }
}

#[test]
fn production_router_source_contains_no_runtime_source_compilation() {
    let source = include_str!("../src/moe.rs");
    assert!(!source.contains("new_library_with_source"));
    assert!(source.contains("new_library_with_data"));
}
