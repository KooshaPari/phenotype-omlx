use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use turbo_quant::QuantizedTensor;

const DATA_SIZES: &[usize] = &[1024, 4096, 16384, 65536];
const BIT_WIDTHS: &[u8] = &[2, 3, 4];
const GROUP_SIZE: usize = 64;

fn generate_data(n: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(n);
    let mut val: f32 = -1.0;
    let step = 2.0 / n as f32;
    for _ in 0..n {
        data.push(val.sin() * 0.8 + val.cos() * 0.3);
        val += step;
    }
    data
}

fn bench_rust_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust_encode");
    for &size in DATA_SIZES {
        let data = generate_data(size);
        for &bits in BIT_WIDTHS {
            let id = BenchmarkId::from_parameter(format!("n={size}_b{bits}"));
            group.bench_with_input(id, &data, |b, d| {
                b.iter(|| QuantizedTensor::encode_uniform(d, bits, GROUP_SIZE));
            });
        }
    }
    group.finish();
}

fn bench_c_encode(c: &mut Criterion) {
    use turbo_quant_c::encode;

    let mut group = c.benchmark_group("c_encode");
    for &size in DATA_SIZES {
        let data = generate_data(size);
        for &bits in BIT_WIDTHS {
            let id = BenchmarkId::from_parameter(format!("n={size}_b{bits}"));
            group.bench_with_input(id, &data, |b, d| {
                b.iter(|| encode(d, bits, GROUP_SIZE).expect("C encode failed"));
            });
        }
    }
    group.finish();
}

#[allow(unused_variables)]
fn bench_zig_encode(c: &mut Criterion) {
    #[cfg(feature = "zig")]
    {
        use turbo_quant_zig::ZigQuantizedTensor;

        let mut group = c.benchmark_group("zig_encode");
        for &size in DATA_SIZES {
            let data = generate_data(size);
            for &bits in BIT_WIDTHS {
                let id = BenchmarkId::from_parameter(format!("n={size}_b{bits}"));
                group.bench_with_input(id, &data, |b, d| {
                    b.iter(|| {
                        ZigQuantizedTensor::encode(d, bits, GROUP_SIZE).expect("Zig encode failed")
                    });
                });
            }
        }
        group.finish();
    }
}

#[allow(unused_variables)]
fn bench_mojo_encode(c: &mut Criterion) {
    #[cfg(feature = "mojo")]
    {
        use turbo_quant_mojo::MojoQuantizedTensor;

        let mut group = c.benchmark_group("mojo_encode");
        for &size in DATA_SIZES {
            let data = generate_data(size);
            for &bits in BIT_WIDTHS {
                let id = BenchmarkId::from_parameter(format!("n={size}_b{bits}"));
                group.bench_with_input(id, &data, |b, d| {
                    b.iter(|| {
                        MojoQuantizedTensor::encode(d, bits, GROUP_SIZE)
                            .expect("Mojo encode failed")
                    });
                });
            }
        }
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_rust_encode,
    bench_c_encode,
    bench_zig_encode,
    bench_mojo_encode,
);

criterion_main!(benches);
