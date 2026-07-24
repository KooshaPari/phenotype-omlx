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

fn bench_rust_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust_decode");
    for &size in DATA_SIZES {
        let data = generate_data(size);
        for &bits in BIT_WIDTHS {
            let encoded = QuantizedTensor::encode_uniform(&data, bits, GROUP_SIZE);
            let id = BenchmarkId::from_parameter(format!("n={size}_b{bits}"));
            group.bench_with_input(id, &data, |b, _| {
                b.iter(|| {
                    let mut out = vec![0f32; size];
                    encoded.decode_uniform(&mut out);
                    out
                });
            });
        }
    }
    group.finish();
}

fn bench_rust_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust_roundtrip");
    for &size in DATA_SIZES {
        let data = generate_data(size);
        for &bits in BIT_WIDTHS {
            let id = BenchmarkId::from_parameter(format!("n={size}_b{bits}"));
            group.bench_with_input(id, &data, |b, d| {
                b.iter(|| {
                    let q = QuantizedTensor::encode_uniform(d, bits, GROUP_SIZE);
                    let mut out = vec![0f32; d.len()];
                    q.decode_uniform(&mut out);
                    out
                });
            });
        }
    }
    group.finish();
}

fn bench_c_decode(c: &mut Criterion) {
    use turbo_quant_c::{decode, encode};

    let mut group = c.benchmark_group("c_decode");
    for &size in DATA_SIZES {
        let data = generate_data(size);
        for &bits in BIT_WIDTHS {
            let encoded = encode(&data, bits, GROUP_SIZE).expect("C encode failed");
            let id = BenchmarkId::from_parameter(format!("n={size}_b{bits}"));
            group.bench_with_input(id, &data, |b, _| {
                b.iter(|| decode(&encoded, size, GROUP_SIZE, bits));
            });
        }
    }
    group.finish();
}

fn bench_c_roundtrip(c: &mut Criterion) {
    use turbo_quant_c::{decode, encode};

    let mut group = c.benchmark_group("c_roundtrip");
    for &size in DATA_SIZES {
        let data = generate_data(size);
        for &bits in BIT_WIDTHS {
            let id = BenchmarkId::from_parameter(format!("n={size}_b{bits}"));
            group.bench_with_input(id, &data, |b, d| {
                b.iter(|| {
                    let q = encode(d, bits, GROUP_SIZE).expect("C encode failed");
                    decode(&q, d.len(), GROUP_SIZE, bits)
                });
            });
        }
    }
    group.finish();
}

#[allow(unused_variables)]
fn bench_zig_decode(c: &mut Criterion) {
    #[cfg(feature = "zig")]
    {
        use turbo_quant_zig::ZigQuantizedTensor;

        let mut group = c.benchmark_group("zig_decode");
        for &size in DATA_SIZES {
            let data = generate_data(size);
            for &bits in BIT_WIDTHS {
                let encoded =
                    ZigQuantizedTensor::encode(&data, bits, GROUP_SIZE).expect("Zig encode failed");
                let id = BenchmarkId::from_parameter(format!("n={size}_b{bits}"));
                group.bench_with_input(id, &data, |b, _| {
                    b.iter(|| encoded.decode(size, GROUP_SIZE, bits));
                });
            }
        }
        group.finish();
    }
}

#[allow(unused_variables)]
fn bench_mojo_decode(c: &mut Criterion) {
    #[cfg(feature = "mojo")]
    {
        use turbo_quant_mojo::MojoQuantizedTensor;

        let mut group = c.benchmark_group("mojo_decode");
        for &size in DATA_SIZES {
            let data = generate_data(size);
            for &bits in BIT_WIDTHS {
                let encoded = MojoQuantizedTensor::encode(&data, bits, GROUP_SIZE)
                    .expect("Mojo encode failed");
                let id = BenchmarkId::from_parameter(format!("n={size}_b{bits}"));
                group.bench_with_input(id, &data, |b, _| {
                    b.iter(|| encoded.decode(size, GROUP_SIZE, bits));
                });
            }
        }
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_rust_decode,
    bench_rust_roundtrip,
    bench_c_decode,
    bench_c_roundtrip,
    bench_zig_decode,
    bench_mojo_decode,
);

criterion_main!(benches);
