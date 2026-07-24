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

fn bytes_for_size(n: usize, bits: u8) -> usize {
    (n * bits as usize).div_ceil(8) + (n / GROUP_SIZE) * 8
}

fn bench_rust_energy_proxy(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust_energy_proxy");
    for &size in DATA_SIZES {
        let data = generate_data(size);
        for &bits in BIT_WIDTHS {
            let bytes = bytes_for_size(size, bits);
            let id = BenchmarkId::from_parameter(format!("n={size}_b{bits}_dispatch={bytes}B"));
            group.bench_with_input(id, &data, |b, d| {
                b.iter(|| {
                    let q = QuantizedTensor::encode_uniform(d, bits, GROUP_SIZE);
                    let mut out = vec![0f32; d.len()];
                    q.decode_uniform(&mut out);
                    (q.packed.len(), q.scales.len(), q.zeros.len())
                });
            });
        }
    }
    group.finish();
}

fn bench_c_energy_proxy(c: &mut Criterion) {
    use turbo_quant_c::{decode, encode};

    let mut group = c.benchmark_group("c_energy_proxy");
    for &size in DATA_SIZES {
        let data = generate_data(size);
        for &bits in BIT_WIDTHS {
            let bytes = bytes_for_size(size, bits);
            let id = BenchmarkId::from_parameter(format!("n={size}_b{bits}_dispatch={bytes}B"));
            group.bench_with_input(id, &data, |b, d| {
                b.iter(|| {
                    let q = encode(d, bits, GROUP_SIZE).expect("C encode failed");
                    let _out = decode(&q, d.len(), GROUP_SIZE, bits);
                    (q.packed.len(), q.scales.len(), q.zeros.len())
                });
            });
        }
    }
    group.finish();
}

#[allow(unused_variables)]
fn bench_zig_energy_proxy(c: &mut Criterion) {
    #[cfg(feature = "zig")]
    {
        use turbo_quant_zig::ZigQuantizedTensor;

        let mut group = c.benchmark_group("zig_energy_proxy");
        for &size in DATA_SIZES {
            let data = generate_data(size);
            for &bits in BIT_WIDTHS {
                let bytes = bytes_for_size(size, bits);
                let id = BenchmarkId::from_parameter(format!("n={size}_b{bits}_dispatch={bytes}B"));
                group.bench_with_input(id, &data, |b, d| {
                    b.iter(|| {
                        let q = ZigQuantizedTensor::encode(d, bits, GROUP_SIZE)
                            .expect("Zig encode failed");
                        let _out = q.decode(d.len(), GROUP_SIZE, bits);
                        (q.packed.len(), q.scales.len(), q.zeros.len())
                    });
                });
            }
        }
        group.finish();
    }
}

#[allow(unused_variables)]
fn bench_mojo_energy_proxy(c: &mut Criterion) {
    #[cfg(feature = "mojo")]
    {
        use turbo_quant_mojo::MojoQuantizedTensor;

        let mut group = c.benchmark_group("mojo_energy_proxy");
        for &size in DATA_SIZES {
            let data = generate_data(size);
            for &bits in BIT_WIDTHS {
                let bytes = bytes_for_size(size, bits);
                let id = BenchmarkId::from_parameter(format!("n={size}_b{bits}_dispatch={bytes}B"));
                group.bench_with_input(id, &data, |b, d| {
                    b.iter(|| {
                        let q = MojoQuantizedTensor::encode(d, bits, GROUP_SIZE)
                            .expect("Mojo encode failed");
                        let _out = q.decode(d.len(), GROUP_SIZE, bits);
                        (q.packed.len(), q.scales.len(), q.zeros.len())
                    });
                });
            }
        }
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_rust_energy_proxy,
    bench_c_energy_proxy,
    bench_zig_energy_proxy,
    bench_mojo_energy_proxy,
);

criterion_main!(benches);
