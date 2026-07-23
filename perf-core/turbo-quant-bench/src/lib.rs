pub fn generate_data(n: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(n);
    let mut val: f32 = -1.0;
    let step = 2.0 / n as f32;
    for _ in 0..n {
        data.push(val.sin() * 0.8 + val.cos() * 0.3);
        val += step;
    }
    data
}
