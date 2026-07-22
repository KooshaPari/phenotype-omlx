// Pre-FFI input/output validation for the safe-Rust-to-unsafe-Mojo boundary.
//
// Every public encode/decode path must call these helpers BEFORE entering
// unsafe Mojo code. Mojo's ABI is signed-Int (isize) and any caller-controlled
// dimension crossing isize::MAX can wrap to a negative index and abort/hang.

pub(crate) const MIN_BITS: u8 = 2;
pub(crate) const MAX_BITS: u8 = 4;

/// Returns the per-group packed byte count for `group_size` × `bits`, or an
/// error if the multiplication would overflow.
pub(crate) fn per_group_packed_bytes(group_size: usize, bits: u8) -> Result<usize, String> {
    let bits_usize = bits as usize;
    let product = group_size
        .checked_mul(bits_usize)
        .ok_or_else(|| format!("group_size {group_size} * bits {bits} overflows usize"))?;
    let padded = product
        .checked_add(7)
        .ok_or_else(|| "group_size * bits + 7 overflows usize".to_string())?;
    Ok(padded / 8)
}

/// Validate `encode` inputs and return `(n, n_groups, expected_packed_len)`
/// ready to be passed (as `isize`) to the Mojo ABI.
#[allow(clippy::manual_is_multiple_of)]
pub(crate) fn validate_encode_inputs(
    data_len: usize,
    bits: u8,
    group_size: usize,
) -> Result<(usize, usize, usize), String> {
    if data_len == 0 {
        return Err("encode: data must be non-empty".to_string());
    }
    if !(MIN_BITS..=MAX_BITS).contains(&bits) {
        return Err(format!(
            "encode: bits must be in {MIN_BITS}..={MAX_BITS}, got {bits}"
        ));
    }
    if group_size == 0 {
        return Err("encode: group_size must be > 0".to_string());
    }
    // Manual modulo (workspace `rust-version = "1.74"` predates the
    // stabilized `usize::is_multiple_of` in 1.87). Safe here because
    // `group_size == 0` was rejected above.
    if data_len % group_size != 0 {
        return Err(format!(
            "encode: data length {data_len} must be divisible by group_size {group_size}"
        ));
    }

    isize::try_from(data_len)
        .map_err(|_| format!("encode: n {data_len} exceeds Mojo Int (isize) range"))?;
    isize::try_from(group_size)
        .map_err(|_| format!("encode: group_size {group_size} exceeds Mojo Int (isize) range"))?;

    let n_groups = data_len / group_size;
    isize::try_from(n_groups)
        .map_err(|_| format!("encode: n_groups {n_groups} exceeds Mojo Int (isize) range"))?;

    let per_group = per_group_packed_bytes(group_size, bits)?;
    let packed_len = n_groups.checked_mul(per_group).ok_or_else(|| {
        format!("encode: n_groups {n_groups} * per_group {per_group} overflows usize")
    })?;
    isize::try_from(packed_len)
        .map_err(|_| format!("encode: packed_len {packed_len} exceeds Mojo Int (isize) range"))?;

    Ok((data_len, n_groups, packed_len))
}

/// Validate a `decode` call's `self`-side fields plus caller dimensions and
/// return `(n, n_groups, expected_packed_len)`.
#[allow(clippy::manual_is_multiple_of)]
pub(crate) fn validate_decode_inputs(
    shape: &[usize],
    packed_len: usize,
    scales_len: usize,
    zeros_len: usize,
    n: usize,
    group_size: usize,
    bits: u8,
) -> Result<(usize, usize, usize), String> {
    if shape.len() != 1 {
        return Err(format!(
            "decode: shape must be exactly [n] (rank 1), got rank {}",
            shape.len()
        ));
    }
    let shape_n = shape[0];

    if n == 0 {
        return Err("decode: n must be > 0".to_string());
    }
    if !(MIN_BITS..=MAX_BITS).contains(&bits) {
        return Err(format!(
            "decode: bits must be in {MIN_BITS}..={MAX_BITS}, got {bits}"
        ));
    }
    if group_size == 0 {
        return Err("decode: group_size must be > 0".to_string());
    }
    // Manual modulo (workspace `rust-version = "1.74"` predates the
    // stabilized `usize::is_multiple_of` in 1.87). Safe here because
    // `group_size == 0` was rejected above.
    if n % group_size != 0 {
        return Err(format!(
            "decode: n {n} must be divisible by group_size {group_size}"
        ));
    }
    if shape_n != n {
        return Err(format!("decode: shape[0] {shape_n} does not match n {n}"));
    }

    isize::try_from(n).map_err(|_| format!("decode: n {n} exceeds Mojo Int (isize) range"))?;
    isize::try_from(group_size)
        .map_err(|_| format!("decode: group_size {group_size} exceeds Mojo Int (isize) range"))?;

    let n_groups = n / group_size;
    isize::try_from(n_groups)
        .map_err(|_| format!("decode: n_groups {n_groups} exceeds Mojo Int (isize) range"))?;

    let per_group = per_group_packed_bytes(group_size, bits)?;
    let expected_packed_len = n_groups.checked_mul(per_group).ok_or_else(|| {
        format!("decode: n_groups {n_groups} * per_group {per_group} overflows usize")
    })?;
    isize::try_from(expected_packed_len).map_err(|_| {
        format!("decode: packed_len {expected_packed_len} exceeds Mojo Int (isize) range")
    })?;

    if packed_len != expected_packed_len {
        return Err(format!(
            "decode: packed length {packed_len} does not match expected {expected_packed_len}"
        ));
    }
    if scales_len != n_groups {
        return Err(format!(
            "decode: scales length {scales_len} does not match expected n_groups {n_groups}"
        ));
    }
    if zeros_len != n_groups {
        return Err(format!(
            "decode: zeros length {zeros_len} does not match expected n_groups {n_groups}"
        ));
    }

    Ok((n, n_groups, expected_packed_len))
}

/// Validate the lengths returned by `tq_mojo_encode` against what Rust expects.
/// All four returned lengths must be non-negative `isize` and match exactly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_encode_outputs(
    shape_len: isize,
    packed_len: isize,
    scales_len: isize,
    zeros_len: isize,
    expected_shape_len: usize,
    expected_packed_len: usize,
    expected_scales_len: usize,
    expected_zeros_len: usize,
) -> Result<(usize, usize, usize, usize), String> {
    if shape_len < 0 || packed_len < 0 || scales_len < 0 || zeros_len < 0 {
        return Err(format!(
            "encode: Mojo returned negative length \
             (shape_len={shape_len}, packed_len={packed_len}, \
              scales_len={scales_len}, zeros_len={zeros_len})"
        ));
    }
    let shape_len = shape_len as usize;
    let packed_len = packed_len as usize;
    let scales_len = scales_len as usize;
    let zeros_len = zeros_len as usize;
    if shape_len != expected_shape_len {
        return Err(format!(
            "encode: returned shape_len {shape_len} != expected {expected_shape_len}"
        ));
    }
    if packed_len != expected_packed_len {
        return Err(format!(
            "encode: returned packed_len {packed_len} != expected {expected_packed_len}"
        ));
    }
    if scales_len != expected_scales_len {
        return Err(format!(
            "encode: returned scales_len {scales_len} != expected {expected_scales_len}"
        ));
    }
    if zeros_len != expected_zeros_len {
        return Err(format!(
            "encode: returned zeros_len {zeros_len} != expected {expected_zeros_len}"
        ));
    }
    Ok((shape_len, packed_len, scales_len, zeros_len))
}

/// Convert a validated non-negative `usize` into `isize` for the FFI boundary.
/// Returns an error if the value would overflow.
pub(crate) fn usize_to_isize(name: &str, value: usize) -> Result<isize, String> {
    isize::try_from(value).map_err(|_| format!("{name}: usize value {value} exceeds isize range"))
}
