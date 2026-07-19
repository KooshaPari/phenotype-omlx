#include "turbo_quant.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ----- internal helpers ------------------------------------------------- */

static inline uint8_t tq_bits_mask(uint8_t bits) {
    return (uint8_t)((1u << bits) - 1u);
}

static inline bool tq_bits_valid(uint8_t bits) {
    return bits >= 2 && bits <= 4;
}

/* Initialise every out-pointer/length the ABI exposes. Called up front so
 * that, even when validation fails partway, callers see all outputs cleared
 * to NULL/0 rather than whatever stale value they happened to hand in. */
static void tq_clear_outputs(
    size_t** out_shape, size_t* out_shape_len,
    uint8_t** out_packed, size_t* out_packed_len,
    float** out_scales, size_t* out_scales_len,
    float** out_zeros, size_t* out_zeros_len) {
    if (out_shape)     *out_shape     = NULL;
    if (out_shape_len) *out_shape_len = 0;
    if (out_packed)    *out_packed    = NULL;
    if (out_packed_len)*out_packed_len= 0;
    if (out_scales)    *out_scales    = NULL;
    if (out_scales_len)*out_scales_len= 0;
    if (out_zeros)     *out_zeros     = NULL;
    if (out_zeros_len) *out_zeros_len = 0;
}

/* Write `bits` low bits of `value` into `packed` starting at absolute
 * bit offset `bit_offset` (LSB-first within each byte). Correctly straddles
 * byte boundaries so 2/3/4-bit values pack contiguously without losing
 * high bits. The caller must guarantee the destination range is in bounds
 * and that the buffer was zero-initialised. */
static void tq_write_bits(uint8_t* packed, size_t bit_offset,
                          uint8_t value, uint8_t bits) {
    uint8_t mask = tq_bits_mask(bits);
    value = (uint8_t)(value & mask);

    size_t byte_idx     = bit_offset >> 3;
    size_t bit_in_byte  = bit_offset & 7u;
    size_t room         = 8u - bit_in_byte;

    if (room >= bits) {
        /* All bits fit in the current byte. */
        uint8_t shift_mask = (uint8_t)(mask << bit_in_byte);
        packed[byte_idx] = (uint8_t)((packed[byte_idx] & (uint8_t)~shift_mask)
                                     | (uint8_t)(value << bit_in_byte));
    } else {
        /* Low bits fill the remainder of `byte_idx`; high bits spill into
         * the next byte. */
        uint8_t lo_mask = (uint8_t)((1u << room) - 1u);
        packed[byte_idx] = (uint8_t)((packed[byte_idx] & (uint8_t)~(lo_mask << bit_in_byte))
                                     | (uint8_t)((value & lo_mask) << bit_in_byte));
        size_t hi_bits = bits - room;
        uint8_t hi_mask = (uint8_t)((1u << hi_bits) - 1u);
        packed[byte_idx + 1] = (uint8_t)((packed[byte_idx + 1] & (uint8_t)~hi_mask)
                                         | (uint8_t)((value >> room) & hi_mask));
    }
}

/* Symmetric reader for the packing produced by tq_write_bits. */
static uint8_t tq_read_bits(const uint8_t* packed, size_t bit_offset,
                            uint8_t bits) {
    uint8_t mask = tq_bits_mask(bits);

    size_t byte_idx    = bit_offset >> 3;
    size_t bit_in_byte = bit_offset & 7u;
    size_t room        = 8u - bit_in_byte;

    if (room >= bits) {
        return (uint8_t)((packed[byte_idx] >> bit_in_byte) & mask);
    }

    uint8_t lo_mask = (uint8_t)((1u << room) - 1u);
    uint8_t lo = (uint8_t)((packed[byte_idx] >> bit_in_byte) & lo_mask);
    size_t  hi_bits = bits - room;
    uint8_t hi_mask = (uint8_t)((1u << hi_bits) - 1u);
    uint8_t hi = (uint8_t)(packed[byte_idx + 1] & hi_mask);
    return (uint8_t)(lo | (uint8_t)(hi << room));
}

/* ----- public ABI ------------------------------------------------------- */

bool tq_c_encode(const float* data, size_t n, uint8_t bits, size_t group_size,
    size_t** out_shape, size_t* out_shape_len,
    uint8_t** out_packed, size_t* out_packed_len,
    float** out_scales, size_t* out_scales_len,
    float** out_zeros, size_t* out_zeros_len) {
    /* Always initialise every output so callers see cleared state on
     * failure too. */
    tq_clear_outputs(out_shape, out_shape_len,
                     out_packed, out_packed_len,
                     out_scales, out_scales_len,
                     out_zeros, out_zeros_len);

    /* Required output slots must be present even if validation fails, so
     * we always have a place to write NULL/0. The pointer-to-pointer
     * indirection itself is required by the contract. */
    if (!out_shape || !out_shape_len || !out_packed || !out_packed_len ||
        !out_scales || !out_scales_len || !out_zeros || !out_zeros_len) {
        return false;
    }

    /* Input validation. */
    if (data == NULL)        return false;
    if (n == 0)              return false;
    if (!tq_bits_valid(bits)) return false;
    if (group_size == 0)     return false;

    /* `g * group_size` for the last group must not overflow, and
     * `start + group_size` likewise. Both are bounded by `n + group_size`. */
    if (n > SIZE_MAX - group_size) return false;

    /* `n * bits` overflow check before computing packed length. */
    if ((size_t)bits > 0 && n > SIZE_MAX / (size_t)bits) return false;
    size_t packed_bits  = n * (size_t)bits;
    size_t packed_bytes = (packed_bits + 7u) / 8u;
    if (packed_bytes == 0) packed_bytes = 1; /* keep calloc(.,1) well-defined */

    /* NaN / Inf inputs are unsafe for affine quantisation: reject early. */
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(data[i])) return false;
    }

    size_t n_groups = (n + group_size - 1u) / group_size;

    /* Allocate everything up front so any failure cleanly frees the rest. */
    size_t* shape = (size_t*)malloc(sizeof(size_t));
    if (!shape) return false;
    shape[0] = n;

    float* scales = (float*)calloc(n_groups, sizeof(float));
    if (!scales) { free(shape); return false; }

    float* zeros = (float*)calloc(n_groups, sizeof(float));
    if (!zeros) { free(shape); free(scales); return false; }

    uint8_t* packed = (uint8_t*)calloc(packed_bytes, 1u);
    if (!packed) { free(shape); free(scales); free(zeros); return false; }

    /* Per-group affine quantisation. The packed bitstream is contiguous
     * across groups: positions [g*group_size + local] * bits in the buffer. */
    uint32_t levels = (uint32_t)((1u << bits) - 1u);

    for (size_t g = 0; g < n_groups; g++) {
        size_t start = g * group_size;
        size_t end   = start + group_size;
        if (end > n) end = n;

        float gmin = data[start];
        float gmax = data[start];
        for (size_t i = start + 1; i < end; i++) {
            if (data[i] < gmin) gmin = data[i];
            if (data[i] > gmax) gmax = data[i];
        }

        float span  = gmax - gmin;
        float scale = span / (float)levels;
        /* Reject degenerate spans AND NaN produced by Inf - Inf. */
        if (!(scale > 0.0f)) scale = 1e-30f;

        scales[g] = scale;
        zeros[g]  = gmin;

        for (size_t i = start; i < end; i++) {
            float qf = (data[i] - gmin) / scale;
            if (!(qf >= 0.0f)) qf = 0.0f; /* NaN-safe clamp */
            uint32_t q = (uint32_t)(qf + 0.5f);
            if (q > levels) q = levels;
            size_t bit_off = (g * group_size + (i - start)) * (size_t)bits;
            tq_write_bits(packed, bit_off, (uint8_t)q, bits);
        }
    }

    *out_shape       = shape;
    *out_shape_len   = 1;
    *out_packed      = packed;
    *out_packed_len  = packed_bytes;
    *out_scales      = scales;
    *out_scales_len  = n_groups;
    *out_zeros       = zeros;
    *out_zeros_len   = n_groups;
    return true;
}

void tq_c_decode(const uint8_t* packed, size_t packed_len,
    const float* scales, const float* zeros,
    size_t n, size_t group_size, uint8_t bits, float* out) {
    /* Validate first; do NOT touch `out` until every check passes. */
    if (packed == NULL || scales == NULL || zeros == NULL || out == NULL) return;
    if (n == 0)               return;
    if (group_size == 0)      return;
    if (!tq_bits_valid(bits)) return;

    /* Overflow guards mirroring encode. */
    if (n > SIZE_MAX - group_size) return;
    if ((size_t)bits > 0 && n > SIZE_MAX / (size_t)bits) return;

    size_t expected_packed_len = (n * (size_t)bits + 7u) / 8u;
    if (packed_len != expected_packed_len) return;

    /* Packed buffer must cover at least one byte per boundary crossing the
     * last partial byte of the bitstream. read_bits walks up to byte_idx+1. */
    if (packed_len < expected_packed_len) return;

    size_t n_groups = (n + group_size - 1u) / group_size;

    for (size_t g = 0; g < n_groups; g++) {
        float scale = scales[g];
        float zero  = zeros[g];

        size_t start = g * group_size;
        size_t end   = start + group_size;
        if (end > n) end = n;

        for (size_t i = start; i < end; i++) {
            size_t bit_off = (g * group_size + (i - start)) * (size_t)bits;
            uint8_t q = tq_read_bits(packed, bit_off, bits);
            out[i] = zero + (float)q * scale;
        }
    }
}

void tq_c_free(void* ptr) { free(ptr); }
