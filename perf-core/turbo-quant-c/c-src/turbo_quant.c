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

/* ----- Native ABI v1 ----------------------------------------------------- *
 *
 * The versioned ABI defined by perf-core/native-abi. New consumers should
 * call `tq_abi_encode` and `tq_abi_decode` directly; the `tq_c_*` aliases
 * above remain for backwards compatibility and route through these entry
 * points. */

/* Compute the number of groups implied by `n` and `group_size`. Returns 0
 * when either argument is invalid. */
static size_t tq_abi_group_count(size_t n, size_t group_size) {
    if (n == 0 || group_size == 0) return 0;
    return (n + group_size - 1u) / group_size;
}

/* Expected packed buffer length for `n` elements at `bits` per element. */
static size_t tq_abi_expected_packed_len(size_t n, uint8_t bits) {
    if (n == 0 || bits == 0) return 0;
    return (n * (size_t)bits + 7u) / 8u;
}

/* Validate an encode request; returns the matching status. On validation
 * failure every output slot is reset to NULL. */
static tq_abi_status tq_abi_validate_encode(
    const tq_abi_encode_request* req, size_t* out_packed_len,
    size_t* out_n_groups) {
    if (req == NULL) return TQ_ABI_ERR_NULL_ARG;
    if (req->abi.major != TQ_ABI_VERSION_MAJOR) {
        return TQ_ABI_ERR_VERSION_MISMATCH;
    }
    if (req->data_ptr == NULL || req->n == 0) return TQ_ABI_ERR_NULL_ARG;
    if (!tq_bits_valid(req->bits)) return TQ_ABI_ERR_INVALID_BITS;
    if (req->group_size == 0) return TQ_ABI_ERR_INVALID_GROUPSZ;
    if (req->out_shape == NULL || req->out_shape_capacity == 0 ||
        req->out_packed == NULL || req->out_packed_capacity == 0 ||
        req->out_scales == NULL || req->out_scales_capacity == 0 ||
        req->out_zeros == NULL || req->out_zeros_capacity == 0) {
        return TQ_ABI_ERR_NULL_ARG;
    }

    /* Overflow checks. */
    if (req->n > SIZE_MAX - req->group_size) return TQ_ABI_ERR_OVERFLOW;
    if ((size_t)req->bits > 0 && req->n > SIZE_MAX / (size_t)req->bits) {
        return TQ_ABI_ERR_OVERFLOW;
    }

    size_t packed_len = tq_abi_expected_packed_len(req->n, req->bits);
    size_t n_groups = tq_abi_group_count(req->n, req->group_size);

    if (req->out_packed_capacity < packed_len ||
        req->out_shape_capacity < 1 ||
        req->out_scales_capacity < n_groups ||
        req->out_zeros_capacity < n_groups) {
        return TQ_ABI_ERR_OVERFLOW;
    }

    /* NaN / Inf rejection. */
    for (size_t i = 0; i < req->n; i++) {
        if (!isfinite(req->data_ptr[i])) return TQ_ABI_ERR_NONFINITE_INPUT;
    }

    *out_packed_len = packed_len;
    *out_n_groups = n_groups;
    return TQ_ABI_OK;
}

/* v1 contract: output buffers are caller-owned. The dispatcher must not
 * free them or invalidate the caller's pointers. We only NULL out the
 * descriptor slots themselves (req->out_shape etc.) — the caller's storage
 * backing those slots is untouched. */
static void tq_abi_reset_output_slots(tq_abi_encode_request* req) {
    if (req->out_shape)  req->out_shape  = NULL;
    if (req->out_packed) req->out_packed = NULL;
    if (req->out_scales) req->out_scales = NULL;
    if (req->out_zeros)  req->out_zeros  = NULL;
}

tq_abi_encode_result tq_abi_encode(const tq_abi_encode_request* req) {
    tq_abi_encode_result res;
    res.status = TQ_ABI_ERR_NULL_ARG;
    res.written_packed_len = 0;
    res.written_shape_len = 0;
    res.written_scales_len = 0;
    res.written_zeros_len = 0;

    if (req == NULL) return res;

    /* The public signature takes a const pointer (immutable request), but the
     * failure path is required by the ABI contract to NULL every output
     * slot, and the success path writes into the caller's buffers via those
     * slots. The slot pointers are mutable by contract (the caller hands
     * them in to be filled or cleared), so we cast away const here. */
    tq_abi_encode_request* mreq = (tq_abi_encode_request*)req;

    size_t packed_len = 0;
    size_t n_groups = 0;
    tq_abi_status vstatus = tq_abi_validate_encode(req, &packed_len, &n_groups);
    if (vstatus != TQ_ABI_OK) {
        tq_abi_reset_output_slots(mreq);
        res.status = vstatus;
        return res;
    }

    /* v1 is caller-owned buffers: the dispatcher writes into the caller's
     * storage and reports populated lengths via the result. Partial
     * allocation is therefore a caller-side concern, not ours. */
    uint32_t levels = (uint32_t)((1u << req->bits) - 1u);

    (*mreq->out_shape)[0] = req->n;

    for (size_t g = 0; g < n_groups; g++) {
        size_t start = g * req->group_size;
        size_t end = start + req->group_size;
        if (end > req->n) end = req->n;

        float gmin = req->data_ptr[start];
        float gmax = req->data_ptr[start];
        for (size_t i = start + 1; i < end; i++) {
            if (req->data_ptr[i] < gmin) gmin = req->data_ptr[i];
            if (req->data_ptr[i] > gmax) gmax = req->data_ptr[i];
        }

        float span = gmax - gmin;
        float scale = span / (float)levels;
        if (!(scale > 0.0f)) scale = 1e-30f;

        (*mreq->out_scales)[g] = scale;
        (*mreq->out_zeros)[g] = gmin;

        for (size_t i = start; i < end; i++) {
            float qf = (req->data_ptr[i] - gmin) / scale;
            if (!(qf >= 0.0f)) qf = 0.0f;
            uint32_t q = (uint32_t)(qf + 0.5f);
            if (q > levels) q = levels;
            size_t bit_off = (g * req->group_size + (i - start)) * (size_t)req->bits;
            tq_write_bits(*mreq->out_packed, bit_off, (uint8_t)q, req->bits);
        }
    }

    res.status = TQ_ABI_OK;
    res.written_packed_len = packed_len;
    res.written_shape_len = 1;
    res.written_scales_len = n_groups;
    res.written_zeros_len = n_groups;
    return res;
}

tq_abi_status tq_abi_decode(const tq_abi_decode_request* req) {
    if (req == NULL) return TQ_ABI_ERR_NULL_ARG;
    if (req->abi.major != TQ_ABI_VERSION_MAJOR) return TQ_ABI_ERR_VERSION_MISMATCH;
    if (req->out_ptr == NULL || req->n == 0) return TQ_ABI_ERR_NULL_ARG;
    if (!tq_bits_valid(req->bits)) return TQ_ABI_ERR_INVALID_BITS;
    if (req->group_size == 0) return TQ_ABI_ERR_INVALID_GROUPSZ;
    if (req->packed_ptr == NULL || req->scales_ptr == NULL || req->zeros_ptr == NULL) {
        return TQ_ABI_ERR_NULL_ARG;
    }

    /* Overflow guards mirroring encode. */
    if (req->n > SIZE_MAX - req->group_size) return TQ_ABI_ERR_OVERFLOW;
    if ((size_t)req->bits > 0 && req->n > SIZE_MAX / (size_t)req->bits) {
        return TQ_ABI_ERR_OVERFLOW;
    }

    size_t expected = tq_abi_expected_packed_len(req->n, req->bits);
    if (req->packed_len != expected) return TQ_ABI_ERR_INVALID_BITS;

    size_t n_groups = tq_abi_group_count(req->n, req->group_size);

    /* The public signature is `const`, but `out_ptr` is the caller's
     * writable buffer that the contract commits to fill on success. Cast
     * away const here, immediately after validation, so the success path
     * can write into it. */
    float* out = (float*)req->out_ptr;

    for (size_t g = 0; g < n_groups; g++) {
        float scale = req->scales_ptr[g];
        float zero  = req->zeros_ptr[g];

        size_t start = g * req->group_size;
        size_t end = start + req->group_size;
        if (end > req->n) end = req->n;

        for (size_t i = start; i < end; i++) {
            size_t bit_off = (g * req->group_size + (i - start)) * (size_t)req->bits;
            uint8_t q = tq_read_bits(req->packed_ptr, bit_off, req->bits);
            out[i] = zero + (float)q * scale;
        }
    }
    return TQ_ABI_OK;
}

void tq_abi_release(tq_abi_release_kind kind, void* ptr, size_t count) {
    if (ptr == NULL || count == 0) return;
    switch (kind) {
        case TQ_ABI_RELEASE_SHAPE:
            free(ptr);
            break;
        case TQ_ABI_RELEASE_PACKED:
            free(ptr);
            break;
        case TQ_ABI_RELEASE_SCALES:
            free(ptr);
            break;
        case TQ_ABI_RELEASE_ZEROS:
            free(ptr);
            break;
        default:
            /* Unknown kind — silently ignore so a forward-compatible caller
             * doesn't crash when given a future kind value. */
            break;
    }
}
