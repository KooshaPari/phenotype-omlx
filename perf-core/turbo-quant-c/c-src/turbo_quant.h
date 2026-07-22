#ifndef TURBO_QUANT_C_H
#define TURBO_QUANT_C_H
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "abi_v1.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Versioned Native ABI v1 ──────────────────────────────────────────────
 *
 * The canonical contract lives in perf-core/native-abi and is included via
 * abi_v1.h. The two entry points below are the C implementation of that
 * contract. The pre-v1 aliases at the bottom of this file remain available
 * for backwards compatibility but translate into the new ABI internally. */

/* Backwards-compatibility aliases for callers compiled against the
 * pre-v1 turbo_quant_c API. New code should call tq_abi_encode / tq_abi_decode
 * directly. */
bool tq_c_encode(const float* data, size_t n, uint8_t bits, size_t group_size,
    size_t** out_shape, size_t* out_shape_len,
    uint8_t** out_packed, size_t* out_packed_len,
    float** out_scales, size_t* out_scales_len,
    float** out_zeros, size_t* out_zeros_len);

void tq_c_decode(const uint8_t* packed, size_t packed_len,
    const float* scales, const float* zeros,
    size_t n, size_t group_size, uint8_t bits, float* out);

void tq_c_free(void* ptr);

#ifdef __cplusplus
}
#endif
#endif
