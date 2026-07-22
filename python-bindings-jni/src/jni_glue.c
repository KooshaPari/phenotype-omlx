/* JNI bridge stub — embeds the JVM via Java 17+ JNI invocation.
 * On macOS: brew install java  (or set JAVA_HOME explicitly).
 * On Linux: apt install default-jdk
 *
 * To wire to the JVM TurboQuant class:
 *   $ javac -d target jvm_src/TurboQuant.java
 *   $ jar cf turboquant.jar -C target .
 *   $ cp turboquant.jar $PHENOTYPE_JNI_LIBS/
 * Then tq_jni_encode calls the JVM via CallStatic*XxxMethod.
 */
#include "jni_glue.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration for libjvm; resolved at link time.
 * The actual TurboQuant + REFRACT code lives in the JVM, not in C.
 */
extern int  tq_jvm_create(const char* path);
extern int  tq_jvm_destroy(void);
extern int  tq_jvm_encode(const float* data, int len, int bits, int group_size,
                          uint8_t** packed, int* p_len,
                          float** scales, int* s_len,
                          float** zeros,  int* z_len);
extern int  tq_jvm_decode(const uint8_t* packed, int p_len,
                          const float* scales, int s_len,
                          const float* zeros,  int z_len,
                          int n, int bits, int group_size,
                          float** out_buf);

static pthread_mutex_t g_init_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_inited = 0;

int tq_jni_init(const char* jvm_path) {
    pthread_mutex_lock(&g_init_lock);
    int rc = 0;
    if (!g_inited) {
        rc = tq_jvm_create(jvm_path);
        g_inited = (rc == 0);
    }
    pthread_mutex_unlock(&g_init_lock);
    return g_inited ? 0 : -1;
}

int tq_jni_encode(
    const float* data, int len, int bits, int group_size,
    uint8_t** out_packed, int* out_packed_len,
    float**   out_scales, int* out_scales_len,
    float**   out_zeros,  int* out_zeros_len) {
    if (!g_inited) return -1;
    return tq_jvm_encode(data, len, bits, group_size,
                         out_packed, out_packed_len,
                         out_scales, out_scales_len,
                         out_zeros,  out_zeros_len);
}

int tq_jni_decode(
    const uint8_t* packed, int packed_len,
    const float* scales, int scales_len,
    const float* zeros,  int zeros_len,
    int n, int bits, int group_size,
    float** out_buf) {
    if (!g_inited) return -1;
    return tq_jvm_decode(packed, packed_len, scales, scales_len, zeros, zeros_len,
                        n, bits, group_size, out_buf);
}

void tq_jni_shutdown(void) {
    pthread_mutex_lock(&g_init_lock);
    if (g_inited) {
        tq_jvm_destroy();
        g_inited = 0;
    }
    pthread_mutex_unlock(&g_init_lock);
}

#ifdef __cplusplus
}
#endif
