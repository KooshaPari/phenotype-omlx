#include <metal_stdlib>
using namespace metal;
kernel void deltanet_step_f32(device const float*q[[buffer(0)]],device const float*k[[buffer(1)]],device const float*v[[buffer(2)]],device const float*s[[buffer(3)]],device const float*p[[buffer(4)]],device float*out[[buffer(5)]],device float*next[[buffer(6)]],constant uint&n[[buffer(7)]],uint gid[[thread_position_in_grid]]){if(gid!=0)return;float beta=p[0];for(uint i=0;i<n;i++){for(uint j=0;j<n;j++){float kk=0;for(uint z=0;z<n;z++)kk+=k[z]*s[z*n+j];next[i*n+j]=s[i*n+j]-beta*k[i]*kk+beta*v[i]*k[j];}}for(uint i=0;i<n;i++){float a=0;for(uint j=0;j<n;j++)a+=q[j]*next[j*n+i];out[i]=a;}}

kernel void deltanet_state_f32(device const float*k[[buffer(0)]],device const float*v[[buffer(1)]],device const float*s[[buffer(2)]],device const float*p[[buffer(3)]],device float*next[[buffer(4)]],constant uint&n[[buffer(5)]],uint gid[[thread_position_in_grid]]){uint total=n*n;if(gid>=total)return;uint i=gid/n;uint j=gid%n;float kk=0.0f;for(uint z=0;z<n;z++)kk+=k[z]*s[z*n+j];next[gid]=s[gid]-p[0]*k[i]*kk+p[0]*v[i]*k[j];}

kernel void deltanet_output_f32(device const float*q[[buffer(0)]],device const float*next[[buffer(1)]],device float*out[[buffer(2)]],constant uint&n[[buffer(3)]],uint gid[[thread_position_in_grid]]){if(gid>=n)return;float a=0.0f;for(uint j=0;j<n;j++)a+=q[j]*next[j*n+gid];out[gid]=a;}

// Two-pass variant.  State update and readout are separate kernels so Metal
// provides an explicit command-encoder ordering barrier between them.  Each
// output element is independent; unlike the legacy one-thread kernel this
// exposes the matrix/output parallelism without changing the ABI buffers.
kernel void deltanet_state_update_f32(
    device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
    device const float* s [[buffer(3)]], device const float* p [[buffer(4)]],
    device float* next [[buffer(6)]], constant uint& n [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = n * n;
    if (gid >= total) return;
    uint i = gid / n;
    uint j = gid - i * n;
    float kk = 0.0f;
    for (uint z = 0; z < n; ++z) kk += k[z] * s[z * n + j];
    float beta = p[0];
    next[gid] = s[gid] - beta * k[i] * kk + beta * v[i] * k[j];
}

kernel void deltanet_readout_f32(
    device const float* q [[buffer(0)]], device const float* next [[buffer(6)]],
    device float* out [[buffer(5)]], constant uint& n [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float a = 0.0f;
    for (uint j = 0; j < n; ++j) a += q[j] * next[j * n + gid];
    out[gid] = a;
}
