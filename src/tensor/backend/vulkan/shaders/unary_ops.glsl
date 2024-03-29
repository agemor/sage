#version 450

#define LOCAL_SIZE 32
#define MAX_DIM 8

layout(local_size_x = LOCAL_SIZE) in;

layout(set = 0, binding = 0) writeonly buffer BufferOut {
    float data[];
} buffer_out;

layout(set = 0, binding = 1) readonly buffer BufferIn {
    float data[];
} buffer_in;


layout(push_constant) uniform PushConstants {
    uint num_dim;
    uint offsets[2];
    uint strides[2 * MAX_DIM];// [out, in] interleaved
};

void main() {
    uint index_out = offsets[0] + gl_GlobalInvocationID.x;
    uint index_in = offsets[1];

    // Map buffer indices
    uint rem = index_out;
    for (uint i = 0; i < num_dim; ++i) {
        uint strides_out = strides[i * 2 + 0];
        uint strides_in = strides[i * 2 + 1];
        uint c = rem / strides_out;
        rem -= c * strides_out;
        index_in += c * strides_in;
    }

    float x = buffer_in.data[index_in];
    float y = 0;

    #ifdef NEG
    y = -x;
    #endif
    #ifdef EXP
    y = exp(x);
    #endif
    #ifdef SIGN
    y = sign(x);
    #endif
    #ifdef SQRT
    y = sqrt(x);
    #endif
    #ifdef RSQRT
    y = inversesqrt(x);
    #endif
    #ifdef ABS
    y = abs(x);
    #endif
    #ifdef TANH
    y = tanh(x);
    #endif
    #ifdef SQUARE
    y = x * x;
    #endif
    #ifdef LOG
    y = log(max(x, vec4(0.0000001)));
    #endif
    #ifdef SIGMOID
    y = 1.f / (1.f + exp(-x));
    #endif
    #ifdef TAN
    y = tan(x);
    #endif
    #ifdef COS
    y = cos(x);
    #endif
    #ifdef SIN
    y = sin(x);
    #endif
    #ifdef CEIL
    y = ceil(x);
    #endif
    #ifdef FLOOR
    y = floor(x);
    #endif
    #ifdef EXPM1
    y = exp(x) - FLOAT4(1);
    #endif
    #ifdef RECIPROCAL
    y = FLOAT4(1) / x;
    #endif
    #ifdef SINH
    y = sinh(x);
    #endif
    #ifdef ASINH
    y = asinh(x);
    #endif
    #ifdef ASIN
    y = asin(x);
    #endif
    #ifdef COSH
    y = cosh(x);
    #endif
    #ifdef ACOSH
    y = acosh(x);
    #endif
    #ifdef ACOS
    y = acos(x);
    #endif
    #ifdef ATAN
    y = atan(x);
    #endif
    #ifdef ATANH
    y = atanh(x);
    #endif
    #ifdef LOG1P
    y = log(FLOAT4(1) + x);
    #endif
    #ifdef ROUND
    y = round(x);
    #endif

    buffer_out.data[index_out] = y;
}