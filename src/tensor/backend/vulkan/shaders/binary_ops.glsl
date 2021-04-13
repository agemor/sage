#version 450

#define MAX_DIM 8

layout(set = 0, binding = 0) writeonly buffer BufferOut {
    float data[];
} buffer_out;

layout(set = 0, binding = 1) readonly buffer BufferIn0 {
    float data[];
} buffer_in0;

layout(set = 0, binding = 2) readonly buffer BufferIn1 {
    float data[];
} buffer_in1;

layout(push_constant) uniform PushConstants {
    uint num_dim;
    uint offsets[3];
    uint strides[3 * MAX_DIM];// [out, in0, in1] interleaved
};

void main() {
    uint index_out = offsets[0] + gl_GlobalInvocationID.x;
    uint index_in0 = offsets[1];
    uint index_in1 = offsets[2];

    // Map buffer indices
    uint rem = index_out;
    for (uint i = 0; i < num_dim; ++i) {
        uint strides_out = strides[i * 3 + 0];
        uint strides_in0 = strides[i * 3 + 1];
        uint strides_in1 = strides[i * 3 + 2];
        uint c = rem / strides_out;
        rem -= c * strides_out;
        index_in0 += c * strides_in0;
        index_in1 += c * strides_in1;
    }

    float x0 = buffer_in0.data[index_in0];
    float x1 = buffer_in1.data[index_in1];
    float y = 0;

    #ifdef ADD
    y = x0 + x1;
    #endif
    #ifdef SUB
    y = x0 - x1;
    #endif
    #ifdef MUL
    y = x0 * x1;
    #endif
    #ifdef DIV
    y = sign(x1) * x0 / max(abs(x1), 0.0000001);
    #endif
    #ifdef POW
    y = pow(x0, x1);
    #endif
    #ifdef VMAX
    y = max(x0, x1);
    #endif
    #ifdef VMIN
    y = min(x0, x1);
    #endif
    #ifdef SQUDIFF
    y = (x0 - x1) * (x0 - x1);
    #endif

    buffer_out.data[index_out] = y;
}