#version 450

#define LOCAL_SIZE 1
#define MAX_DIM 8

layout(local_size_x = LOCAL_SIZE) in;

layout(set = 0, binding = 0) writeonly buffer BufferOutput {
    float data[];
} buffer_out;

layout(set = 0, binding = 1) readonly buffer BufferInput {
    float data[];
} buffer_in;

layout(push_constant) uniform PushConstants {
// Buffer offsets
    uint offset_out;
    uint offset_in;

// Number of dimensions
    uint ndim_red;
    uint ndim_pre;

    uint size_red;

    uint stride_pre[MAX_DIM];
    uint stride_pre_ref[MAX_DIM];

    uint stride_red[MAX_DIM];
    uint stride_red_ref[MAX_DIM];
};

shared float local_buffer[LOCAL_SIZE];

uint translate(uint index, uint len, uint stride[MAX_DIM], uint output_stride[MAX_DIM]) {
    uint output_index = 0;
    uint rem = index;
    for (uint i = 0; i < len; ++i) {
        uint c = rem / stride[i];
        rem -= c * stride[i];
        output_index += c * output_stride[i];
    }
    return output_index;
}

float reduce(float x1, float x2) {
    float y = 0;
    #ifdef ADD
    y = x1 + x2;
    #endif
    #ifdef MUL
    y = x1 * x2;
    #endif
    #ifdef VMAX
    y = max(x1, x2);
    #endif
    #ifdef VMIN
    y = min(x1, x2);
    #endif
    return y;
}

void main() {

    uint index_out = offset_out + gl_WorkGroupID.x;
    uint index_in_base = offset_in + translate(gl_WorkGroupID.x, ndim_pre, stride_pre, stride_pre_ref);

    uint num_elems = ((size_red + LOCAL_SIZE - 1) / LOCAL_SIZE);// k/32
    uint offset_inv = gl_LocalInvocationID.x * num_elems;

    float y = buffer_in.data[index_in_base + translate(offset_inv, ndim_red, stride_red, stride_red_ref)];

    for (uint i = 1; i < num_elems; ++i) {
        uint index_inv = offset_inv + i;
        uint index_in = index_in_base + translate(index_inv, ndim_red, stride_red, stride_red_ref);

        float x = buffer_in.data[index_in];

        // product
        y = reduce(y, x);
    }
    local_buffer[gl_LocalInvocationID.x] = y;

    // sum partial products
    barrier();
    float local_y = local_buffer[0];
    for (uint i = 1; i < LOCAL_SIZE; ++i) {
        local_y = reduce(local_y, local_buffer[i]);
    }

    // write the output
    barrier();
    buffer_out.data[index_out] = local_y;
}