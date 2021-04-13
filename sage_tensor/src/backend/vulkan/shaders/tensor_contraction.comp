#version 450

#define LOCAL_SIZE 1
#define MAX_DIM 8

layout(local_size_x = LOCAL_SIZE) in;

layout(set = 0, binding = 0) writeonly buffer BufferOutput {
    float data[];
} buffer_out;

layout(set = 0, binding = 1) readonly buffer BufferInput1 {
    float data[];
} buffer_in1;

layout(set = 0, binding = 2) readonly buffer BufferInput2 {
    float data[];
} buffer_in2;

layout(push_constant) uniform PushConstants {
// Buffer offsets
    uint offset_out;
    uint offset_in1;
    uint offset_in2;

// Number of dimensions
    uint ndim_cont;
    uint ndim_pre1;
    uint ndim_pre2;

    uint size_cont;

    uint stride_pre1[MAX_DIM];
    uint stride_pre2[MAX_DIM];
    uint stride_pre_ref1[MAX_DIM];
    uint stride_pre_ref2[MAX_DIM];

    uint stride_cont[MAX_DIM];
    uint stride_cont_ref1[MAX_DIM];
    uint stride_cont_ref2[MAX_DIM];
};

shared float sum_partial[LOCAL_SIZE];

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

void main() {

    uint index_out = offset_out + gl_WorkGroupID.x * gl_NumWorkGroups.y + gl_WorkGroupID.y;
    uint index_in1_base = offset_in1 + translate(gl_WorkGroupID.x, ndim_pre1, stride_pre1, stride_pre_ref1);
    uint index_in2_base = offset_in2 + translate(gl_WorkGroupID.y, ndim_pre2, stride_pre2, stride_pre_ref2);

    uint num_elems = ((size_cont + LOCAL_SIZE - 1) / LOCAL_SIZE);// k/32
    uint offset_inv = gl_LocalInvocationID.x * num_elems;

    // calculate partial products
    float y = 0;
    for (uint i = 0; i < num_elems; ++i) {
        uint index_inv = offset_inv + i;
        uint index_in1 = index_in1_base + translate(index_inv, ndim_cont, stride_cont, stride_cont_ref1);
        uint index_in2 = index_in2_base + translate(index_inv, ndim_cont, stride_cont, stride_cont_ref2);

        float x1 = buffer_in1.data[index_in1];
        float x2 = buffer_in2.data[index_in2];

        // product
        y += x1 * x2;
    }
    sum_partial[gl_LocalInvocationID.x] = y;

    // sum partial products
    barrier();
    float sum_total = 0;
    for (uint i = 0; i < LOCAL_SIZE; ++i) {
        sum_total += sum_partial[i];
    }

    // write the output
    barrier();
    buffer_out.data[index_out] = sum_total;
}