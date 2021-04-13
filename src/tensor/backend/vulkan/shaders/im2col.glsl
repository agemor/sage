#version 450

layout(set = 0, binding = 0) writeonly buffer BufferOutput {
    float data[];
} buffer_out;

layout(set = 0, binding = 1) readonly buffer BufferInput {
    float data[];
} buffer_in;

layout(push_constant) uniform PushConstants {
    uvec2 image_size;
//uvec3 image_strides;
    uvec2 output_size;
//uvec3 output_strides;
    uvec2 kernel_size;
    uvec2 stride;
    uvec2 padding;
    uvec2 dilation;
};

void main () {

    // in: [N, C, H, W]
    // out: [N, C, NCOL, K]

    uint image_len = image_size.x * image_size.y;
    uint kernel_len = kernel_size.x * kernel_size.y;

    uint index_in_base = image_len * gl_NumWorkGroups.y * gl_WorkGroupID.z
                       + image_len * gl_WorkGroupID.y;

    uint index_out_base = kernel_len * gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_WorkGroupID.z
                        + kernel_len * gl_NumWorkGroups.x * gl_WorkGroupID.y
                        + kernel_len * gl_WorkGroupID.x;

    uvec2 pos = uvec2(gl_WorkGroupID.x % output_size.x, gl_WorkGroupID.x / output_size.x);

    uvec2 offset = stride * pos - padding;

    for (uint x = 0; x < kernel_size.x; ++x) {
        uint img_x = x * dilation.x + offset.x;

        for (uint y = 0; y < kernel_size.y; ++y) {
            uint img_y = y * dilation.y + offset.y;

            // default value (applied for padding)
            float value = 0;
            if (img_x >=0 && img_y >= 0 && img_x < image_size.x && img_y < image_size.y) {
                uint index_in = index_in_base + img_y * image_size.x + img_x;
                value = buffer_in.data[index_in];
            }
            uint index_out = index_out_base + y * kernel_size.x + x;

            buffer_out.data[index_out] = value;
        }
    }
}


