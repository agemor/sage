use vulkano::descriptor::descriptor::ShaderStages;
use vulkano::pipeline::shader::{ComputeEntryPoint, ShaderModule};
use vulkano_shaders;

// Binary ops --------------------------------------------------------------------------------------

static SHADER_PATH:&str = "src/tensor/backend/vulkan/shaders";

pub mod binary_ops {
    vulkano_shaders::shader! { ty: "compute", path: "src/tensor/backend/vulkan/shaders/binary_ops.glsl"  }
}

pub mod binary_ops_add {
    vulkano_shaders::shader! {  ty: "compute",  path: "src/tensor/backend/vulkan/shaders/binary_ops.glsl", define:[("ADD", "")] }
}

pub mod binary_ops_sub {
    vulkano_shaders::shader! { ty: "compute", path: "src/tensor/backend/vulkan/shaders/binary_ops.glsl", define:[("SUB", "")],  }
}

// Unary ops ---------------------------------------------------------------------------------------

// Inner products ----------------------------------------------------------------------------------

pub mod tensor_contraction {
    vulkano_shaders::shader! { ty: "compute", path: "src/tensor/backend/vulkan/shaders/tensor_contraction.glsl" }
}

pub mod tensor_reduction {
    vulkano_shaders::shader! { ty: "compute", path: "src/tensor/backend/vulkan/shaders/tensor_reduction.glsl" }
}

pub mod tensor_reduction_add {
    vulkano_shaders::shader! { ty: "compute", path: "src/tensor/backend/vulkan/shaders/tensor_reduction.glsl", define:[("ADD", "")] }
}

pub mod tensor_reduction_mul {
    vulkano_shaders::shader! { ty: "compute", path: "src/tensor/backend/vulkan/shaders/tensor_reduction.glsl", define:[("MUL", "")] }
}

// conv ops

pub mod im2col {
    vulkano_shaders::shader! { ty: "compute", path: "src/tensor/backend/vulkan/shaders/im2col.glsl" }
}

pub mod col2im {
    vulkano_shaders::shader! { ty: "compute", path: "src/tensor/backend/vulkan/shaders/col2im.glsl" }
}

pub fn derived_shaders<S, L>(shader: &ShaderModule, layout: L) -> ComputeEntryPoint<S, L> {
    static NAME: [u8; 5] = [109, 97, 105, 110, 0]; // main
    unsafe {
        shader.compute_entry_point(
            ::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _),
            layout,
        )
    }
}
