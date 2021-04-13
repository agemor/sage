use vulkano::pipeline::shader::{ComputeEntryPoint, ShaderModule};
use vulkano_shaders;


// Unary ops ------------

pub mod unary_ops {
    vulkano_shaders::shader! { ty: "compute", path: "src/backend/vulkan/shaders/unary_ops.comp" }
}

pub mod unary_ops_copy {
    vulkano_shaders::shader! {  ty: "compute",  path: "src/backend/vulkan/shaders/unary_ops.comp", define:[("COPY", "")] }
}

pub mod unary_ops_neg {
    vulkano_shaders::shader! {  ty: "compute",  path: "src/backend/vulkan/shaders/unary_ops.comp", define:[("NEG", "")] }
}

// Binary ops --------------------------------------------------------------------------------------

pub mod binary_ops {
    vulkano_shaders::shader! { ty: "compute", path: "src/backend/vulkan/shaders/binary_ops.comp" }
}

pub mod binary_ops_add {
    vulkano_shaders::shader! {  ty: "compute",  path: "src/backend/vulkan/shaders/binary_ops.comp", define:[("ADD", "")] }
}

pub mod binary_ops_sub {
    vulkano_shaders::shader! { ty: "compute", path: "src/backend/vulkan/shaders/binary_ops.comp", define:[("SUB", "")],  }
}

// Unary ops ---------------------------------------------------------------------------------------

// Inner products ----------------------------------------------------------------------------------

pub mod tensor_contraction {
    vulkano_shaders::shader! { ty: "compute", path: "src/backend/vulkan/shaders/tensor_contraction.comp" }
}

pub mod tensor_reduction {
    vulkano_shaders::shader! { ty: "compute", path: "src/backend/vulkan/shaders/tensor_reduction.comp" }
}

pub mod tensor_reduction_add {
    vulkano_shaders::shader! { ty: "compute", path: "src/backend/vulkan/shaders/tensor_reduction.comp", define:[("ADD", "")] }
}

pub mod tensor_reduction_mul {
    vulkano_shaders::shader! { ty: "compute", path: "src/backend/vulkan/shaders/tensor_reduction.comp", define:[("MUL", "")] }
}

pub mod tensor_reduction_index {
    vulkano_shaders::shader! { ty: "compute", path: "src/backend/vulkan/shaders/tensor_reduction_index.comp" }
}

pub mod tensor_reduction_index_max {
    vulkano_shaders::shader! { ty: "compute", path: "src/backend/vulkan/shaders/tensor_reduction_index.comp", define:[("MAX", "")]  }
}

pub mod tensor_reduction_index_min {
    vulkano_shaders::shader! { ty: "compute", path: "src/backend/vulkan/shaders/tensor_reduction_index.comp", define:[("MIN", "")]  }
}



// conv ops

pub mod im2col {
    vulkano_shaders::shader! { ty: "compute", path: "src/backend/vulkan/shaders/im2col.comp" }
}

pub mod col2im {
    vulkano_shaders::shader! { ty: "compute", path: "src/backend/vulkan/shaders/col2im.comp" }
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

#[macro_export]
macro_rules! create_pipeline {
    ($layout:ident, $shader:ident, $device:expr) => {
        Arc::new(
            ComputePipeline::new(
                $device.clone(),
                &derived_shaders(
                    &load_shader!($shader, $device),
                    shader::$layout::Layout(ShaderStages::compute()),
                ),
                &(),
                None,
            )
            .unwrap(),
        )
    };
}
#[macro_export]
macro_rules! load_shader {
    ($shader:ident, $device:expr) => {
        shader::$shader::Shader::load($device.clone())
            .unwrap()
            .module()
    };
}
