use crate::tensor::{backend, Tensor};
use crate::tensor::backend::vulkan::shader;
use std::sync::Arc;
use vulkano::descriptor::pipeline_layout::PipelineLayout;
use vulkano::device::{Device, Queue};
use vulkano::pipeline::ComputePipeline;
use crate::tensor::backend::{BinaryIndexOperation, UnaryOperation, BinaryOperation};
use vulkano::command_buffer::AutoCommandBufferBuilder;

pub struct Processor {
    device: Arc<Device>,
    queue: Arc<Queue>,

    pub binary_ops_add: Arc<ComputePipeline<PipelineLayout<shader::binary_ops::Layout>>>,
    pub binary_ops_sub: Arc<ComputePipeline<PipelineLayout<shader::binary_ops::Layout>>>,

    pub tensor_contraction:
        Arc<ComputePipeline<PipelineLayout<shader::tensor_contraction::Layout>>>,
    pub tensor_reduction_add:
        Arc<ComputePipeline<PipelineLayout<shader::tensor_reduction::Layout>>>,
    pub tensor_reduction_mul:
        Arc<ComputePipeline<PipelineLayout<shader::tensor_reduction::Layout>>>,

    pub im2col: Arc<ComputePipeline<PipelineLayout<shader::im2col::Layout>>>,
    pub col2im: Arc<ComputePipeline<PipelineLayout<shader::col2im::Layout>>>,
}

impl Processor {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device,
            queue,
            binary_ops_add: create_pipeline!(binary_ops, binary_ops_add, device),
            binary_ops_sub: create_pipeline!(binary_ops, binary_ops_sub, device),
            tensor_contraction: create_pipeline!(tensor_contraction, tensor_contraction, device),
            tensor_reduction_add: create_pipeline!(tensor_reduction, tensor_reduction_add, device),
            tensor_reduction_mul: create_pipeline!(tensor_reduction, tensor_reduction_mul, device),
            im2col: create_pipeline!(im2col, im2col, device),
            col2im: create_pipeline!(col2im, col2im, device),
        }
    }
}

impl backend::Processor for Processor {
    fn copy(&self, input: &Tensor, output: &mut Tensor) {
        todo!()
    }

    fn concat(&self, inputs: &[&Tensor], output: &mut Tensor, axis: usize) {
        todo!()
    }

    fn unary_op(&self, input: &Tensor, output: &mut Tensor, op: UnaryOperation) {
        todo!()
    }

    fn binary_op(&self, input1: &Tensor, input2: &Tensor, output: &mut Tensor, op: BinaryOperation) {
        // ComputePipeline<PipelineLayout<
        // assert same shape
        if input0.shape != input1.shape
            || input1.shape != output.shape
            || output.shape != input0.shape
        {
            panic!("all tensors should be in the same shape");
        }

        // TODO: Assert same device
        let backend = output.backend();

        let pipeline = match ops {
            BinaryOps::Add => backend.pipelines.binary_ops_add.clone(),
            BinaryOps::Sub => backend.pipelines.binary_ops_sub.clone(),
            _ => {
                unimplemented!()
            }
        };

        let layout = pipeline.descriptor_set_layout(0).unwrap();

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(output.buffer())
                .unwrap()
                .add_buffer(input0.buffer())
                .unwrap()
                .add_buffer(input1.buffer())
                .unwrap()
                .build()
                .unwrap(),
        );

        let push_constants = shaders::binary_ops_add::ty::PushConstants {
            num_dim: output.shape.len() as u32,
            offsets: [
                output.offset as u32,
                input0.offset as u32,
                input1.offset as u32,
            ],
            strides: interleave_strides::<24>(&[&output.strides, &input0.strides, &input1.strides]),
        };

        // uint num_dim;
        // uint offsets[3];
        // uint strides[3 * MAX_DIM];// [out, in0, in1] interleaved
        //

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            backend.device(),
            backend.queue().family(),
        )
            .unwrap();

        builder
            .dispatch(
                [output.shape.size() as u32, 1, 1],
                // pipeline
                pipeline.clone(),
                // descriptor set
                descriptor_set.clone(),
                // constants
                push_constants,
                vec![],
            )
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let finished = command_buffer.execute(backend.queue()).unwrap();
        println!("Submitted command buffer to queue");

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        println!("Command buffer finished executing");
    }

    fn reduction(&self, input: &Tensor, output: &mut Tensor, op: BinaryOperation, axes: Vec<usize>) {
        let axes_red = axes.to_indices(tensor_in.shape.len());

        let axes_pre = (0..tensor_in.shape.len())
            .filter(|a| !axes_red.contains(a))
            .collect::<Vec<usize>>();

        fn subshape(t: &Tensor, axes: &[usize]) -> (Shape, Vec<usize>, Vec<usize>) {
            let subshape = Shape::new(&axes.iter().map(|&a| t.shape[a]).collect::<Vec<usize>>());
            let stride = Shape::default_strides(subshape);
            let stride_ref = axes.iter().map(|&a| t.strides[a]).collect::<Vec<usize>>();

            (subshape, stride, stride_ref)
        }

        let (shape_pre, stride_pre, stride_pre_ref) = subshape(tensor_in, &axes_pre);
        let (shape_red, stride_red, stride_red_ref) = subshape(tensor_in, &axes_red);

        let backend = tensor_out.backend();

        let pipeline = match ops {
            BinaryOps::Add => backend.pipelines.tensor_reduction_add.clone(),
            BinaryOps::Mul => backend.pipelines.tensor_reduction_mul.clone(),
            _ => {
                unimplemented!()
            }
        };
        let layout = pipeline.descriptor_set_layout(0).unwrap();

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(tensor_out.buffer())
                .unwrap()
                .add_buffer(tensor_in.buffer())
                .unwrap()
                .build()
                .unwrap(),
        );

        let push_constants = shaders::tensor_reduction::ty::PushConstants {
            offset_out: tensor_out.offset as u32,
            offset_in: tensor_in.offset as u32,
            ndim_red: axes_red.len() as u32,
            ndim_pre: axes_pre.len() as u32,
            size_red: shape_red.size() as u32,
            stride_pre: vec_to_sized_u32(stride_pre),
            stride_pre_ref: vec_to_sized_u32(stride_pre_ref),
            stride_red: vec_to_sized_u32(stride_red),
            stride_red_ref: vec_to_sized_u32(stride_red_ref),
        };

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            backend.device(),
            backend.queue().family(),
        )
            .unwrap();

        builder
            .dispatch(
                [shape_pre.size() as u32, 1, 1],
                // pipeline`
                pipeline.clone(),
                // descriptor set
                descriptor_set.clone(),
                // constants
                push_constants,
                vec![],
            )
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let finished = command_buffer.execute(backend.queue()).unwrap();

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    fn reduction_index(&self, input: &Tensor, output: &mut Tensor, op: BinaryIndexOperation, axis: usize) {
        todo!()
    }

    fn contraction(&self, input1: &Tensor, input2: &Tensor, output: &mut Tensor, axes1: Vec<usize>, axes2: Vec<usize>) {
        let axes_cont1 = axes1.to_indices(tensor_in1.shape.len());
        let axes_cont2 = axes2.to_indices(tensor_in2.shape.len());

        let axes_pre1 = (0..tensor_in1.shape.len())
            .filter(|a| !axes_cont1.contains(a))
            .collect::<Vec<usize>>();

        let axes_pre2 = (0..tensor_in2.shape.len())
            .filter(|a| !axes_cont2.contains(a))
            .collect::<Vec<usize>>();

        // println!("{:?}", axes_cont1);
        // println!("{:?}", axes_cont2);
        // println!("{:?}", axes_pre1);
        // println!("{:?}", axes_pre2);

        fn subshape(t: &Tensor, axes: &[usize]) -> (Shape, Vec<usize>, Vec<usize>) {
            let subshape = Shape::new(&axes.iter().map(|&a| t.shape[a]).collect::<Vec<usize>>());
            let stride = Shape::default_strides(subshape);
            let stride_ref = axes.iter().map(|&a| t.strides[a]).collect::<Vec<usize>>();

            (subshape, stride, stride_ref)
        }

        let (shape_pre1, stride_pre1, stride_pre_ref1) = subshape(tensor_in1, &axes_pre1);
        let (shape_pre2, stride_pre2, stride_pre_ref2) = subshape(tensor_in2, &axes_pre2);
        let (shape_cont, stride_cont, stride_cont_ref1) = subshape(tensor_in1, &axes_cont1);
        let (_, stride_cont, stride_cont_ref2) = subshape(tensor_in2, &axes_cont2);

        let backend = tensor_out.backend();

        let pipeline = backend.pipelines.tensor_contraction.clone();

        let layout = pipeline.descriptor_set_layout(0).unwrap();

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(tensor_out.buffer())
                .unwrap()
                .add_buffer(tensor_in1.buffer())
                .unwrap()
                .add_buffer(tensor_in2.buffer())
                .unwrap()
                .build()
                .unwrap(),
        );

        // println!("{:?}", stride_pre1);
        // println!("{:?}", stride_pre2);
        // println!("{:?}", stride_pre_ref1);
        // println!("{:?}", stride_pre_ref2);
        // println!("{:?}", stride_cont);
        // println!("{:?}", stride_cont_ref1);
        // println!("{:?}", stride_cont_ref2);

        let push_constants = shaders::tensor_contraction::ty::PushConstants {
            offset_out: tensor_out.offset as u32,
            offset_in1: tensor_in1.offset as u32,
            offset_in2: tensor_in2.offset as u32,
            ndim_cont: axes_cont1.len() as u32,
            ndim_pre1: axes_pre1.len() as u32,
            ndim_pre2: axes_pre2.len() as u32,
            size_cont: shape_cont.size() as u32,
            stride_pre1: vec_to_sized_u32(stride_pre1),
            stride_pre2: vec_to_sized_u32(stride_pre2),
            stride_pre_ref1: vec_to_sized_u32(stride_pre_ref1),
            stride_pre_ref2: vec_to_sized_u32(stride_pre_ref2),
            stride_cont: vec_to_sized_u32(stride_cont),
            stride_cont_ref1: vec_to_sized_u32(stride_cont_ref1),
            stride_cont_ref2: vec_to_sized_u32(stride_cont_ref2),
        };

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            backend.device(),
            backend.queue().family(),
        )
            .unwrap();

        builder
            .dispatch(
                [shape_pre1.size() as u32, shape_pre2.size() as u32, 1],
                // pipeline`
                pipeline.clone(),
                // descriptor set
                descriptor_set.clone(),
                // constants
                push_constants,
                vec![],
            )
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let finished = command_buffer.execute(backend.queue()).unwrap();

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    fn im2col(&self, input: &Tensor, output: &mut Tensor, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize), dilation: (usize, usize)) {
        // assert output size
        let backend = output.backend();

        let pipeline = backend.pipelines.im2col.clone();

        let layout = pipeline.descriptor_set_layout(0).unwrap();

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(output.buffer())
                .unwrap()
                .add_buffer(input.buffer())
                .unwrap()
                .build()
                .unwrap(),
        );

        // uvec2 image_size;
        // uvec2 output_size;
        // uvec2 kernel_size;
        // uvec2 stride;
        // uvec2 dilation;
        // uvec2 padding;

        // input.shape = [N, C, H, W]

        let batch_size = input.shape[0] as u32;
        let num_channels = input.shape[1] as u32;
        let img_h = input.shape[2] as u32;
        let img_w = input.shape[3] as u32;

        let output_size = (
            (img_w + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1) / stride.0 + 1,
            (img_h + 2 * padding.1 - dilation.1 * (kernel_size.1 - 1) - 1) / stride.1 + 1,
        );

        // println!("{:?}", (img_w, img_h));
        // println!("{:?}", output_size);
        // println!("{:?}", kernel_size);
        // println!("{:?}", stride);
        // println!("{:?}", padding);
        // println!("{:?}", dilation);

        let img_len = img_w * img_h;
        let kernel_len = kernel_size.0 * kernel_size.1;
        let output_len = output_size.0 * output_size.1;

        let push_constants = shaders::im2col::ty::PushConstants {
            image_size: [img_w, img_h],
            //image_strides: [1, img_len, img_len * num_channels],
            output_size: [output_size.0, output_size.1],
            // output_strides: [
            //     kernel_len,
            //     kernel_len * output_len,
            //     kernel_len * output_len * num_channels,
            // ],
            kernel_size: [kernel_size.0, kernel_size.1],
            stride: [stride.0, stride.1],
            padding: [padding.0, padding.1],
            dilation: [dilation.0, dilation.1],
        };

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            backend.device(),
            backend.queue().family(),
        )
            .unwrap();

        builder
            .dispatch(
                [output_len, num_channels, batch_size],
                // pipeline
                pipeline.clone(),
                // descriptor set
                descriptor_set.clone(),
                // constants
                push_constants,
                vec![],
            )
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let finished = command_buffer.execute(backend.queue()).unwrap();

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    fn col2im(&self, input: &Tensor, output: &mut Tensor, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize), dilation: (usize, usize)) {
        // assert output size
        let backend = output.backend();

        let pipeline = backend.pipelines.col2im.clone();

        let layout = pipeline.descriptor_set_layout(0).unwrap();

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(output.buffer())
                .unwrap()
                .add_buffer(input.buffer())
                .unwrap()
                .build()
                .unwrap(),
        );

        // uvec2 image_size;
        // uvec2 output_size;
        // uvec2 kernel_size;
        // uvec2 stride;
        // uvec2 dilation;
        // uvec2 padding;

        // input.shape = (N, C, OH, OW, KH, KW)

        let batch_size = input.shape[0] as u32;
        let num_channels = input.shape[1] as u32;
        let output_h = input.shape[2] as u32;
        let output_w = input.shape[3] as u32;
        let kernel_h = input.shape[4] as u32;
        let kernel_w = input.shape[5] as u32;

        // output.shape = (N, C, H, W)
        let img_h = output.shape[2] as u32;
        let img_w = output.shape[3] as u32;

        let img_len = img_w * img_h;
        let kernel_len = kernel_size.0 * kernel_size.1;
        let output_len = output_w * output_h;

        let push_constants = shaders::col2im::ty::PushConstants {
            image_size: [img_w, img_h],
            output_size: [output_w, output_h],
            kernel_size: [kernel_size.0, kernel_size.1],
            stride: [stride.0, stride.1],
            padding: [padding.0, padding.1],
            dilation: [dilation.0, dilation.1],
        };

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            backend.device(),
            backend.queue().family(),
        )
            .unwrap();

        builder
            .dispatch(
                [output_len, num_channels, batch_size],
                // pipeline
                pipeline.clone(),
                // descriptor set
                descriptor_set.clone(),
                // constants
                push_constants,
                vec![],
            )
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let finished = command_buffer.execute(backend.queue()).unwrap();

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
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
                    shaders::$layout::Layout(ShaderStages::compute()),
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
        shaders::$shader::Shader::load($device.clone())
            .unwrap()
            .module()
    };
}
