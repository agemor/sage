#[macro_use]
mod shader;
mod shader_def_ex;

use crate::backend::vulkan::shader::derived_shaders;
use crate::backend::{Backend, Buffer};
use crate::ops::{BinaryIndexOperation, BinaryLogicOperation, BinaryOperation, UnaryOperation};
use crate::tensor::{Element, MemoryLayout, Tensor};
use itertools::Itertools;
use num_traits::{Float, NumOps};
use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;
use std::cmp::{min, Ordering};
use std::collections::HashMap;
use std::lazy::SyncLazy;
use std::sync::{Arc, Mutex};
use vulkano::buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::descriptor::descriptor::ShaderStages;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::pipeline_layout::PipelineLayout;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, Queue};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;

static INSTANCE: SyncLazy<Arc<vulkano::instance::Instance>> = SyncLazy::new(create_instance);
static CONTEXTS: SyncLazy<Mutex<HashMap<usize, Arc<BackendContext>>>> =
    SyncLazy::new(|| Mutex::new(HashMap::new()));

fn create_instance() -> Arc<vulkano::instance::Instance> {
    let app_infos = app_info_from_cargo_toml!();
    vulkano::instance::Instance::new(
        Some(&app_infos),
        &vulkano::instance::InstanceExtensions::none(),
        None,
    )
    .expect("No Vulkan implementations available.")
}

fn create_context(device_id: usize) -> BackendContext {
    let physical_device =
        vulkano::instance::PhysicalDevice::from_index(INSTANCE.borrow(), device_id)
            .expect("Failed to get the specified physical device.");

    let queue_family = physical_device
        .queue_families()
        .find(|&q| q.supports_compute())
        .expect("Couldn't find a compute queue family.");

    let device_extensions = vulkano::device::DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..vulkano::device::DeviceExtensions::none()
    };

    let (device, mut queues) = vulkano::device::Device::new(
        physical_device,
        physical_device.supported_features(),
        &device_extensions,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .expect("failed to create device");

    let queue = queues.next().unwrap();

    BackendContext::new(device, queue)
}

pub fn is_available(device_id: usize) -> bool {
    matches!(
        vulkano::instance::PhysicalDevice::from_index(INSTANCE.borrow(), device_id),
        Some(_)
    )
}

pub fn get_context(device_id: usize) -> Arc<BackendContext> {
    let mut map = CONTEXTS.lock().unwrap();
    map.borrow_mut()
        .entry(device_id)
        .or_insert_with(|| Arc::new(create_context(device_id)))
        .clone()
}

pub struct VulkanBuffer<T> {
    device_id: usize,
    data: Arc<CpuAccessibleBuffer<[T]>>,
    size: usize,
}

impl<T> VulkanBuffer<T>
where
    T: Element,
{
    pub fn uninit(size: usize, ctx: &BackendContext) -> Self {
        unsafe {
            VulkanBuffer {
                device_id: ctx.device_id(),
                size,
                data: CpuAccessibleBuffer::uninitialized_array(
                    ctx.device.clone(),
                    size,
                    BufferUsage::all(),
                    false,
                )
                .unwrap(),
            }
        }
    }

    pub fn from_iter<I>(data: I, ctx: &BackendContext) -> Self
    where
        I: ExactSizeIterator<Item = T>,
    {
        VulkanBuffer {
            device_id: ctx.device_id(),
            size: data.len(),
            data: CpuAccessibleBuffer::from_iter(
                ctx.device.clone(),
                BufferUsage::all(),
                false,
                data,
            )
            .unwrap(),
        }
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.data.read().unwrap().to_vec()
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

pub struct BackendContext {
    device: Arc<vulkano::device::Device>,
    queue: Arc<vulkano::device::Queue>,

    unary_ops_copy: Arc<ComputePipeline<PipelineLayout<shader::unary_ops::Layout>>>,
    unary_ops_neg: Arc<ComputePipeline<PipelineLayout<shader::unary_ops::Layout>>>,

    binary_ops_add: Arc<ComputePipeline<PipelineLayout<shader::binary_ops::Layout>>>,
    binary_ops_sub: Arc<ComputePipeline<PipelineLayout<shader::binary_ops::Layout>>>,

    tensor_contraction: Arc<ComputePipeline<PipelineLayout<shader::tensor_contraction::Layout>>>,
    tensor_reduction_add: Arc<ComputePipeline<PipelineLayout<shader::tensor_reduction::Layout>>>,
    tensor_reduction_mul: Arc<ComputePipeline<PipelineLayout<shader::tensor_reduction::Layout>>>,

    tensor_reduction_index:
        Arc<ComputePipeline<PipelineLayout<shader::tensor_reduction_index::Layout>>>,
    tensor_reduction_index_max:
        Arc<ComputePipeline<PipelineLayout<shader::tensor_reduction_index::Layout>>>,
    tensor_reduction_index_min:
        Arc<ComputePipeline<PipelineLayout<shader::tensor_reduction_index::Layout>>>,

    im2col: Arc<ComputePipeline<PipelineLayout<shader::im2col::Layout>>>,
    col2im: Arc<ComputePipeline<PipelineLayout<shader::col2im::Layout>>>,
}

impl BackendContext {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device: device.clone(),
            queue,
            unary_ops_copy: create_pipeline!(unary_ops, unary_ops_copy, device),
            unary_ops_neg: create_pipeline!(unary_ops, unary_ops_neg, device),
            binary_ops_add: create_pipeline!(binary_ops, binary_ops_add, device),
            binary_ops_sub: create_pipeline!(binary_ops, binary_ops_sub, device),
            tensor_contraction: create_pipeline!(tensor_contraction, tensor_contraction, device),
            tensor_reduction_add: create_pipeline!(tensor_reduction, tensor_reduction_add, device),
            tensor_reduction_mul: create_pipeline!(tensor_reduction, tensor_reduction_mul, device),

            tensor_reduction_index: create_pipeline!(
                tensor_reduction_index,
                tensor_reduction_index,
                device
            ),
            tensor_reduction_index_max: create_pipeline!(
                tensor_reduction_index,
                tensor_reduction_index_max,
                device
            ),
            tensor_reduction_index_min: create_pipeline!(
                tensor_reduction_index,
                tensor_reduction_index_min,
                device
            ),

            im2col: create_pipeline!(im2col, im2col, device),
            col2im: create_pipeline!(col2im, col2im, device),
        }
    }

    pub fn device_id(&self) -> usize {
        self.device.physical_device().index()
    }

    pub fn alloc_mem<T>(&self, size: usize) -> VulkanBuffer<T>
    where
        T: Element,
    {
        VulkanBuffer::uninit(size, self)
    }

    pub fn alloc_mem_from_iter<I, T>(&self, data: I) -> VulkanBuffer<T>
    where
        T: Element,
        I: ExactSizeIterator<Item = T>,
    {
        VulkanBuffer::from_iter(data, self)
    }

    pub fn unary_op_pipeline<T>(
        &self,
        input: &Tensor<T>,
        pipeline: Arc<ComputePipeline<PipelineLayout<shader::unary_ops::Layout>>>,
    ) -> Tensor<T>
    where
        T: Element,
    {
        // create output tensor
        let buffer = self.alloc_mem(input.size());
        let output = Tensor::from_buffer(input.shape(), Buffer::Vulkan(buffer));

        // create output
        let buffer_in = &input.buffer.as_vulkan().data;
        let buffer_out = &output.buffer.as_vulkan().data;

        let layout = pipeline.descriptor_set_layout(0).unwrap();

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(buffer_out.clone())
                .unwrap()
                .add_buffer(buffer_in.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let push_constants = shader::unary_ops::ty::PushConstants {
            num_dim: input.order() as u32,
            offsets: [0, input.offset() as u32],
            strides: interleave_strides(&[output.strides(), input.strides()]),
        };

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )
        .unwrap();

        builder
            .dispatch(
                [output.size() as u32, 1, 1],
                // pipeline
                pipeline.clone(),
                // descriptor set
                descriptor_set,
                // constants
                push_constants,
                vec![],
            )
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let finished = command_buffer.execute(self.queue.clone()).unwrap();

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        output
    }

    pub fn unary_op<T>(&self, input: &Tensor<T>, op: UnaryOperation) -> Tensor<T>
    where
        T: Element,
    {
        let pipeline = match op {
            UnaryOperation::Neg => self.unary_ops_neg.clone(),
            _ => todo!(),
        };

        self.unary_op_pipeline(input, pipeline)
    }

    pub fn copy<T>(&self, input: &Tensor<T>) -> Tensor<T>
    where
        T: Element,
    {
        let pipeline = self.unary_ops_copy.clone();
        self.unary_op_pipeline(input, pipeline)
    }

    pub fn copy_to<T>(&self, input: &Tensor<T>, dest: &Tensor<T>)
    where
        T: Element,
    {
        assert_eq!(input.shape(), dest.shape());

        let pipeline = self.unary_ops_copy.clone();

        // create output
        let buffer_in = &input.buffer.as_vulkan().data;
        let buffer_out = &dest.buffer.as_vulkan().data;

        let layout = pipeline.descriptor_set_layout(0).unwrap();

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(buffer_out.clone())
                .unwrap()
                .add_buffer(buffer_in.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let push_constants = shader::unary_ops::ty::PushConstants {
            num_dim: input.order() as u32,
            offsets: [dest.offset() as u32, input.offset() as u32],
            strides: interleave_strides(&[dest.strides(), input.strides()]),
        };

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )
        .unwrap();

        builder
            .dispatch(
                [input.size() as u32, 1, 1],
                // pipeline
                pipeline.clone(),
                // descriptor set
                descriptor_set,
                // constants
                push_constants,
                vec![],
            )
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let finished = command_buffer.execute(self.queue.clone()).unwrap();

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    pub fn binary_op_pipeline<T>(
        &self,
        input1: &Tensor<T>,
        input2: &Tensor<T>,
        pipeline: Arc<ComputePipeline<PipelineLayout<shader::binary_ops::Layout>>>,
    ) -> Tensor<T>
    where
        T: Element,
    {
        if input1.shape() != input2.shape() {
            panic!("all tensors should be in the same shape");
        }

        // create output tensor
        let buffer = self.alloc_mem(input1.size());
        let output = Tensor::from_buffer(input1.shape(), Buffer::Vulkan(buffer));

        // create output
        let buffer_in1 = &input1.buffer.as_vulkan().data;
        let buffer_in2 = &input2.buffer.as_vulkan().data;
        let buffer_out = &output.buffer.as_vulkan().data;

        let layout = pipeline.descriptor_set_layout(0).unwrap();

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(buffer_out.clone())
                .unwrap()
                .add_buffer(buffer_in1.clone())
                .unwrap()
                .add_buffer(buffer_in2.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let push_constants = shader::binary_ops_add::ty::PushConstants {
            num_dim: input1.order() as u32,
            offsets: [0, input1.offset() as u32, input2.offset() as u32],
            strides: interleave_strides::<24>(&[
                output.strides(),
                input1.strides(),
                input2.strides(),
            ]),
        };

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )
        .unwrap();

        builder
            .dispatch(
                [output.size() as u32, 1, 1],
                // pipeline
                pipeline.clone(),
                // descriptor set
                descriptor_set,
                // constants
                push_constants,
                vec![],
            )
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let finished = command_buffer.execute(self.queue.clone()).unwrap();

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        output
    }

    pub fn binary_op<T>(
        &self,
        input1: &Tensor<T>,
        input2: &Tensor<T>,
        op: BinaryOperation,
    ) -> Tensor<T>
    where
        T: Element,
    {
        let pipeline = match op {
            BinaryOperation::Add => self.binary_ops_add.clone(),
            BinaryOperation::Sub => self.binary_ops_sub.clone(),
            _ => {
                unimplemented!()
            }
        };
        self.binary_op_pipeline(input1, input2, pipeline)
    }

    pub fn binary_logic_op<T>(
        &self,
        input1: &Tensor<T>,
        input2: &Tensor<T>,
        op: BinaryLogicOperation,
    ) -> Tensor<T>
    where
        T: Element,
    {
        let pipeline = match op {
            BinaryLogicOperation::Max => self.binary_ops_add.clone(),
            _ => {
                unimplemented!()
            }
        };
        self.binary_op_pipeline(input1, input2, pipeline)
    }

    pub fn reduction_op_pipeline<T>(
        &self,
        input: &Tensor<T>,
        pipeline: Arc<ComputePipeline<PipelineLayout<shader::tensor_reduction::Layout>>>,
        axes: &[usize],
    ) -> Tensor<T>
    where
        T: Element,
    {
        let (reduced_layout, preserved_layout) = input.mem_layout.split(axes, false);
        let (reduced_layoutd, preserved_layoutd) =
            MemoryLayout::with_default(input.shape()).split(axes, false);

        let buffer = self.alloc_mem(preserved_layout.size());
        let output = Tensor::from_buffer(preserved_layout.extents(), Buffer::Vulkan(buffer));

        let buffer_in = &input.buffer.as_vulkan().data;
        let buffer_out = &output.buffer.as_vulkan().data;

        let layout = pipeline.descriptor_set_layout(0).unwrap();

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(buffer_out.clone())
                .unwrap()
                .add_buffer(buffer_in.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let push_constants = shader::tensor_reduction::ty::PushConstants {
            offset_out: 0,
            offset_in: preserved_layout.offset as u32,
            ndim_red: reduced_layout.num_axes() as u32,
            ndim_pre: preserved_layout.num_axes() as u32,
            size_red: reduced_layout.size() as u32,
            stride_pre: vec_to_sized_u32(reduced_layoutd.strides.clone()),
            stride_pre_ref: vec_to_sized_u32(preserved_layout.strides.clone()),
            stride_red: vec_to_sized_u32(reduced_layoutd.strides),
            stride_red_ref: vec_to_sized_u32(reduced_layout.strides),
        };

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )
        .unwrap();

        builder
            .dispatch(
                [preserved_layout.size() as u32, 1, 1],
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

        let finished = command_buffer.execute(self.queue.clone()).unwrap();

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        output
    }

    pub fn reduction_op<T>(
        &self,
        input: &Tensor<T>,
        op: BinaryOperation,
        axes: &[usize],
    ) -> Tensor<T>
    where
        T: Element,
    {
        let pipeline = match op {
            BinaryOperation::Add => self.tensor_reduction_add.clone(),
            BinaryOperation::Mul => self.tensor_reduction_mul.clone(),
            _ => {
                unimplemented!()
            }
        };
        self.reduction_op_pipeline(input, pipeline, axes)
    }

    pub fn reduction_logic_op<T>(
        &self,
        input: &Tensor<T>,
        op: BinaryLogicOperation,
        axes: &[usize],
    ) -> Tensor<T>
    where
        T: Element,
    {
        let pipeline = match op {
            BinaryLogicOperation::Max => self.tensor_reduction_add.clone(),
            _ => {
                unimplemented!()
            }
        };
        self.reduction_op_pipeline(input, pipeline, axes)
    }

    pub fn reduction_index_op_pipeline<T>(
        &self,
        input: &Tensor<T>,
        pipeline: Arc<ComputePipeline<PipelineLayout<shader::tensor_reduction_index::Layout>>>,
        axis: usize,
    ) -> Tensor<u32>
    where
        T: Element,
    {
        let (reduced_layout, preserved_layout) = input.mem_layout.split(&[axis], false);
        let (reduced_layoutd, preserved_layoutd) =
            MemoryLayout::with_default(input.shape()).split(&[axis], false);

        let buffer = self.alloc_mem(preserved_layout.size());
        let output = Tensor::from_buffer(preserved_layout.extents(), Buffer::Vulkan(buffer));

        let buffer_in = &input.buffer.as_vulkan().data;
        let buffer_out = &output.buffer.as_vulkan().data;

        let layout = pipeline.descriptor_set_layout(0).unwrap();

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(buffer_out.clone())
                .unwrap()
                .add_buffer(buffer_in.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let push_constants = shader::tensor_reduction_index::ty::PushConstants {
            offset_out: 0,
            offset_in: preserved_layout.offset as u32,
            ndim_pre: preserved_layout.num_axes() as u32,
            size_red: reduced_layout.size() as u32,
            stride_pre: vec_to_sized_u32(preserved_layoutd.strides.clone()),
            stride_pre_ref: vec_to_sized_u32(preserved_layout.strides.clone()),
            stride_red: reduced_layoutd.strides[0] as u32,
            stride_red_ref: reduced_layout.strides[0] as u32,
        };

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )
        .unwrap();

        builder
            .dispatch(
                [preserved_layout.size() as u32, 1, 1],
                // pipeline`
                pipeline.clone(),
                // descriptor set
                descriptor_set,
                // constants
                push_constants,
                vec![],
            )
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let finished = command_buffer.execute(self.queue.clone()).unwrap();

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        output
    }

    pub fn reduction_index_op<T>(
        &self,
        input: &Tensor<T>,
        op: BinaryIndexOperation,
        axis: usize,
    ) -> Tensor<u32>
    where
        T: Element,
    {
        let pipeline = match op {
            BinaryIndexOperation::Max => self.tensor_reduction_index_max.clone(),
            BinaryIndexOperation::Min => self.tensor_reduction_index_min.clone(),
        };

        self.reduction_index_op_pipeline(input, pipeline, axis)
    }


    pub fn contraction<T>(
        &self,
        input1: &Tensor<T>,
        input2: &Tensor<T>,
        axes1: &[usize],
        axes2: &[usize],
    ) -> Tensor<T>
    where
        T: Element,
    {
        let (reduced_layout1, preserved_layout1) = input1.mem_layout.split(axes1, false);
        let (reduced_layout2, preserved_layout2) = input2.mem_layout.split(axes2, false);

        let (reduced_layoutd, preserved_layout1d) =
            MemoryLayout::with_default(input1.shape()).split(axes1, false);
        let (_, preserved_layout2d) =
            MemoryLayout::with_default(input2.shape()).split(axes2, false);

        assert_eq!(reduced_layout1.extents(), reduced_layout2.extents());

        let mut output_shape = Vec::<usize>::new();
        output_shape.extend(preserved_layout1.extents());
        output_shape.extend(preserved_layout2.extents());

        let buffer = self.alloc_mem(output_shape.iter().product());
        let output = Tensor::from_buffer(output_shape, Buffer::Vulkan(buffer));

        let buffer_in1 = &input1.buffer.as_vulkan().data;
        let buffer_in2 = &input2.buffer.as_vulkan().data;
        let buffer_out = &output.buffer.as_vulkan().data;

        let pipeline = self.tensor_contraction.clone();

        let layout = pipeline.descriptor_set_layout(0).unwrap();

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(buffer_out.clone())
                .unwrap()
                .add_buffer(buffer_in1.clone())
                .unwrap()
                .add_buffer(buffer_in2.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let push_constants = shader::tensor_contraction::ty::PushConstants {
            offset_out: 0,
            offset_in1: preserved_layout1.offset() as u32,
            offset_in2: preserved_layout2.offset() as u32,
            ndim_cont: reduced_layout1.num_axes() as u32,
            ndim_pre1: preserved_layout1.num_axes() as u32,
            ndim_pre2: preserved_layout2.num_axes() as u32,
            size_cont: reduced_layout1.size() as u32,
            stride_pre1: vec_to_sized_u32(preserved_layout1d.strides),
            stride_pre2: vec_to_sized_u32(preserved_layout2d.strides),
            stride_pre_ref1: vec_to_sized_u32(preserved_layout1.strides.clone()),
            stride_pre_ref2: vec_to_sized_u32(preserved_layout2.strides.clone()),
            stride_cont: vec_to_sized_u32(reduced_layoutd.strides),
            stride_cont_ref1: vec_to_sized_u32(reduced_layout1.strides),
            stride_cont_ref2: vec_to_sized_u32(reduced_layout2.strides),
        };

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )
        .unwrap();

        builder
            .dispatch(
                [
                    preserved_layout1.size() as u32,
                    preserved_layout2.size() as u32,
                    1,
                ],
                pipeline.clone(),
                descriptor_set,
                push_constants,
                vec![],
            )
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let finished = command_buffer.execute(self.queue.clone()).unwrap();

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        output
    }

    pub fn concat<T>(&self, inputs: &[&Tensor<T>], axis: usize) -> Tensor<T>
    where
        T: Element,
    {
        if !inputs
            .iter()
            .map(|t| (&t.shape()[..axis], &t.shape()[axis + 1..]))
            .all_equal()
        {
            panic!("does not match shape");
        }

        let (reduced_list, preserved_list): (Vec<_>, Vec<_>) = inputs
            .iter()
            .map(|t| t.mem_layout.split(&[axis], true))
            .unzip();

        let concat_size = reduced_list.iter().map(|p| p.extents()[0]).sum::<usize>();

        let mut output_shape = Vec::new();
        output_shape.extend(&inputs[0].shape()[0..axis]);
        output_shape.push(concat_size);
        output_shape.extend(&inputs[0].shape()[axis + 1..]);

        let buffer = self.alloc_mem(preserved_list[0].size() * concat_size);
        let output = Tensor::from_buffer(output_shape, Buffer::Vulkan(buffer));

        let mut offset = 0;
        for i in 0..inputs.len() {
            let size = reduced_list[i].extents()[0];
            let dest = output.slice(offset, offset + size, axis);
            self.copy_to(inputs[i], &dest);
            offset += size;
        }

        output
    }

    pub fn im2col<T>(
        &self,
        input: &Tensor<T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Tensor<T>
    where
        T: Element,
    {
        // assert output size
        assert_eq!(input.order(), 4);

        let batch_size = input.shape()[0];
        let num_channels = input.shape()[1];
        let img_h = input.shape()[2];
        let img_w = input.shape()[3];

        let output_size = (
            (img_w + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1) / stride.0 + 1,
            (img_h + 2 * padding.1 - dilation.1 * (kernel_size.1 - 1) - 1) / stride.1 + 1,
        );

        let img_len = img_w * img_h;
        let kernel_len = kernel_size.0 * kernel_size.1;
        let output_len = output_size.0 * output_size.1;

        let buffer = self.alloc_mem(batch_size * output_len * kernel_len);
        let output_shape = [
            batch_size,
            num_channels,
            output_size.1,
            output_size.0,
            kernel_size.1,
            kernel_size.0,
        ];
        let output = Tensor::from_buffer(output_shape, Buffer::Vulkan(buffer));

        let buffer_in = &input.buffer.as_vulkan().data;
        let buffer_out = &output.buffer.as_vulkan().data;

        let pipeline = self.im2col.clone();

        let layout = pipeline.descriptor_set_layout(0).unwrap();

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(buffer_out.clone())
                .unwrap()
                .add_buffer(buffer_in.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let push_constants = shader::im2col::ty::PushConstants {
            image_size: [img_w as u32, img_h as u32],
            //image_strides: [1, img_len, img_len * num_channels],
            output_size: [output_size.0 as u32, output_size.1 as u32],
            // output_strides: [
            //     kernel_len,
            //     kernel_len * output_len,
            //     kernel_len * output_len * num_channels,
            // ],
            kernel_size: [kernel_size.0 as u32, kernel_size.1 as u32],
            stride: [stride.0 as u32, stride.1 as u32],
            padding: [padding.0 as u32, padding.1 as u32],
            dilation: [dilation.0 as u32, dilation.1 as u32],
        };

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )
        .unwrap();

        builder
            .dispatch(
                [output_len as u32, num_channels as u32, batch_size as u32],
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

        let finished = command_buffer.execute(self.queue.clone()).unwrap();

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        output
    }

    pub fn col2im<T>(
        &self,
        input: &Tensor<T>,
        output_size: (usize, usize),
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Tensor<T>
    where
        T: Element + Float,
    {
        // assert output size

        let batch_size = input.shape()[0];
        let num_channels = input.shape()[1];
        let input_h = input.shape()[2];
        let input_w = input.shape()[3];
        let kernel_h = input.shape()[4];
        let kernel_w = input.shape()[5];

        // output.shape = (N, C, H, W)
        let input_len = input_w * input_h;
        let kernel_len = kernel_size.0 * kernel_size.1;
        let output_len = output_size.0 * output_size.1;

        let buffer = self.alloc_mem(batch_size * output_len);
        let output_shape = [batch_size, num_channels, output_size.1, output_size.0];
        let output = Tensor::from_buffer(output_shape, Buffer::Vulkan(buffer));

        let buffer_in = &input.buffer.as_vulkan().data;
        let buffer_out = &output.buffer.as_vulkan().data;

        let pipeline = self.col2im.clone();

        let layout = pipeline.descriptor_set_layout(0).unwrap();

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(buffer_out.clone())
                .unwrap()
                .add_buffer(buffer_in.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        // input.shape = (N, C, OH, OW, KH, KW)

        let push_constants = shader::col2im::ty::PushConstants {
            image_size: [output_size.0 as u32, output_size.1 as u32],
            output_size: [input_w as u32, input_h as u32],
            kernel_size: [kernel_size.0 as u32, kernel_size.1 as u32],
            stride: [stride.0 as u32, stride.1 as u32],
            padding: [padding.0 as u32, padding.1 as u32],
            dilation: [dilation.0 as u32, dilation.1 as u32],
        };

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )
        .unwrap();

        builder
            .dispatch(
                [input_len as u32, num_channels as u32, batch_size as u32],
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

        let finished = command_buffer.execute(self.queue.clone()).unwrap();

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        output
    }
}

pub fn interleave_strides<const N: usize>(strides: &[&[usize]]) -> [u32; N] {
    let mut out = [0; N];
    for i in 0..strides[0].len() {
        for j in 0..strides.len() {
            out[i * strides.len() + j] = strides[j][i] as u32;
        }
    }
    out
}

fn vec_to_sized_u32<const C: usize>(v: Vec<usize>) -> [u32; C] {
    let mut out = [0; C];
    for i in 0..min(v.len(), C) {
        out[i] = v[i] as u32;
    }
    out
}
