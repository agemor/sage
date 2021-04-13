use crate::backend::native::NativeBuffer;
use crate::backend::vulkan::VulkanBuffer;
use crate::ops::{BinaryIndexOperation, BinaryLogicOperation, BinaryOperation, UnaryOperation};
use crate::tensor::{Element, Tensor};
use num_traits::{Float, NumOps, Pow, WrappingNeg};
use std::iter::Sum;
use thiserror::Error;

mod native;
#[macro_use]
mod vulkan;

#[derive(Error, Debug)]
pub enum BackendError {
    #[error("specified backend is not available")]
    NotAvailable,
}

#[derive(Clone)]
pub enum Backend {
    Native,
    Vulkan(usize),
}

impl Backend {
    pub fn native() -> Backend {
        Backend::Native
    }

    pub fn assign<T>(input: &Tensor<T>, backend: Backend) -> Tensor<T>
    where
        T: Element,
    {
        match (input.backend(), backend) {
            (Backend::Native, Backend::Vulkan(n)) => todo!(),
            (Backend::Vulkan(_), Backend::Native) => Tensor::from_buffer(
                input.shape(),
                Buffer::Native(NativeBuffer::from_vec(input.to_vec())),
            ),
            (_, _) => input.twin(),
        }
    }

    pub fn is_available(&self) -> bool {
        match self {
            Backend::Native => true,
            Backend::Vulkan(n) => vulkan::is_available(*n),
        }
    }

    pub fn alloc_mem<T>(&self, size: usize) -> Buffer<T>
    where
        T: Element,
    {
        match self {
            Backend::Native => Buffer::Native(native::alloc_mem(size)),
            Backend::Vulkan(n) => Buffer::Vulkan(vulkan::get_context(*n).alloc_mem(size)),
        }
    }

    pub fn alloc_mem_from_iter<I, T>(&self, data: I) -> Buffer<T>
    where
        T: Element,
        I: ExactSizeIterator<Item = T>,
    {
        match self {
            Backend::Native => Buffer::Native(native::alloc_mem_from_iter(data)),
            Backend::Vulkan(n) => Buffer::Vulkan(vulkan::get_context(*n).alloc_mem_from_iter(data)),
        }
    }

    pub fn copy<T>(&self, input: &Tensor<T>) -> Tensor<T>
    where
        T: Element,
    {
        match self {
            Backend::Native => native::copy(input),
            Backend::Vulkan(n) => vulkan::get_context(*n).copy(input),
        }
    }

    pub fn concat<T>(&self, inputs: &[&Tensor<T>], axis: usize) -> Tensor<T>
    where
        T: Element,
    {
        match self {
            Backend::Native => native::concat(inputs, axis),
            Backend::Vulkan(n) => vulkan::get_context(*n).concat(inputs, axis),
        }
    }

    pub fn unary_op<T>(&self, input: &Tensor<T>, op: UnaryOperation) -> Tensor<T>
    where
        T: Element + Float,
    {
        match self {
            Backend::Native => native::unary_op(input, op),
            Backend::Vulkan(n) => vulkan::get_context(*n).unary_op(input, op),
        }
    }

    pub fn binary_op<T>(
        &self,
        input1: &Tensor<T>,
        input2: &Tensor<T>,
        op: BinaryOperation,
    ) -> Tensor<T>
    where
        T: Element + NumOps,
    {
        match self {
            Backend::Native => native::binary_op(input1, input2, op),
            Backend::Vulkan(n) => vulkan::get_context(*n).binary_op(input1, input2, op),
        }
    }

    pub fn binary_logic_op<T>(
        &self,
        input1: &Tensor<T>,
        input2: &Tensor<T>,
        op: BinaryLogicOperation,
    ) -> Tensor<T>
    where
        T: Element + PartialOrd,
    {
        match self {
            Backend::Native => native::binary_logic_op(input1, input2, op),
            Backend::Vulkan(n) => vulkan::get_context(*n).binary_logic_op(input1, input2, op),
        }
    }

    pub fn reduction_op<T>(
        &self,
        input: &Tensor<T>,
        op: BinaryOperation,
        axes: &[usize],
    ) -> Tensor<T>
    where
        T: Element + NumOps,
    {
        match self {
            Backend::Native => native::reduction_op(input, op, axes),
            Backend::Vulkan(n) => vulkan::get_context(*n).reduction_op(input, op, axes),
        }
    }

    pub fn reduction_logic_op<T>(
        &self,
        input: &Tensor<T>,
        op: BinaryLogicOperation,
        axes: &[usize],
    ) -> Tensor<T>
    where
        T: Element + PartialOrd,
    {
        match self {
            Backend::Native => native::reduction_logic_op(input, op, axes),
            Backend::Vulkan(n) => vulkan::get_context(*n).reduction_logic_op(input, op, axes),
        }
    }

    pub fn reduction_index_op<T>(
        &self,
        input: &Tensor<T>,
        op: BinaryIndexOperation,
        axis: usize,
    ) -> Tensor<u32>
    where
        T: Element + PartialOrd,
    {
        match self {
            Backend::Native => native::reduction_index(input, op, axis),
            Backend::Vulkan(n) => vulkan::get_context(*n).reduction_index_op(input, op, axis),
        }
    }

    pub fn contraction<T>(
        &self,
        input1: &Tensor<T>,
        input2: &Tensor<T>,
        axes1: &[usize],
        axes2: &[usize],
    ) -> Tensor<T>
    where
        T: Element + NumOps + Sum,
    {
        match self {
            Backend::Native => native::contraction(input1, input2, axes1, axes2),
            Backend::Vulkan(n) => vulkan::get_context(*n).contraction(input1, input2, axes1, axes2),
        }
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
        T: Element + Float,
    {
        match self {
            Backend::Native => native::im2col(input, kernel_size, stride, padding, dilation),
            Backend::Vulkan(n) => {
                vulkan::get_context(*n).im2col(input, kernel_size, stride, padding, dilation)
            }
        }
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
        match self {
            Backend::Native => {
                native::col2im(input, output_size, kernel_size, stride, padding, dilation)
            }

            Backend::Vulkan(n) => vulkan::get_context(*n).col2im(
                input,
                output_size,
                kernel_size,
                stride,
                padding,
                dilation,
            ),
        }
    }
}

pub enum Buffer<T> {
    Native(NativeBuffer<T>),
    Vulkan(VulkanBuffer<T>),
}

impl<T> Buffer<T>
where
    T: Element,
{
    pub fn backend(&self) -> Backend {
        match self {
            Buffer::Native(_) => Backend::Native,
            Buffer::Vulkan(b) => Backend::Vulkan(b.device_id()),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Buffer::Native(b) => b.size(),
            Buffer::Vulkan(b) => b.size(),
        }
    }

    pub fn same_kind(&self, other: &Buffer<T>) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    pub fn to_native(&self) -> Buffer<T> {
        let buffer = match self {
            Buffer::Native(b) => b.clone(),
            Buffer::Vulkan(b) => NativeBuffer::from_vec(b.to_vec()),
        };
        Buffer::Native(buffer)
    }

    pub fn as_native(&self) -> &NativeBuffer<T> {
        match self {
            Buffer::Native(b) => b,
            _ => panic!("convert the buffer type first!"),
        }
    }

    pub fn as_vulkan(&self) -> &VulkanBuffer<T> {
        match self {
            Buffer::Vulkan(b) => b,
            _ => panic!("convert the buffer type first!"),
        }
    }
}

impl<T> Buffer<T>
where
    T: Element,
{
    pub fn to_vec(&self) -> Vec<T> {
        match self {
            Buffer::Native(b) => b.data().to_vec(),
            Buffer::Vulkan(b) => b.to_vec(),
        }
    }
}
