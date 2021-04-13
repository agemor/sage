use crate::backend::Buffer;
use crate::ops::{BinaryIndexOperation, BinaryLogicOperation, BinaryOperation, UnaryOperation};
use crate::tensor::{Element, Tensor};
use itertools::Itertools;
use num_traits::{Float, NumOps, Pow};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::cell::UnsafeCell;
use std::cmp::Ordering;
use std::iter::Sum;

impl UnaryOperation {
    pub fn to_fn<T>(&self) -> fn(T) -> T
    where
        T: Element + Float,
    {
        match self {
            UnaryOperation::Neg => |x| -x,
            UnaryOperation::Abs => |x| x.abs(),
            UnaryOperation::Recip => |x| x.recip(),
            UnaryOperation::Sqrt => |x| x.sqrt(),
            UnaryOperation::Rsqrt => |x| x.sqrt().recip(),
            UnaryOperation::Square => |x| x.powi(2),
            UnaryOperation::Exp => |x| x.exp(),
            UnaryOperation::Expm1 => |x| x.exp() - T::one(),
            UnaryOperation::Log => |x| x.ln(),
            UnaryOperation::Log1p => |x| (T::one() + x).ln(),
            UnaryOperation::Ceil => |x| x.ceil(),
            UnaryOperation::Floor => |x| x.floor(),
            UnaryOperation::Round => |x| x.round(),
            UnaryOperation::Sign => |x| x.signum(),
            UnaryOperation::Sin => |x| x.sin(),
            UnaryOperation::Sinh => |x| x.sinh(),
            UnaryOperation::Asinh => |x| x.asinh(),
            UnaryOperation::Asin => |x| x.asin(),
            UnaryOperation::Cos => |x| x.cos(),
            UnaryOperation::Cosh => |x| x.cosh(),
            UnaryOperation::Acosh => |x| x.acosh(),
            UnaryOperation::Acos => |x| x.acos(),
            UnaryOperation::Tan => |x| x.tan(),
            UnaryOperation::Tanh => |x| x.tanh(),
            UnaryOperation::Atan => |x| x.atan(),
            UnaryOperation::Atanh => |x| x.atanh(),
            UnaryOperation::Sigmoid => |x| T::one() / (T::one() + (-x).exp()),
        }
    }
}

impl BinaryOperation {
    pub fn to_fn<T>(&self) -> fn(T, T) -> T
    where
        T: Element + NumOps,
    {
        match self {
            BinaryOperation::Add => |x1, x2| x1 + x2,
            BinaryOperation::Sub => |x1, x2| x1 - x2,
            BinaryOperation::Mul => |x1, x2| x1 * x2,
            BinaryOperation::Div => |x1, x2| x1 / x2,
            BinaryOperation::Squidiff => |x1, x2| (x1 - x2) * (x1 - x2),
        }
    }
}

impl BinaryLogicOperation {
    pub fn to_fn<T>(&self) -> fn(T, T) -> T
    where
        T: Element + PartialOrd,
    {
        match self {
            BinaryLogicOperation::Max => |x1, x2| if x1 > x2 { x1 } else { x2 },
            BinaryLogicOperation::Min => |x1, x2| if x1 < x2 { x1 } else { x2 },
            BinaryLogicOperation::Eq => |x1, x2| if x1 == x2 { T::one() } else { T::zero() },
            BinaryLogicOperation::NotEq => |x1, x2| if x1 != x2 { T::one() } else { T::zero() },
            BinaryLogicOperation::GreaterThan => {
                |x1, x2| if x1 > x2 { T::one() } else { T::zero() }
            }
            BinaryLogicOperation::LessThan => |x1, x2| if x1 < x2 { T::one() } else { T::zero() },
            BinaryLogicOperation::GreaterOrEq => {
                |x1, x2| if x1 >= x2 { T::one() } else { T::zero() }
            }
            BinaryLogicOperation::LessOrEq => |x1, x2| if x1 <= x2 { T::one() } else { T::zero() },
        }
    }
}

impl BinaryIndexOperation {
    pub fn to_fn<T>(&self) -> fn(T, T) -> Ordering
    where
        T: Element + PartialOrd,
    {
        match self {
            BinaryIndexOperation::Max => |x1, x2| x1.partial_cmp(&x2).unwrap(),
            BinaryIndexOperation::Min => |x1, x2| x2.partial_cmp(&x1).unwrap(),
        }
    }
}

pub struct NativeBuffer<T> {
    data: UnsafeCell<Vec<T>>,
}

impl<T> NativeBuffer<T>
where
    T: Element,
{
    pub fn uninit(size: usize) -> Self {
        let mut v = Vec::with_capacity(size);

        // uninitialized buffer
        unsafe {
            v.set_len(size);
        }
        NativeBuffer {
            data: UnsafeCell::new(v),
        }
    }

    pub fn from_iter<I>(data: I) -> Self
    where
        I: ExactSizeIterator<Item = T>,
    {
        NativeBuffer {
            data: UnsafeCell::new(data.collect_vec()),
        }
    }

    pub fn from_vec(vec: Vec<T>) -> Self {
        NativeBuffer {
            data: UnsafeCell::new(vec),
        }
    }

    pub fn size(&self) -> usize {
        self.data().len()
    }

    pub fn data<'a>(&self) -> &'a Vec<T> {
        unsafe { self.data.get().as_ref().unwrap() }
    }

    pub fn data_mut<'a>(&self) -> &'a mut Vec<T> {
        unsafe { self.data.get().as_mut().unwrap() }
    }
}

impl<T> Clone for NativeBuffer<T>
where
    T: Element,
{
    fn clone(&self) -> Self {
        NativeBuffer::from_vec(self.data().to_vec())
    }
}

pub fn alloc_mem<T>(size: usize) -> NativeBuffer<T>
where
    T: Element,
{
    NativeBuffer::uninit(size)
}

pub fn alloc_mem_from_iter<I, T>(data: I) -> NativeBuffer<T>
where
    T: Element,
    I: ExactSizeIterator<Item = T>,
{
    NativeBuffer::from_iter(data)
}

pub fn copy<T>(input: &Tensor<T>) -> Tensor<T>
where
    T: Element,
{
    let data_out = input.iter().map(|(_, v)| v).collect::<Vec<T>>();

    let buffer = Buffer::Native(NativeBuffer::from_vec(data_out));
    let output = Tensor::from_buffer(input.shape(), buffer);

    output
}

pub fn unary_fn<T, F>(input: &Tensor<T>, f: F) -> Tensor<T>
where
    T: Element,
    F: Fn(T) -> T + Sync,
{
    let data_out = input.par_iter().map(|(_, v)| f(v)).collect::<Vec<T>>();

    let buffer = Buffer::Native(NativeBuffer::from_vec(data_out));
    let output = Tensor::from_buffer(input.shape(), buffer);

    output
}

pub fn binary_fn<T, F>(input1: &Tensor<T>, input2: &Tensor<T>, f: F) -> Tensor<T>
where
    T: Element,
    F: Fn(T, T) -> T + Sync,
{
    let data_out = input1
        .par_iter()
        .zip(input2.par_iter())
        .map(|((_, v1), (_, v2))| f(v1, v2))
        .collect::<Vec<T>>();

    let buffer = Buffer::Native(NativeBuffer::from_vec(data_out));
    let output = Tensor::from_buffer(input1.shape(), buffer);

    output
}

pub fn reduction_fn<T, F>(input: &Tensor<T>, f: F, axes: &[usize]) -> Tensor<T>
where
    T: Element,
    F: Fn(T, T) -> T + Sync,
{
    let (reduced_layout, preserved_layout) = input.mem_layout.split(axes, true);

    let data_in = input.buffer.as_native().data();

    let data_out = preserved_layout
        .par_iter()
        .map(|i| {
            reduced_layout
                .iter()
                .map(|j| data_in[i + j])
                .fold1(|acc, v| f(acc, v))
                .unwrap()
        })
        .collect::<Vec<T>>();

    let buffer = Buffer::Native(NativeBuffer::from_vec(data_out));
    let output = Tensor::from_buffer(preserved_layout.extents(), buffer);

    output
}

pub fn reduction_index_fn<T, F>(input: &Tensor<T>, f: F, axis: usize) -> Tensor<u32>
where
    T: Element,
    F: Fn(T, T) -> Ordering + Sync,
{
    let (reduced_layout, preserved_layout) = input.mem_layout.split(&[axis], true);

    let data_in = input.buffer.as_native().data();

    let data_out = preserved_layout
        .par_iter()
        .map(|i| {
            let (index, _) = reduced_layout
                .iter()
                .enumerate()
                .map(|(index, j)| (index, data_in[i + j]))
                .max_by(|(_, v1), (_, v2)| f(*v1, *v2))
                .unwrap();
            index as u32
        })
        .collect::<Vec<u32>>();

    let buffer = Buffer::Native(NativeBuffer::from_vec(data_out));
    let output = Tensor::<u32>::from_buffer(preserved_layout.extents(), buffer);

    output
}

pub fn unary_op<T>(input: &Tensor<T>, op: UnaryOperation) -> Tensor<T>
where
    T: Element + Float,
{
    unary_fn(input, op.to_fn())
}

pub fn binary_op<T>(input1: &Tensor<T>, input2: &Tensor<T>, op: BinaryOperation) -> Tensor<T>
where
    T: Element + NumOps,
{
    binary_fn(input1, input2, op.to_fn())
}

pub fn binary_logic_op<T>(
    input1: &Tensor<T>,
    input2: &Tensor<T>,
    op: BinaryLogicOperation,
) -> Tensor<T>
where
    T: Element + PartialOrd,
{
    binary_fn(input1, input2, op.to_fn())
}

pub fn reduction_op<T>(input: &Tensor<T>, op: BinaryOperation, axes: &[usize]) -> Tensor<T>
where
    T: Element + NumOps,
{
    reduction_fn(input, op.to_fn(), axes)
}

pub fn reduction_logic_op<T>(
    input: &Tensor<T>,
    op: BinaryLogicOperation,
    axes: &[usize],
) -> Tensor<T>
where
    T: Element + PartialOrd,
{
    reduction_fn(input, op.to_fn(), axes)
}

pub fn reduction_index<T>(input: &Tensor<T>, op: BinaryIndexOperation, axis: usize) -> Tensor<u32>
where
    T: Element + PartialOrd,
{
    reduction_index_fn(input, op.to_fn(), axis)
}

pub fn contraction<T>(
    input1: &Tensor<T>,
    input2: &Tensor<T>,
    axes1: &[usize],
    axes2: &[usize],
) -> Tensor<T>
where
    T: Element + NumOps + Sum,
{
    let (reduced_layout1, preserved_layout1) = input1.mem_layout.split(axes1, true);
    let (reduced_layout2, preserved_layout2) = input2.mem_layout.split(axes2, true);

    assert_eq!(reduced_layout1.extents(), reduced_layout2.extents());

    let mut output_shape = Vec::<usize>::new();
    output_shape.extend(preserved_layout1.extents());
    output_shape.extend(preserved_layout2.extents());

    let data_in1 = input1.buffer.as_native().data();
    let data_in2 = input2.buffer.as_native().data();

    let data_out = preserved_layout1
        .par_iter()
        .flat_map(|i| preserved_layout2.par_iter().map(move |j| (i, j)))
        .map(|(i, j)| {
            reduced_layout1
                .iter()
                .zip(reduced_layout2.iter())
                .map(|(ir, jr)| data_in1[i + ir] * data_in2[j + jr])
                .sum()
        })
        .collect::<Vec<T>>();

    let buffer = Buffer::Native(NativeBuffer::from_vec(data_out));
    let output = Tensor::from_buffer(output_shape, buffer);

    output
}

pub fn concat<T>(inputs: &[&Tensor<T>], axis: usize) -> Tensor<T>
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

    let data_in_list = inputs
        .iter()
        .map(|t| t.buffer.as_native().data())
        .collect_vec();

    let data_out = data_in_list
        .iter()
        .zip(reduced_list.iter().zip(preserved_list.iter()))
        // layout -> (preserved indices, buffer)
        .flat_map(|(&data_in, (reduced, preserved))| {
            reduced.iter().map(move |i| (data_in, i, preserved))
        })
        // (preserved indices, buffer) -> (pres.., reduced.., buffer) -> (data)
        .flat_map(|(data_in, i, preserved)| preserved.iter().map(move |j| data_in[i + j]))
        .collect_vec();

    let concat_size = inputs.iter().map(|t| t.shape()[axis]).sum::<usize>();

    let mut output_shape = Vec::new();
    output_shape.extend(&inputs[0].shape()[0..axis]);
    output_shape.push(concat_size);
    output_shape.extend(&inputs[0].shape()[axis + 1..]);

    let buffer = Buffer::Native(NativeBuffer::from_vec(data_out));
    let output = Tensor::from_buffer(output_shape, buffer);

    output
}

// (N, C, H, W) -> (N, C, OH, OW, KH, KW) or (N, C, O, K)
pub fn im2col<T>(
    input: &Tensor<T>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Tensor<T>
where
    T: Element + Float,
{
    // (width, height)

    let batch_size = input.shape()[0];
    let num_channels = input.shape()[1];
    let input_size = (input.shape()[3], input.shape()[2]);

    let output_size = (
        (input_size.0 + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1) / stride.0 + 1,
        (input_size.1 + 2 * padding.1 - dilation.1 * (kernel_size.1 - 1) - 1) / stride.1 + 1,
    );

    let input_len = input_size.0 * input_size.1;
    let output_len = output_size.0 * output_size.1;
    let kernel_len = kernel_size.0 * kernel_size.1;

    let data_in = input.buffer.as_native().data();
    let mut data_out = Vec::with_capacity(batch_size * num_channels * output_len * kernel_len);

    let mut index_in_offset = input.offset();
    let mut index_out_offset = 0;

    for n in 0..batch_size {
        index_in_offset += n * num_channels * input_len;
        index_out_offset += n * num_channels * output_len * kernel_len;

        for c in 0..num_channels {
            index_in_offset += c * input_len;
            index_out_offset += c * output_len * kernel_len;

            for h in 0..output_size.1 {
                index_out_offset += h * output_size.0 * kernel_len;

                for w in 0..output_size.0 {
                    index_out_offset += w * kernel_len;

                    let input_x_offset = (w * stride.0) as isize - padding.0 as isize;
                    let input_y_offset = (h * stride.1) as isize - padding.1 as isize;

                    for kh in 0..kernel_size.1 {
                        for kw in 0..kernel_size.0 {
                            let input_x = (kw * dilation.0) as isize + input_x_offset;
                            let input_y = (kh * dilation.1) as isize + input_y_offset;

                            let mut val = T::zero();
                            if input_x >= 0
                                && input_y >= 0
                                && input_x < input_size.0 as isize
                                && input_y < input_size.1 as isize
                            {
                                let index_in = input_y as usize * input_size.0 * input_x as usize
                                    + index_in_offset;
                                val = data_in[index_in];
                            }
                            let index_out = kh * kernel_size.0 + kw + index_out_offset;
                            data_out[index_out] = val;
                        }
                    }
                }
            }
        }
    }

    let output_shape = [
        batch_size,
        num_channels,
        output_size.1,
        output_size.0,
        kernel_size.1,
        kernel_size.0,
    ];
    let buffer = Buffer::Native(NativeBuffer::from_vec(data_out));
    let output = Tensor::from_buffer(output_shape, buffer);

    output
}

pub fn col2im<T>(
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
    // (width, height)

    let batch_size = input.shape()[0];
    let num_channels = input.shape()[1];
    let input_size = (input.shape()[3], input.shape()[2]);

    let input_len = input_size.0 * input_size.1;
    let output_len = output_size.0 * output_size.1;
    let kernel_len = kernel_size.0 * kernel_size.1;

    let data_in = input.buffer.as_native().data();
    let mut data_out = Vec::with_capacity(batch_size * num_channels * output_len);

    let mut index_in_offset = input.offset();
    let mut index_out_offset = 0;

    for n in 0..batch_size {
        index_in_offset += n * num_channels * output_len * kernel_len;
        index_out_offset += n * num_channels * input_len;

        for c in 0..num_channels {
            index_in_offset += c * output_len * kernel_len;
            index_out_offset += c * input_len;

            for h in 0..output_size.1 {
                index_in_offset += h * output_size.0 * kernel_len;

                for w in 0..output_size.0 {
                    index_in_offset += w * kernel_len;

                    let input_x_offset = (w * stride.0) as isize - padding.0 as isize;
                    let input_y_offset = (h * stride.1) as isize - padding.1 as isize;

                    for kh in 0..kernel_size.1 {
                        for kw in 0..kernel_size.0 {
                            let output_x = (kw * dilation.0) as isize + input_x_offset;
                            let output_y = (kh * dilation.1) as isize + input_y_offset;

                            let index_in = kh * kernel_size.0 + kw + index_in_offset;

                            if output_x >= 0
                                && output_y >= 0
                                && output_x < input_size.0 as isize
                                && output_y < input_size.1 as isize
                            {
                                let index_out =
                                    output_y as usize * input_size.0 * output_x as usize
                                        + index_out_offset;
                                data_out[index_out] = data_in[index_in];
                            }
                        }
                    }
                }
            }
        }
    }

    let output_shape = [batch_size, num_channels, output_size.1, output_size.0];
    let buffer = Buffer::Native(NativeBuffer::from_vec(data_out));
    let output = Tensor::from_buffer(output_shape, buffer);

    output
}
