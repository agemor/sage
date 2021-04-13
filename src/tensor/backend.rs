
use crate::tensor::{backend, Tensor};
use std::any::Any;
use std::rc::Rc;

pub mod native;
pub mod vulkan;

pub enum BinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
    Squidiff,
    // comparison operators
    Eq,
    NotEq,
    GreaterThan,
    LessThan,
    GreaterOrEq,
    LessOrEq,
}

pub enum BinaryIndexOperation {
    Max,
    Min,
}

pub enum UnaryOperation {
    Neg,
    Abs,
    Recip,

    // +- 1/2 exp
    Sqrt,
    Rsqrt,
    Square,

    // exponential & logarithm
    Exp,
    Expm1,
    Log,
    Log1p,

    // precision
    Ceil,
    Floor,
    Round,
    Sign,

    // trigonometric
    Sin,
    Sinh,
    Asinh,
    Asin,
    Cos,
    Cosh,
    Acosh,
    Acos,
    Tan,
    Tanh,
    Atan,
    Atanh,

    // misc.
    Sigmoid,
}

pub trait Backend {
    fn alloc_mem(&self, size: usize) -> Buffer;

    fn alloc_mem_from_iter<I>(&self, data: I) -> Buffer
    where
        I: ExactSizeIterator<Item = f32>;

    fn copy(&self, input: &Tensor, output: &mut Tensor);

    fn concat(&self, inputs: &[&Tensor], output: &mut Tensor, axis: usize);

    fn unary_op(&self, input: &Tensor, output: &mut Tensor, op: UnaryOperation);
    fn binary_op(&self, input1: &Tensor, input2: &Tensor, output: &mut Tensor, op: BinaryOperation);

    fn reduction(&self, input: &Tensor, output: &mut Tensor, op: BinaryOperation, axes: Vec<usize>);

    fn reduction_index(
        &self,
        input: &Tensor,
        output: &mut Tensor,
        op: BinaryIndexOperation,
        axis: usize,
    );

    fn contraction(
        &self,
        input1: &Tensor,
        input2: &Tensor,
        output: &mut Tensor,
        axes1: Vec<usize>,
        axes2: Vec<usize>,
    );

    fn im2col(
        &self,
        input: &Tensor,
        output: &mut Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    );

    fn col2im(
        &self,
        input: &Tensor,
        output: &mut Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    );
}

#[derive(Clone)]
pub enum Buffer {
    Native(native::Buffer),
    Vulkan(vulkan::Buffer),
}

impl Buffer {
    pub fn backend(&self) -> &dyn Backend {
        match self {
            Buffer::Native(b) => b.backend(),
            Buffer::Vulkan(b) => b.backend(),
        }
    }
    
    pub fn to_vec(&self) -> Vec<f32> {
        match self {
            Buffer::Native(b) => b.to_vec(),
            Buffer::Vulkan(b) => b.to_vec()
        }
    }

    pub fn same_kind(&self, other: &Buffer) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    pub fn as_native(&self) -> &native::Buffer {
        match self {
            Buffer::Native(b) => b,
            Buffer::Vulkan(b) => panic!("cannot cast"),
        }
    }

    pub fn as_graphics(&self) -> &vulkan::Buffer {
        match self {
            Buffer::Native(b) => panic!("cannot cast"),
            Buffer::Vulkan(b) => b,
        }
    }
}
