use crate::tensor::{Element, Tensor};
use num_traits::{Float, NumOps, Pow};
use std::ops::{Add, Div, Mul, Neg, Sub};

mod conv;
mod linalg;
mod math;
mod stats;

#[derive(Copy, Clone, Debug)]
pub enum BinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    Squidiff,
}

#[derive(Copy, Clone, Debug)]
pub enum BinaryLogicOperation {
    Max,
    Min,
    Eq,
    NotEq,
    GreaterThan,
    LessThan,
    GreaterOrEq,
    LessOrEq,
}

#[derive(Copy, Clone, Debug)]
pub enum BinaryIndexOperation {
    Max,
    Min,
}

#[derive(Copy, Clone, Debug)]
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

impl<T> Neg for Tensor<T>
where
    T: Element + Float,
{
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        self.unary_op(UnaryOperation::Neg)
    }
}

impl<'a, T> Neg for &'a Tensor<T>
where
    T: Element + Float,
{
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        self.unary_op(UnaryOperation::Neg)
    }
}

macro_rules! impl_tensor_binop {
    ($imp:ident, $method:ident) => {
        // A + B
        impl<T> $imp<Tensor<T>> for Tensor<T>
        where
            T: Element + NumOps,
        {
            type Output = Tensor<T>;

            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self.binary_op(&rhs, BinaryOperation::$imp)
            }
        }

        // &A + B
        impl<T> $imp<&Tensor<T>> for Tensor<T>
        where
            T: Element + NumOps,
        {
            type Output = Tensor<T>;

            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                self.binary_op(rhs, BinaryOperation::$imp)
            }
        }

        // A + &B
        impl<'a, T> $imp<Tensor<T>> for &'a Tensor<T>
        where
            T: Element + NumOps,
        {
            type Output = Tensor<T>;

            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self.binary_op(&rhs, BinaryOperation::$imp)
            }
        }

        // &A + &B
        impl<'a, 'b, T> $imp<&'a Tensor<T>> for &'b Tensor<T>
        where
            T: Element + NumOps,
        {
            type Output = Tensor<T>;

            fn $method(self, rhs: &'a Tensor<T>) -> Self::Output {
                self.binary_op(rhs, BinaryOperation::$imp)
            }
        }

        // A + 1
        impl<T> $imp<T> for Tensor<T>
        where
            T: Element + NumOps,
        {
            type Output = Tensor<T>;

            fn $method(self, rhs: T) -> Self::Output {
                let s = Tensor::from_elem([1], rhs, self.backend());
                self.binary_op(&s, BinaryOperation::$imp)
            }
        }

        // &A + 1
        impl<'a, T> $imp<T> for &'a Tensor<T>
        where
            T: Element + NumOps + Pow<T, Output = T> + PartialOrd,
        {
            type Output = Tensor<T>;

            fn $method(self, rhs: T) -> Self::Output {
                let s = Tensor::from_elem([1], rhs, self.backend());
                self.binary_op(&s, BinaryOperation::$imp)
            }
        }
    };
}

impl_tensor_binop!(Add, add);
impl_tensor_binop!(Sub, sub);
impl_tensor_binop!(Mul, mul);
impl_tensor_binop!(Div, div);
