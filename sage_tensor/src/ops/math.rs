use crate::ops::UnaryOperation;
use crate::shape::Axis;
use crate::tensor::{Element, Tensor};
use num_traits::Float;

impl<T> Tensor<T>
where
    T: Element + Float,
{
    pub fn softmax<A>(&self, axis: A) -> Tensor<T>
    where
        A: Axis,
    {
        let axis = axis.to_usize(self.order()).unwrap();

        let max = self.max([axis], true);

        // for numerical stability
        let y = (self - max).exp();

        let sum = y.sum([axis], true);
        y / sum
    }

    pub fn log_sum_exp<A>(&self, axis: A, retain_axis: bool) -> Tensor<T>
    where
        A: Axis,
    {
        let axis = axis.to_usize(self.order()).unwrap();

        // (N, K) -> (N, 1)

        let max = self.max([axis], true);

        // (N, K) - (N, 1) -> (N, K)
        let y = (self - &max).exp();

        // (N, K) -> (N, '1)
        let sum = y.sum([axis], retain_axis).log();

        // (N, '1) + (N, 1)
        if retain_axis {
            sum + max
        } else {
            sum + max.squeeze_axis(axis)
        }
    }

    pub fn abs(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Abs)
    }

    pub fn recip(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Recip)
    }

    pub fn sqrt(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Sqrt)
    }

    pub fn rsqrt(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Rsqrt)
    }

    pub fn square(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Square)
    }

    pub fn exp(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Exp)
    }

    pub fn expm1(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Expm1)
    }

    pub fn log(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Log)
    }

    pub fn log1p(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Log1p)
    }

    pub fn ceil(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Ceil)
    }

    pub fn floor(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Floor)
    }

    pub fn round(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Round)
    }

    pub fn sign(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Sign)
    }

    pub fn sin(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Sin)
    }

    pub fn sinh(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Sinh)
    }

    pub fn asinh(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Asinh)
    }

    pub fn asin(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Asin)
    }

    pub fn cos(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Cos)
    }

    pub fn cosh(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Cosh)
    }

    pub fn acosh(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Acosh)
    }

    pub fn acos(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Acos)
    }

    pub fn tan(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Tan)
    }

    pub fn tanh(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Tanh)
    }

    pub fn atan(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Atan)
    }

    pub fn atanh(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Atanh)
    }

    pub fn sigmoid(&self) -> Tensor<T> {
        self.unary_op(UnaryOperation::Sigmoid)
    }
}
