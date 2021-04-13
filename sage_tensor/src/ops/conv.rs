use crate::tensor::{Element, Tensor};
use num_traits::Float;

impl<T> Tensor<T>
where
    T: Element + Float,
{
    pub fn im2col(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Tensor<T> {
        self.backend()
            .im2col(self, kernel_size, stride, padding, dilation)
    }

    pub fn col2im(
        &self,
        output_size: (usize, usize),
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Tensor<T> {
        self.backend()
            .col2im(self, output_size, kernel_size, stride, padding, dilation)
    }
}
