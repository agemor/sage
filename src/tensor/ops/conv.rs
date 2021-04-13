use crate::tensor::shape::ToShape;
use crate::tensor::Tensor;

impl Tensor {
    pub fn im2col(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Tensor {
        let batch_size = self.shape[0];
        let num_channels = self.shape[1];
        let input_size = (self.shape[3], self.shape[2]);

        let output_size = (
            (input_size.0 + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1) / stride.0 + 1,
            (input_size.1 + 2 * padding.1 - dilation.1 * (kernel_size.1 - 1) - 1) / stride.1 + 1,
        );

        let mut output = Tensor::uninit(
            [batch_size, num_channels, output_size.1, output_size.0],
            self.backend().clone(),
        );

        self.backend().processor().im2col(
            self,
            &mut output,
            kernel_size,
            stride,
            padding,
            dilation,
        );

        output
    }

    pub fn col2im(
        &self,
        output_size: (usize, usize),
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Tensor {
        let batch_size = self.shape[0];
        let num_channels = self.shape[1];

        let mut output = Tensor::uninit(
            [batch_size, num_channels, output_size.1, output_size.0],
            self.backend().clone(),
        );

        self.backend().processor().col2im(
            self,
            &mut output,
            kernel_size,
            stride,
            padding,
            dilation,
        );

        output
    }
}

#[cfg(test)]
mod test {
    use super::*;
}
