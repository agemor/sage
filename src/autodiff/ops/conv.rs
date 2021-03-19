use crate::autodiff::ops::{DebugInfo, Operator};
use crate::autodiff::var::Var;
use crate::profile::{torch_var, Profiler};
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

// pub struct Conv2d {
//     stride: usize,
//     padding: usize,
//     dilation: usize,
// }

pub struct Im2Col {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

pub struct Col2Im {
    img_w: usize,
    img_h: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

pub fn get_conv_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    (input_size + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1
}

pub fn get_deconv_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
}

// impl Operator<2> for Conv2d {
//     fn compute(&self, x: [&Tensor; 2]) -> Tensor {
//         let img = x[0];
//         let kernel = x[1];
//
//         unimplemented!()
//     }
//
//     fn forward(self, x: [&Var; 2]) -> Var {
//         // (N, C, H, W)
//         let img = x[0];
//
//         // (C_out, C_in, H, W)
//         let kernel = x[1];
//
//         if img.rank() != 4 || kernel.rank() != 4 {
//             panic!("invalid img or kernel rank");
//         }
//
//         let batch_size = img.shape()[0];
//         let img_c = img.shape()[1];
//         let img_h = img.shape()[2];
//         let img_w = img.shape()[3];
//
//         let kernel_c_out = kernel.shape()[0];
//         let kernel_c_in = kernel.shape()[1];
//         let kernel_h = kernel.shape()[2];
//         let kernel_w = kernel.shape()[3];
//
//         if img_c != kernel_c_in {
//             panic!("img and kernel not compatible");
//         }
//
//         let img_h_out = get_conv_size(img_h, kernel_h, self.stride, self.padding, self.dilation);
//         let img_w_out = get_conv_size(img_w, kernel_w, self.stride, self.padding, self.dilation);
//
//         let shape = [batch_size, kernel_c_out, img_h_out, img_w_out];
//
//         Var::from_binary_op(shape, self, [img, kernel])
//     }
//
//     fn backward(&self, x: [&Var; 2], gy: &Var) -> [Var; 2] {
//         // x, W, b = self.inputs
//         // # ==== gx ====
//         //     gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad,
//         //                   outsize=(x.shape[2], x.shape[3]))
//         // # ==== gW ====
//         //     gW = Conv2DGradW(self)(x, gy)
//         // # ==== gb ====
//         //     gb = None
//         // if b.data is not None:
//         //     gb = gy.sum(axis=(0, 2, 3))
//         // return gx, gW, gb
//
//         let img = x[0];
//         let kernel = x[1];
//
//         let gimg = deconv2d(gy, kernel, self.stride, self.padding, self.dilation);
//
//         //let gkernel
//         unimplemented!()
//     }
// }

impl Operator<1> for Im2Col {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        unimplemented!()
    }

    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &mut Profiler) -> DebugInfo {
        let uid = format!(
            "im2col_{}_{}_{}_{}",
            x[0].shape().to_id(),
            self.kernel_size,
            self.stride,
            self.padding
        );

        let mut comp_time = self.kernel_size * self.kernel_size * 219;

        if let Some(t) = profiler.comp_time(&uid) {
            comp_time = t;
        } else {
            let v1 = &uid;

            profiler.add_benchmark(
                &uid,
                {
                    // prep code
                    torch_var(v1, x[0].shape())
                },
                {
                    // exec code
                    format!(
                        "torch.nn.functional.unfold({}, {}, 1, {}, {})",
                        v1, self.kernel_size, self.padding, self.stride
                    )
                },
            );
        }

        DebugInfo::new("Im2col", y.shape().size(), comp_time)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        // (N, C, H, W)
        let img = x[0];

        if img.rank() != 4 {
            panic!("only tensors with rank=4 is supported");
        }

        let batch_size = img.shape()[0];
        let channels = img.shape()[1];
        let img_h = img.shape()[2];
        let img_w = img.shape()[3];

        // (N, C, KH, KW, OH, OW)

        let col_h = get_conv_size(
            img_h,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        );
        let col_w = get_conv_size(
            img_w,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        );

        let shape = [
            batch_size,
            channels,
            self.kernel_size,
            self.kernel_size,
            col_h,
            col_w,
        ];

        Var::from_unary_op(shape, self, img)
    }

    fn backward(&self, x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let img: &Var = x[0];

        let img_h = img.shape()[2];
        let img_w = img.shape()[3];

        let gx = gy.col2im(
            img_w,
            img_h,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        );
        [gx]
    }
}

impl Operator<1> for Col2Im {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        unimplemented!()
    }

    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &mut Profiler) -> DebugInfo {
        let uid = format!(
            "col2im_{}_{}_{}_{}",
            x[0].shape().to_id(),
            self.kernel_size,
            self.stride,
            self.padding
        );

        let mut comp_time = self.kernel_size * self.kernel_size * 219;

        if let Some(t) = profiler.comp_time(&uid) {
            comp_time = t;
        } else {
            let v1 = &uid;

            profiler.add_benchmark(
                &uid,
                {
                    // prep code
                    let s = x[0].shape();

                    torch_var(v1, [s[0], s[1] * s[2] * s[3], s[4], s[5]])
                },
                {
                    // exec code
                    format!(
                        "torch.nn.functional.unfold({}, {}, 1, {}, {})",
                        v1, self.kernel_size, self.padding, self.stride
                    )
                },
            );
        }

        DebugInfo::new("Col2im", y.shape().size(), comp_time)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        // (N, C, KH, KW, OH, OW)
        let img = x[0];

        if img.rank() != 6 {
            println!("img {}", img.shape());
            panic!("only tensors with rank=6 is supported");
        }

        let batch_size = img.shape()[0];
        let channels = img.shape()[1];
        let col_h = img.shape()[4];
        let col_w = img.shape()[5];

        // (N, C, H, W)

        let img_h = get_deconv_size(
            col_h,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        );
        let img_w = get_deconv_size(
            col_w,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        );

        let shape = [batch_size, channels, self.img_h, self.img_w];

        Var::from_unary_op(shape, self, img)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = gy.im2col(self.kernel_size, self.stride, self.padding, self.dilation);
        [gx]
    }
}

impl Var {
    pub fn im2col(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Var {
        Im2Col {
            kernel_size,
            stride,
            padding,
            dilation,
        }
        .forward([self])
    }

    pub fn col2im(
        &self,
        img_w: usize,
        img_h: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Var {
        Col2Im {
            img_w,
            img_h,
            kernel_size,
            stride,
            padding,
            dilation,
        }
        .forward([self])
    }
}
