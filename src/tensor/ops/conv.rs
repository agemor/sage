use crate::tensor::shape::ToShape;
use crate::tensor::Tensor;

impl Tensor {
    pub fn im2col(&self, kernel_size: usize, stride: usize, padding: usize) -> Tensor {
        // N, C, H, W = img.shape
        // KH, KW = pair(kernel_size)
        // SH, SW = pair(stride)
        // PH, PW = pair(pad)
        // OH = get_conv_outsize(H, KH, SH, PH)
        // OW = get_conv_outsize(W, KW, SW, PW)
        //
        // xp = cuda.get_array_module(img)
        // if xp != np:
        //     col = _im2col_gpu(img, kernel_size, stride, pad)
        // else:
        // img = np.pad(img,
        //              ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
        //              mode='constant', constant_values=(0,))
        // col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)
        //
        // for j in range(KH):
        //     j_lim = j + SH * OH
        // for i in range(KW):
        //     i_lim = i + SW * OW
        // col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]
        //
        // if to_matrix:
        //     col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))
        //
        Tensor::null()
    }

    pub fn col2im<S>(
        &self,
        img_shape: S,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Tensor
    where
        S: ToShape,
    {
        // N, C, H, W = img_shape
        // KH, KW = pair(kernel_size)
        // SH, SW = pair(stride)
        // PH, PW = pair(pad)
        // OH = get_conv_outsize(H, KH, SH, PH)
        // OW = get_conv_outsize(W, KW, SW, PW)
        //
        // if to_matrix:
        //     col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
        //
        // xp = cuda.get_array_module(col)
        // if xp != np:
        //     img = _col2im_gpu(col, SH, SW, PH, PW, H, W)
        // return img
        // else:
        // img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
        //                dtype=col.dtype)
        // for j in range(KH):
        //     j_lim = j + SH * OH
        // for i in range(KW):
        //     i_lim = i + SW * OW
        // img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
        // return img[:, :, PH:H + PH, PW:W + PW]

        Tensor::null()
    }
}

#[cfg(test)]
mod test {
    use super::*;
}
