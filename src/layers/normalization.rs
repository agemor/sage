use crate::autodiff::var::Var;
use crate::layers::base::Dense;
use crate::layers::{Parameter, Stackable};
use crate::tensor::shape::{Shape, ToShape};

pub struct LayerNorm {
    shape: Shape,
    eps: f32,
}

impl LayerNorm {
    pub fn new<S>(shape: S, eps: f32) -> Self
    where
        S: ToShape,
    {
        LayerNorm {
            shape: shape.to_shape(),
            eps,
        }
    }
}

impl Parameter for LayerNorm {}

impl Stackable for LayerNorm {
    fn forward(&self, x: &Var) -> Var {
        x.clone()
    }
}

pub struct BatchNorm2d {
    num_features: usize,
    eps: f32,

    gamma: Var,
    beta: Var,

    // average batch mean
    running_mean: Var,

    // average batch variance
    running_var: Var,
}

impl BatchNorm2d {
    pub fn new(num_features: usize, eps: f32) -> Self {
        BatchNorm2d {
            num_features,
            eps,
            gamma: (),
            beta: (),
            running_mean: (),
            running_var: (),
        }
    }
}

impl Parameter for BatchNorm2d {}

impl Stackable for BatchNorm2d {
    fn forward(&self, x: &Var) -> Var {

        // (N, C, H, W) -> (N*H*W, C)
        let x = x.permute([0, 2, 3, 1]).reshape([-1, self.num_features]);
        //
        // if dezero.Config.train:
        //     mean = x.mean(axis=0)
        // var = x.var(axis=0)
        // inv_std = 1 / xp.sqrt(var + self.eps)
        // xc = (x - mean) * inv_std
        //
        // m = x.size // gamma.size
        // s = m - 1. if m - 1. > 1. else 1.
        // adjust = m / s  # unbiased estimation
        // self.avg_mean *= self.decay
        // self.avg_mean += (1 - self.decay) * mean
        // self.avg_var *= self.decay
        // self.avg_var += (1 - self.decay) * adjust * var
        // self.inv_std = inv_std
        // else:
        // inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
        // xc = (x - self.avg_mean) * inv_std
        // y = gamma * xc + beta
        //
        // if x_ndim == 4:
        // # (N*H*W, C) -> (N, C, H, W)
        // y = y.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        // return y


        let mean =x.mean(0,true);
        let var = x

        x.clone()
    }
}
