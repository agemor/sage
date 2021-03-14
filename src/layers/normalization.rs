use crate::autodiff::var::{ToVar, Var};
use crate::layers::base::Dense;
use crate::layers::{Parameter, Stackable};
use crate::tensor::shape::{Shape, ToShape};
use crate::tensor::Tensor;

pub struct LayerNorm {
    shape: Shape,
    eps: f32,
    gamma: Var,
    beta: Var,
}

impl LayerNorm {
    pub fn new<S>(shape: S, eps: f32) -> Self
    where
        S: ToShape,
    {
        LayerNorm {
            shape: shape.to_shape(0),
            eps,
            gamma: Var::with_shape([1]),
            beta: Var::with_shape([1]),
        }
    }
}

impl Parameter for LayerNorm {
    fn init(&self) {
        self.gamma.set_data(Tensor::scalar(1.0));
        self.beta.set_data(Tensor::scalar(0.0));
    }

    fn params(&self) -> Option<Vec<&Var>> {
        Some(vec![&self.gamma, &self.beta])
    }
}

impl Stackable for LayerNorm {
    fn forward(&self, x: &Var) -> Var {
        let input_shape = x.shape();

        // (N, A, S1, S2, ..., Sn) -> (N * A, S)
        let x = x.reshape([0, self.shape.size()]);
        // TODO: update running mean and variance
        let mean = x.mean(1, true);
        let var = x.var(1, true);
        let xc = (x - mean) / var.scalar_add(self.eps).sqrt();
        let y = &self.gamma * xc + &self.beta;
        // (N * A, S) -> (N, A, S1, S2, ..., Sn)
        y.reshape(input_shape)
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
            gamma: Var::with_shape([1]),
            beta: Var::with_shape([1]),
            running_mean: Var::with_shape([num_features]),
            running_var: Var::with_shape([num_features]),
        }
    }
}

impl Parameter for BatchNorm2d {
    fn init(&self) {
        self.gamma.set_data(Tensor::scalar(1.0));
        self.beta.set_data(Tensor::scalar(0.0));
    }

    fn params(&self) -> Option<Vec<&Var>> {
        Some(vec![&self.gamma, &self.beta])
    }
}

impl Stackable for BatchNorm2d {
    fn forward(&self, x: &Var) -> Var {
        if x.rank() != 4 {
            panic!("only supports rank=4");
        }

        let batch_size = x.shape()[0];
        let channels: usize = x.shape()[1];
        let img_h = x.shape()[2];
        let img_w = x.shape()[3];

        // (N, C, H, W) -> (N*H*W, C)
        let x = x.permute([0, 2, 3, 1]).reshape([0, channels]);

        let mean = x.mean(0, true);
        let var = x.var(0, true);

        // TODO: update running mean and variance

        let xc = (x - mean) / var.scalar_add(self.eps).sqrt();

        let y = &self.gamma * xc + &self.beta;
        // (N*H*W, C) -> (N, C, H, W)
        y.reshape([batch_size, img_h, img_w, 0])
            .permute([0, 3, 1, 2])
    }
}
