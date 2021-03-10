use crate::autodiff::var::Var;
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
        unimplemented!()
    }
}

pub struct BatchNorm2d {
    num_features: usize,
    eps: f32,
}

impl BatchNorm2d {
    pub fn new(num_features: usize, eps: f32) -> Self {
        BatchNorm2d { num_features, eps }
    }
}

impl Parameter for BatchNorm2d {}

impl Stackable for BatchNorm2d {
    fn forward(&self, x: &Var) -> Var {
        unimplemented!()
    }
}
