use crate::autodiff::ops::activations::{leaky_relu, relu, sigmoid, tanh};
use crate::autodiff::var::Var;
use crate::layers::{Parameter, Stackable};

pub struct Relu;

impl Parameter for Relu {}
impl Stackable for Relu {
    fn forward(&self, x: &Var) -> Var {
        relu(x)
    }
}

pub struct LeakyRelu {
    alpha: f32,
}

impl LeakyRelu {
    pub fn new(alpha: f32) -> Self {
        LeakyRelu { alpha }
    }
}

impl Parameter for LeakyRelu {}
impl Stackable for LeakyRelu {
    fn forward(&self, x: &Var) -> Var {
        leaky_relu(x, self.alpha)
    }
}

pub struct Sigmoid;

impl Parameter for Sigmoid {}
impl Stackable for Sigmoid {
    fn forward(&self, x: &Var) -> Var {
        sigmoid(x)
    }
}

pub struct Tanh;

impl Parameter for Tanh {}
impl Stackable for Tanh {
    fn forward(&self, x: &Var) -> Var {
        tanh(x)
    }
}
