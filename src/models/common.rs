use crate::autodiff::Var;
use crate::layers::Layer;
use crate::shape::{Shape, ToShape};

pub struct Dropout {
    prob: f32,
}

impl Dropout {
    pub fn new(prob: f32) -> Self {
        Dropout { prob }
    }
}

impl Layer for Dropout {
    fn init(&self) {
        unimplemented!()
    }

    fn forward(&self, x: &Var) -> Var {
        unimplemented!()
    }

    fn params(&self) -> Option<Vec<&Var>> {
        unimplemented!()
    }
}

pub struct Softmax {
    axis: isize,
}

impl Softmax {
    pub fn new(axis: isize) -> Self {
        Softmax { axis }
    }
}

impl Layer for Softmax {
    fn init(&self) {
        unimplemented!()
    }

    fn forward(&self, x: &Var) -> Var {
        unimplemented!()
    }

    fn params(&self) -> Option<Vec<&Var>> {
        unimplemented!()
    }
}

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

impl Layer for LayerNorm {
    fn init(&self) {
        unimplemented!()
    }

    fn forward(&self, x: &Var) -> Var {
        unimplemented!()
    }

    fn params(&self) -> Option<Vec<&Var>> {
        unimplemented!()
    }
}

pub struct Embedding {
    num_embeddings: usize,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Embedding {
            num_embeddings,
            embedding_dim,
        }
    }

    pub fn forward_with(&self, ids: &[usize]) -> Var {
        unimplemented!();
    }
}

impl Layer for Embedding {
    fn init(&self) {
        unimplemented!()
    }

    fn forward(&self, x: &Var) -> Var {
        unimplemented!()
    }

    fn params(&self) -> Option<Vec<&Var>> {
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

impl Layer for BatchNorm2d {
    fn init(&self) {
        unimplemented!()
    }

    fn forward(&self, x: &Var) -> Var {
        unimplemented!()
    }

    fn params(&self) -> Option<Vec<&Var>> {
        unimplemented!()
    }
}

pub struct Conv2d {
    num_in_chan: usize,
    num_out_chan: usize,
    kernel_size: usize,
    stride: usize,
}

impl Conv2d {
    pub fn new(num_in_chan: usize, num_out_chan: usize, kernel_size: usize, stride: usize) -> Self {
        Conv2d {
            num_in_chan,
            num_out_chan,
            kernel_size,
            stride,
        }
    }
}

impl Layer for Conv2d {
    fn init(&self) {
        unimplemented!()
    }

    fn forward(&self, x: &Var) -> Var {
        unimplemented!()
    }

    fn params(&self) -> Option<Vec<&Var>> {
        unimplemented!()
    }
}

pub struct AvgPool2d {
    kernel_size: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize) -> Self {
        AvgPool2d { kernel_size }
    }
}

impl Layer for AvgPool2d {
    fn init(&self) {
        unimplemented!()
    }

    fn forward(&self, x: &Var) -> Var {
        unimplemented!()
    }

    fn params(&self) -> Option<Vec<&Var>> {
        unimplemented!()
    }
}
