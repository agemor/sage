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

pub struct Embedding {}

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
