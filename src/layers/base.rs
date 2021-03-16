use crate::autodiff::var::Var;
use crate::layers::{Parameter, Stackable};
use crate::tensor;
use crate::tensor::Tensor;

pub struct Sequential {
    pub layers: Vec<Box<dyn Stackable>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    pub fn from(layers: Vec<Box<dyn Stackable>>) -> Self {
        Sequential { layers }
    }

    pub fn add(&mut self, layer: Box<dyn Stackable>) {
        self.layers.push(layer);
    }

    pub fn extend(&mut self, layers: Vec<Box<dyn Stackable>>) {
        self.layers.extend(layers);
    }
}

impl Parameter for Sequential {
    fn init(&self) {
        for layer in self.layers.iter() {
            layer.init();
        }
    }

    fn params(&self) -> Option<Vec<&Var>> {
        let mut params = Vec::new();

        for layer in self.layers.iter() {
            if let Some(p) = layer.params() {
                params.extend(p)
            }
        }
        Some(params)
    }
}

impl Stackable for Sequential {
    fn forward(&self, x: &Var) -> Var {
        self.layers.iter().fold(x.clone(), |x, layer| {
            let y = layer.forward(&x);
            y
        })
    }
}

pub struct Dense {
    pub kernel: Var,
    pub bias: Var,
}

impl Dense {
    pub fn new(input: usize, output: usize) -> Self {
        Dense {
            kernel: Var::with_shape([output, input]),
            bias: Var::with_shape([output]),
        }
    }
}

impl Parameter for Dense {
    fn init(&self) {
        // do some Kaiming init (targeted for the ReLU)
        //self.kernel
        //    .set_data(tensor::init::kaiming_uniform(self.kernel.shape(), 1.0));

        self.kernel.set_data(Tensor::null());

        self.bias.set_data(Tensor::zeros(self.bias.shape()));
    }
    fn params(&self) -> Option<Vec<&Var>> {
        Some(vec![&self.kernel, &self.bias])
    }
}

impl Stackable for Dense {
    fn forward(&self, x: &Var) -> Var {
        self.kernel.matvec(x) + &self.bias
    }
}

pub struct Dropout {
    prob: f32,
}

impl Dropout {
    pub fn new(prob: f32) -> Self {
        Dropout { prob }
    }
}

impl Parameter for Dropout {}

impl Stackable for Dropout {
    fn forward(&self, x: &Var) -> Var {
        x.clone()
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

impl Parameter for Softmax {}

impl Stackable for Softmax {
    fn forward(&self, x: &Var) -> Var {
        x.softmax(self.axis)
    }
}
