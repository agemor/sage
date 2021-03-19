use crate::autodiff::var::Var;
use crate::layers::activations::Relu;
use crate::layers::base::{Dense, Sequential};
use crate::layers::recurrent::Lstm;
use crate::layers::{gather_params, Parameter, Stackable};

pub struct StackedLstmConfig {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_classes: usize,
}

impl StackedLstmConfig {
    pub fn d5() -> Self {
        StackedLstmConfig {
            input_size: 256,
            hidden_size: 256,
            num_layers: 5,
            num_classes: 10,
        }
    }
}

pub struct StackedLstm {
    pass: Sequential,
    classifier: Dense,
}

impl StackedLstm {
    pub fn new(config: StackedLstmConfig) -> Self {
        let mut pass = Sequential::new();

        pass.extend(vec![
            box Lstm::new(config.input_size, config.hidden_size),
            box Relu,
        ]);

        for _ in 1..config.num_layers {
            pass.extend(vec![
                box Lstm::new(config.hidden_size, config.hidden_size),
                box Relu,
            ]);
        }

        let classifier = Dense::new(config.hidden_size, config.num_classes);

        StackedLstm { pass, classifier }
    }

    pub fn forward(&self, x: &Var) -> Var {
        let y = self.pass.forward(x);
        let y_last = y.index(-1, 1).squeeze(1);

        let logits = self.classifier.forward(&y_last);
        logits
    }
}

impl Parameter for StackedLstm {
    fn init(&self) {
        self.pass.init();
        self.classifier.init();
    }
    fn params(&self) -> Option<Vec<&Var>> {
        gather_params(vec![self.pass.params(), self.classifier.params()])
    }
}
