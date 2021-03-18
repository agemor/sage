use crate::autodiff::var::Var;
use crate::layers::activations::Relu;
use crate::layers::base::{Dense, Dropout, Sequential};
use crate::layers::convolutional::Conv2d;
use crate::layers::normalization::BatchNorm2d;
use crate::layers::pooling::AvgPool2d;
use crate::layers::{gather_params, Parameter, Stackable};

#[derive(Copy, Clone)]
pub struct DenseNetConfig {
    depth: usize,
    growth_rate: usize,

    batch_norm_eps: f32,
    dropout_prob: f32,

    num_classes: usize,
}

impl DenseNetConfig {
    pub fn d121() -> Self {
        DenseNetConfig {
            depth: 121,
            growth_rate: 12,
            batch_norm_eps: 0.0001,
            dropout_prob: 0.2,
            num_classes: 10,
        }
    }

    pub fn d169() -> Self {
        DenseNetConfig {
            depth: 169,
            growth_rate: 24,
            batch_norm_eps: 0.0001,
            dropout_prob: 0.2,
            num_classes: 10,
        }
    }

    pub fn d201() -> Self {
        DenseNetConfig {
            depth: 201,
            growth_rate: 12,
            batch_norm_eps: 0.0001,
            dropout_prob: 0.2,
            num_classes: 10,
        }
    }
}

pub struct DenseNet {
    model: Sequential,
    classifier: Dense,
}

impl DenseNet {
    pub fn new(config: DenseNetConfig) -> Self {
        let mut model = Sequential::new();
        let n = (config.depth - 4) / 6;

        let mut in_planes = 2 * config.growth_rate;

        model.add(box Conv2d::new(3, in_planes, 3, 1, 1));

        for i in 0..3 {
            model.add(box Self::dense_layer(n, in_planes, config));

            in_planes += n * config.growth_rate;

            if i < 2 {
                model.add(box Self::transition_layer(in_planes, in_planes / 2, config));
                in_planes /= 2;
            }
        }

        model.add(box BatchNorm2d::new(in_planes, config.batch_norm_eps));
        model.add(box Relu);
        // model.add(box AvgPool2d::new(8));

        DenseNet {
            model,
            classifier: Dense::new(in_planes, config.num_classes),
        }
    }

    fn transition_layer(in_planes: usize, out_planes: usize, config: DenseNetConfig) -> Sequential {
        Sequential::from(vec![
            box BatchNorm2d::new(in_planes, config.batch_norm_eps),
            box Relu,
            box Conv2d::new(in_planes, out_planes, 1, 1, 0),
            box Dropout::new(config.dropout_prob),
            box AvgPool2d::new(2),
        ])
    }

    fn dense_layer(num_layers: usize, in_planes: usize, config: DenseNetConfig) -> Sequential {
        Sequential::from(
            (0..num_layers)
                .into_iter()
                .map(|i| {
                    box BottleneckLayer::new(
                        in_planes + i * config.growth_rate,
                        config.growth_rate,
                        config,
                    ) as Box<dyn Stackable>
                })
                .collect(),
        )
    }

    pub fn forward(&self, x: &Var) -> Var {
        let y = self.model.forward(x);
        let pooled = AvgPool2d::new(y.shape()[2]).forward(&y);

        let y = pooled.reshape([y.shape()[0], 0]);

        let logits = self.classifier.forward(&y);

        logits
    }
}

impl Parameter for DenseNet {
    fn init(&self) {
        self.model.init();
        self.classifier.init();
    }

    fn params(&self) -> Option<Vec<&Var>> {
        gather_params(vec![self.model.params(), self.classifier.params()])
    }
}

struct BottleneckLayer {
    pass: Sequential,
}

impl BottleneckLayer {
    pub fn new(in_planes: usize, out_planes: usize, config: DenseNetConfig) -> Self {
        let inter_planes = out_planes * 4;
        BottleneckLayer {
            pass: Sequential::from(vec![
                box BatchNorm2d::new(in_planes, config.batch_norm_eps),
                box Relu,
                box Conv2d::new(in_planes, inter_planes, 1, 1, 0),
                box BatchNorm2d::new(inter_planes, config.batch_norm_eps),
                box Conv2d::new(inter_planes, out_planes, 3, 1, 1),
                box Dropout::new(config.dropout_prob),
            ]),
        }
    }
}

impl Parameter for BottleneckLayer {
    fn init(&self) {
        self.pass.init()
    }

    fn params(&self) -> Option<Vec<&Var>> {
        self.pass.params()
    }
}

impl Stackable for BottleneckLayer {
    fn forward(&self, x: &Var) -> Var {
        let y = self.pass.forward(x);
        // println!(
        //     "x: {}    y: {}     c:{}",
        //     x.shape(),
        //     y.shape(),
        //     x.concat(&y, 1).shape()
        // );

        // key idea of the DenseNet
        x.concat(y, 1)
    }
}
