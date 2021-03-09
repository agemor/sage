use crate::autodiff::Var;
use crate::layers::activations::Relu;
use crate::layers::{gather_params, Affine, Layer, Sequential};
use crate::models::common::{AvgPool2d, BatchNorm2d, Conv2d, Dropout};
use crate::ops::concat;

struct BottleneckLayer {
    pass: Sequential,
}

impl BottleneckLayer {
    pub fn new(
        in_planes: usize,
        out_planes: usize,
        dropout_prob: f32,
        batch_norm_eps: f32,
    ) -> Self {
        let inter_planes = out_planes * 4;
        BottleneckLayer {
            pass: Sequential::from(vec![
                box BatchNorm2d::new(in_planes, batch_norm_eps),
                box Relu,
                box Conv2d::new(in_planes, inter_planes, 1, 1),
                box BatchNorm2d::new(inter_planes, batch_norm_eps),
                box Conv2d::new(inter_planes, out_planes, 3, 1),
                box Dropout::new(dropout_prob),
            ]),
        }
    }
}

impl Layer for BottleneckLayer {
    fn init(&self) {
        self.pass.init()
    }

    fn forward(&self, x: &Var) -> Var {
        let y = self.pass.forward(x);

        // key component of the DenseNet
        concat(&x, &y, 1)
    }

    fn params(&self) -> Option<Vec<&Var>> {
        self.pass.params()
    }
}

struct TransitionLayer {
    pass: Sequential,
}

impl TransitionLayer {
    pub fn new(
        in_planes: usize,
        out_planes: usize,
        dropout_prob: f32,
        batch_norm_eps: f32,
    ) -> Self {
        TransitionLayer {
            pass: Sequential::from(vec![
                box BatchNorm2d::new(in_planes, batch_norm_eps),
                box Relu,
                box Conv2d::new(in_planes, out_planes, 1, 1),
                box Dropout::new(dropout_prob),
                box AvgPool2d::new(2),
            ]),
        }
    }
}

impl Layer for TransitionLayer {
    fn init(&self) {
        self.pass.init();
    }

    fn forward(&self, x: &Var) -> Var {
        self.pass.forward(x)
    }

    fn params(&self) -> Option<Vec<&Var>> {
        self.pass.params()
    }
}

struct DenseLayer {
    pass: Sequential,
}

impl DenseLayer {
    pub fn new(num_layers: usize, in_planes: usie, growth_rate: usize, dropout_prob: f32) -> Self {
        DenseLayer {
            pass: Sequential::from(
                (0..num_layers)
                    .into_iter()
                    .map(|i| {
                        box BottleneckLayer::new(
                            in_planes * i * growth_rate,
                            growth_rate,
                            dropout_prob,
                            0.001,
                        )
                    })
                    .collect(),
            ),
        }
    }
}

impl Layer for DenseLayer {
    fn init(&self) {
        self.pass.init();
    }

    fn forward(&self, x: &Var) -> Var {
        self.pass.forward(x)
    }

    fn params(&self) -> Option<Vec<&Var>> {
        self.pass.params()
    }
}

pub struct DenseNet {
    pass: Sequential,
    classifier: Affine,
}

impl DenseNet {
    pub fn new(
        depth: usize,
        growth_rate: usize,
        num_classes: usize,
        dropout_prob: f32,
        batch_norm_eps: f32,
    ) -> Self {
        let mut pass = Sequential::new();
        let n = (depth - 4) / 3;

        let mut in_planes = 2 * growth_rate;

        pass.add(box Conv2d::new(3, in_planes, 3, 1));

        for i in 0..n {
            pass.add(box DenseLayer::new(n, in_planes, growth_rate, dropout_prob));

            in_planes = in_planes + n * growth_rate;

            if i < n - 1 {
                pass.add(box TransitionLayer::new(
                    in_planes,
                    in_planes / 2,
                    dropout_prob,
                    batch_norm_eps,
                ));
                in_planes = in_planes / 2;
            }
        }

        pass.add(box BatchNorm2d::new(in_planes, batch_norm_eps));
        pass.add(box Relu);
        pass.add(box AvgPool2d::new(8));

        DenseNet {
            pass,
            classifier: Affine::new(in_planes, num_classes),
        }
    }

    pub fn d121() -> Self {
        DenseNet::new(121, 12, 10, 0.2, 0.0001)
    }

    pub fn d169() -> Self {
        DenseNet::new(169, 12, 10, 0.2, 0.0001)
    }

    pub fn d201() -> Self {
        DenseNet::new(201, 12, 10, 0.2, 0.0001)
    }

    pub fn d264() -> Self {
        DenseNet::new(264, 12, 10, 0.2, 0.0001)
    }
}

impl Layer for DenseNet {
    fn init(&self) {
        self.pass.init();
        self.classifier.init();
    }

    fn forward(&self, x: &Var) -> Var {
        let y = self.pass.forward(x);
        let batch_size = y.shape()[0];
        let logit_size = y.shape().size() / batch_size;
        let y = y.reshape([batch_size, logit_size]);
        self.classifier.forward(&y)
    }

    fn params(&self) -> Option<Vec<&Var>> {
        gather_params(vec![self.pass.params(), self.classifier.params()])
    }
}
