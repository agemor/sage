use crate::autodiff::var::Var;
use crate::layers::activations::Relu;
use crate::layers::base::{Dense, Sequential};
use crate::layers::convolutional::Conv2d;
use crate::layers::normalization::BatchNorm2d;
use crate::layers::pooling::AvgPool2d;
use crate::layers::{gather_params, Parameter, Stackable};

#[derive(Clone, Copy)]
pub struct ResNetConfig {
    eps: f32,
    num_classes: usize,
    num_blocks: [usize; 4],
    expansion: usize,
}

impl ResNetConfig {
    pub fn d50() -> Self {
        ResNetConfig {
            eps: 0.0001,
            num_classes: 10,
            num_blocks: [3, 4, 6, 3],
            expansion: 4,
        }
    }

    pub fn d101() -> Self {
        ResNetConfig {
            eps: 0.0001,
            num_classes: 10,
            num_blocks: [3, 4, 23, 3],
            expansion: 4,
        }
    }

    pub fn d152() -> Self {
        ResNetConfig {
            eps: 0.0001,
            num_classes: 10,
            num_blocks: [3, 8, 36, 3],
            expansion: 4,
        }
    }
}

pub struct ResNet {
    pass: Sequential,
    classifier: Dense,
}

impl ResNet {
    pub fn new(config: ResNetConfig) -> Self {
        let mut pass = Sequential::new();

        let mut in_planes = 64;
        let mut out_planes = in_planes;

        pass.extend(vec![
            box Conv2d::new(3, in_planes, 3, 1, 1),
            box BatchNorm2d::new(in_planes, config.eps),
            box Relu,
        ]);

        for i in 0..4 {
            let stride = if i == 0 { 1 } else { 2 };
            pass.add(box Self::residual_layers(
                config.num_blocks[i],
                in_planes,
                out_planes,
                stride,
                config,
            ));
            in_planes = out_planes * config.expansion;
            out_planes *= 2;
        }

        //pass.add(box AvgPool2d::new(8));

        let classifier = Dense::new(512 * config.expansion, config.num_classes);

        ResNet { pass, classifier }
    }

    fn residual_layers(
        num_blocks: usize,
        in_planes: usize,
        out_planes: usize,
        stride: usize,
        config: ResNetConfig,
    ) -> Sequential {
        let mut layers = Sequential::new();

        let mut in_planes = in_planes;

        for i in 0..num_blocks {
            let stride = if i == 0 { stride } else { 1 };
            layers.add(box BottleneckLayer::new(
                in_planes, out_planes, stride, config,
            ));
            in_planes = out_planes * config.expansion;
        }

        layers
    }
}

impl Parameter for ResNet {
    fn init(&self) {
        self.pass.init();
        self.classifier.init();
    }
    fn params(&self) -> Option<Vec<&Var>> {
        gather_params(vec![self.pass.params(), self.classifier.params()])
    }
}

impl Stackable for ResNet {
    fn forward(&self, x: &Var) -> Var {
        let y = self.pass.forward(x);

        let pooled = AvgPool2d::new(y.shape()[2]).forward(&y);

        let y = pooled.reshape([y.shape()[0], 0]);

        let logits = self.classifier.forward(&y);

        logits
    }
}

struct BottleneckLayer {
    pass: Sequential,
    shortcut: Sequential,
}

impl BottleneckLayer {
    fn new(in_planes: usize, planes: usize, stride: usize, config: ResNetConfig) -> Self {
        let mut pass = Sequential::new();
        let mut shortcut = Sequential::new();

        let out_planes = config.expansion * planes;

        pass.extend(vec![
            box Conv2d::new(in_planes, planes, 1, 1, 0),
            box BatchNorm2d::new(planes, config.eps),
            box Relu,
            box Conv2d::new(planes, planes, 3, stride, 1),
            box BatchNorm2d::new(planes, config.eps),
            box Relu,
            box Conv2d::new(planes, out_planes, 1, 1, 0),
            box BatchNorm2d::new(out_planes, config.eps),
        ]);

        if stride != 1 || in_planes != out_planes {
            shortcut.extend(vec![
                box Conv2d::new(in_planes, out_planes, 1, stride, 0),
                box BatchNorm2d::new(out_planes, config.eps),
            ]);
        }

        BottleneckLayer { pass, shortcut }
    }
}

impl Parameter for BottleneckLayer {
    fn init(&self) {
        self.pass.init();
        self.shortcut.init();
    }
    fn params(&self) -> Option<Vec<&Var>> {
        gather_params(vec![self.pass.params(), self.shortcut.params()])
    }
}

impl Stackable for BottleneckLayer {
    fn forward(&self, x: &Var) -> Var {
        let y_long = self.pass.forward(x);
        let y_short = self.shortcut.forward(x);

        let y = y_long + y_short;
        Relu.forward(&y)
    }
}
