use crate::autodiff::var::Var;
use crate::layers::activations::{LeakyRelu, Relu, Sigmoid, Tanh};
use crate::layers::base::Sequential;
use crate::layers::convolutional::{Conv2d, TransposedConv2d};
use crate::layers::normalization::BatchNorm2d;
use crate::layers::{gather_params, Parameter, Stackable};

#[derive(Clone, Copy)]
pub struct DcGanConfig {
    nz: usize,
    ngf: usize,
    nc: usize,
    ndf: usize,
    eps: f32,
}

impl DcGanConfig {
    pub fn default() -> Self {
        DcGanConfig {
            nz: 100,
            ngf: 128,
            nc: 3,
            ndf: 128,
            eps: 0.0001,
        }
    }
}

pub struct DcGan {
    generator: Sequential,
    discriminator: Sequential,
}

impl DcGan {
    pub fn new(config: DcGanConfig) -> Self {
        let generator = Self::generator(config);
        let discriminator = Self::discriminator(config);

        DcGan {
            generator,
            discriminator,
        }
    }

    pub fn forward(&self, x: &Var) -> Var {
        let fake = self.generator.forward(x);

        // omitted generator loss
        let det = self.discriminator.forward(&fake);

        det.reshape([0, 1])
    }

    fn generator(config: DcGanConfig) -> Sequential {
        // nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
        // nn.BatchNorm2d(ngf * 8),
        // nn.ReLU(True),
        // # state size. (ngf*8) x 4 x 4
        // nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        // nn.BatchNorm2d(ngf * 4),
        // nn.ReLU(True),
        // # state size. (ngf*4) x 8 x 8
        // nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        // nn.BatchNorm2d(ngf * 2),
        // nn.ReLU(True),
        // # state size. (ngf*2) x 16 x 16
        // nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
        // nn.BatchNorm2d(ngf),
        // nn.ReLU(True),
        // # state size. (ngf) x 32 x 32
        // nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
        // nn.Tanh()

        Sequential::from(vec![
            box TransposedConv2d::new(config.nz, config.ngf * 8, 4, 1, 0),
            box BatchNorm2d::new(config.ngf * 8, config.eps),
            box Relu,
            box TransposedConv2d::new(config.ngf * 8, config.ngf * 4, 4, 2, 1),
            box BatchNorm2d::new(config.ngf * 4, config.eps),
            box Relu,
            box TransposedConv2d::new(config.ngf * 4, config.ngf * 2, 4, 2, 1),
            box BatchNorm2d::new(config.ngf * 2, config.eps),
            box Relu,
            box TransposedConv2d::new(config.ngf * 2, config.ngf, 4, 2, 1),
            box BatchNorm2d::new(config.ngf, config.eps),
            box Relu,
            box TransposedConv2d::new(config.ngf, config.nc, 4, 2, 1),
            box Tanh,
        ])
    }

    fn discriminator(config: DcGanConfig) -> Sequential {
        // self.main = nn.Sequential(
        // # input is (nc) x 64 x 64
        // nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        // nn.LeakyReLU(0.2, inplace=True),
        // # state size. (ndf) x 32 x 32

        // nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        // nn.BatchNorm2d(ndf * 2),
        // nn.LeakyReLU(0.2, inplace=True),

        // # state size. (ndf*2) x 16 x 16
        // nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        // nn.BatchNorm2d(ndf * 4),
        // nn.LeakyReLU(0.2, inplace=True),

        // # state size. (ndf*4) x 8 x 8
        // nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        // nn.BatchNorm2d(ndf * 8),
        // nn.LeakyReLU(0.2, inplace=True),

        // # state size. (ndf*8) x 4 x 4
        // nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        // nn.Sigmoid()
        // )

        Sequential::from(vec![
            box Conv2d::new(config.nc, config.ndf, 4, 2, 1),
            box LeakyRelu::new(0.2),
            box Conv2d::new(config.ndf, config.ndf * 2, 4, 2, 1),
            box BatchNorm2d::new(config.ndf * 2, config.eps),
            box LeakyRelu::new(0.2),
            box Conv2d::new(config.ndf * 2, config.ndf * 4, 4, 2, 1),
            box BatchNorm2d::new(config.ndf * 4, config.eps),
            box LeakyRelu::new(0.2),
            box Conv2d::new(config.ndf * 4, config.ndf * 8, 4, 2, 1),
            box BatchNorm2d::new(config.ndf * 8, config.eps),
            box LeakyRelu::new(0.2),
            box Conv2d::new(config.ndf * 8, 1, 4, 1, 0),
            box Sigmoid,
        ])
    }
}

impl Parameter for DcGan {
    fn init(&self) {
        self.generator.init();
        self.discriminator.init();
    }
    fn params(&self) -> Option<Vec<&Var>> {
        gather_params(vec![self.generator.params(), self.discriminator.params()])
    }
}
