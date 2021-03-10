use crate::autodiff::ops::Operator;
use crate::autodiff::var::{ToVar, Var};
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

// matrix operations
struct Matmul;

struct Matvec;

struct Transpose {
    axis_a: usize,
    axis_b: usize,
}

impl Operator<2> for Matmul {
    fn compute(&self, x: [&Tensor; 2]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];

        x0.matmul(x1)
    }

    fn forward(self, x: [&Var; 2]) -> Var {
        let x0 = x[0];
        let x1 = x[1];

        if x0.rank() < 2 || x1.rank() < 2 {
            panic!("should provide a matrix");
        }

        if x0.shape()[x0.rank() - 1] != x1.shape()[x1.rank() - 2] {
            panic!("matrix not compatible");
        }

        let (x0_batch, _) = x0.shape().split(x0.rank() - 2);
        let (x1_batch, _) = x1.shape().split(x1.rank() - 2);

        // shape broadcast
        let mut batch = Shape::union(x0_batch, x1_batch).unwrap();

        // add matrix dim
        batch.insert(-1, x0.shape()[x0.rank() - 2]);
        batch.insert(-1, x1.shape()[x1.rank() - 1]);

        Var::from_binary_op(batch, self, [x0, x1])
    }

    fn backward(&self, x: [&Var; 2], gy: &Var) -> [Var; 2] {
        // (*, A, B)
        let x0 = x[0];

        // (*, B, C)
        let x1 = x[1];

        // gy: (*, A, C)

        // (*, A, B) = (*, A, C) (*, C, B)
        let gx0 = gy.matmul(x1.transpose(-1, -2)).sum_to(x0.shape());

        // (*, B, C) = (*, B, A) (*, A, C)
        let gx1 = x0.transpose(-1, -2).matmul(gy).sum_to(x1.shape());

        [gx0, gx1]
    }
}

impl Operator<2> for Matvec {
    fn compute(&self, x: [&Tensor; 2]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];

        x0.matvec(x1)
    }

    fn forward(self, x: [&Var; 2]) -> Var {
        let x0 = x[0];
        let x1 = x[1];

        if x0.rank() < 2 || x1.rank() < 1 {
            panic!("invalid matrix or vector");
        }

        if x0.shape()[x0.rank() - 1] != x1.shape()[x1.rank() - 1] {
            panic!("incompatible matrix-vector");
        }

        let (x0_batch, _) = x0.shape().split(x0.rank() - 2);
        let (x1_batch, _) = x1.shape().split(x1.rank() - 1);

        // shape broadcast
        let mut batch = Shape::union(x0_batch, x1_batch).unwrap();

        // add matrix dim
        batch.insert(-1, x0.shape()[x0.rank() - 2]);

        Var::from_binary_op(batch, self, [x0, x1])
    }

    fn backward(&self, x: [&Var; 2], gy: &Var) -> [Var; 2] {
        // (*, A, B)
        let x0 = x[0];

        // (*, B)
        let x1 = x[1];

        // gy: (*, A)

        // (*, A, B) = (*, A, 1) (*, 1, B)
        //let gx0 = sum_to(
        //    &matmul(&expand(gy, gy.rank()), &expand(x1, x1.rank() - 1)),
        //    x0.shape(),
        //);
        let gx0 = gy.unsqueeze(-1).matmul(x1.unsqueeze(-2)).sum_to(x0.shape());

        // (*, B) = (*, B, A) (*, A)
        //let gx1 = sum_to(&matvec(&transpose(x0, -1, -2), gy), x1.shape());
        let gx1 = x0.transpose(-1, -2).matvec(gy).sum_to(x1.shape());
        [gx0, gx1]
    }
}

// Swap last two components of tensor
impl Operator<1> for Transpose {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x.transpose(self.axis_a, self.axis_b)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];

        if x.rank() < 2 {
            panic!("cannot transpose on a vector");
        }

        let mut shape = x.shape();
        shape.swap(self.axis_a, self.axis_b);

        Var::from_unary_op(shape, self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = gy.transpose(self.axis_a, self.axis_b);
        [gx]
    }
}

impl Var {
    pub fn matmul<V: ToVar>(&self, other: V) -> Var {
        Matmul.forward([self, &other.to_var()])
    }

    pub fn matvec<V: ToVar>(&self, other: V) -> Var {
        Matvec.forward([self, &other.to_var()])
    }

    pub fn matvec_<V: ToVar>(&self, other: V) -> Var {
        self.matmul(other.to_var().unsqueeze(-1)).squeeze(-1)
    }
}
