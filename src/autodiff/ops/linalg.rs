use crate::autodiff::ops::Operator;
use crate::autodiff::var::{ToVar, Var};
use crate::tensor::shape::{Shape, ToIndex};
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

    fn is_fdb(&self) -> bool {
        true
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

    fn is_fdb(&self) -> bool {
        true
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

    pub fn transpose<I, J>(&self, axis_a: I, axis_b: J) -> Var
    where
        I: ToIndex,
        J: ToIndex,
    {
        Transpose {
            axis_a: axis_a.to_index(self.rank()),
            axis_b: axis_b.to_index(self.rank()),
        }
        .forward([self])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::diff;

    #[test]
    fn test_matmul() {
        let a_data = Tensor::from_slice(
            [2, 3, 5],
            &[
                0.59291, -0.93341, -0.18223, 0.59965, 0.00716, -0.46003, -0.42066, -0.64802,
                0.55233, 1.73416, -0.22048, 0.19131, -1.55627, 0.71593, 0.46732, -1.04307,
                -0.35595, 1.09157, 1.20627, 1.08326, 0.99519, -1.14578, 0.43299, -1.93212,
                -0.79032, 0.45944, -0.53933, -0.36932, 0.57457, -0.82304,
            ],
        );

        let a_grad_data = Tensor::from_slice(
            [2, 3, 5],
            &[
                -0.07216, -2.58671, 2.36674, 0.32271, 1.93365, -0.07216, -2.58671, 2.36674,
                0.32271, 1.93365, -0.07216, -2.58671, 2.36674, 0.32271, 1.93365, -0.07216,
                -2.58671, 2.36674, 0.32271, 1.93365, -0.07216, -2.58671, 2.36674, 0.32271, 1.93365,
                -0.07216, -2.58671, 2.36674, 0.32271, 1.93365,
            ],
        );

        let b_data = Tensor::from_slice(
            [5, 4],
            &[
                -0.07156, 0.44339, -0.14176, -0.30223, -0.12564, -1.72730, -0.69673, -0.03704,
                1.63514, -0.71166, 0.25041, 1.19285, 0.26871, -0.66543, -0.07882, 0.79825, 0.03763,
                0.64892, 0.37141, 0.87569,
            ],
        );

        let b_grad_data = Tensor::from_slice(
            [5, 4],
            &[
                0.32396, 0.32396, 0.32396, 0.32396, -3.20382, -3.20382, -3.20382, -3.20382,
                -1.23128, -1.23128, -1.23128, -1.23128, 1.71663, 1.71663, 1.71663, 1.71663,
                1.67854, 1.67854, 1.67854, 1.67854,
            ],
        );

        let c_data = Tensor::from_slice(
            [2, 3, 4],
            &[
                -0.06173, 1.61048, 0.47605, 0.12295, -0.76016, 1.74160, 0.79658, 1.34111, -2.34301,
                0.50618, -0.37460, -0.81613, 2.26913, -0.72422, 0.97646, 3.54202, 0.23182, 2.88506,
                0.62440, -1.97623, -0.44558, 0.48170, -0.13282, -0.82150,
            ],
        );

        let a = Var::with_data(a_data);
        let b = Var::with_data(b_data);

        let c = a.matmul(&b);

        // forward check

        assert!(c.data().equals(&c_data, 0.001));

        let grads = diff(&c, &[&a, &b]);

        let a_grad = grads.get(&a).unwrap();
        let b_grad = grads.get(&b).unwrap();

        // backward check
        assert!(a_grad.data().equals(&a_grad_data, 0.001));
        assert!(b_grad.data().equals(&b_grad_data, 0.001));
    }

    #[test]
    fn test_matvec() {
        let a_data = Tensor::from_slice(
            [2, 3, 5],
            &[
                -3.07866, -0.79024, -0.94439, 1.67945, 1.27508, -0.09492, 1.62357, 2.64593,
                1.74636, 0.82562, 1.41111, 0.05571, -0.48702, -0.46361, 0.35668, -1.28441,
                -0.91176, -1.09115, -0.19053, 1.63678, 2.29281, 1.03232, 1.15872, 1.36020,
                -0.72647, 1.21023, 1.16816, -1.05799, 0.13244, -0.97486,
            ],
        );

        let a_grad_data = Tensor::from_slice(
            [2, 3, 5],
            &[
                2.17630, 0.64114, 0.94362, 0.66252, -0.90203, 2.17630, 0.64114, 0.94362, 0.66252,
                -0.90203, 2.17630, 0.64114, 0.94362, 0.66252, -0.90203, 2.17630, 0.64114, 0.94362,
                0.66252, -0.90203, 2.17630, 0.64114, 0.94362, 0.66252, -0.90203, 2.17630, 0.64114,
                0.94362, 0.66252, -0.90203,
            ],
        );

        let b_data = Tensor::from_slice([5], &[2.17630, 0.64114, 0.94362, 0.66252, -0.90203]);

        let b_grad_data = Tensor::from_slice([5], &[0.45616, 2.17776, 0.22410, 4.26431, 2.39283]);

        let c_data = Tensor::from_slice(
            [2, 3],
            &[-8.13538, 3.74338, 2.01827, -6.01211, 8.30155, 3.35153],
        );

        let a = Var::with_data(a_data);
        let b = Var::with_data(b_data);

        let c = a.matvec(&b);

        // forward check
        assert!(c.data().equals(&c_data, 0.001));

        let grads = diff(&c, &[&a, &b]);

        let a_grad = grads.get(&a).unwrap();
        let b_grad = grads.get(&b).unwrap();

        // backward check
        assert!(a_grad.data().equals(&a_grad_data, 0.001));
        assert!(b_grad.data().equals(&b_grad_data, 0.001));
    }
}
