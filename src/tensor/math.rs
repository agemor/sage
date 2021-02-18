use crate::tensor::{to_unsigned_index, Tensor};

impl Tensor {
    pub fn sum(&self) -> f32 {
        self.logical_iter().sum()
    }

    pub fn max_axis(&self, axis: isize) -> Tensor {
        // wow!
        self.fold_axis(axis, f32::MIN, |&a, &b| if a > b { a } else { b })
    }

    pub fn sum_axis(&self, axis: isize) -> Tensor {
        let axis_u = to_unsigned_index(axis, self.rank());

        let mut res_dim = self.dim.clone();
        res_dim.remove(axis_u);
        let mut res = Tensor::zeros(res_dim);

        for t in self.along_axis(axis) {
            res = res + t;
        }
        res
    }

    pub fn softmax(&self, axis: isize) -> Tensor {
        let max = self.max_axis(axis).expand_dims(axis);

        // for numerical stability
        let mut y = self - max;
        y.mapv_inplace(|x| x.exp());

        let sum = y.sum_axis(axis);
        y / sum
    }

    pub fn log_sum_exp(&self, axis: isize) -> Tensor {
        let max = self.max_axis(axis).expand_dims(axis);

        let mut y = self - &max;
        y.mapv_inplace(|x| x.exp());
        let mut sum = y.sum_axis(axis);
        sum.mapv_inplace(move |x| x.ln());
        sum + max
    }
}

// math utility methods
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    a.zip_map(b, |&a, &b| a + b).unwrap()
}

pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    a.zip_map(b, |&a, &b| a - b).unwrap()
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    a.zip_map(b, |&a, &b| a * b).unwrap()
}

pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    a.zip_map(b, |&a, &b| a / b).unwrap()
}

pub fn neg(a: &Tensor) -> Tensor {
    a.map(|&a| -a)
}

pub fn exp(t: &mut Tensor) {
    t.mapv_inplace(|x| x.exp());
}

pub fn ln(t: &mut Tensor) {
    t.mapv_inplace(|x| x.ln());
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sum() {
        let a = Tensor::from_elem([3, 10, 10], 3.0);
        assert_eq!(a.sum(), 900.0);
    }

    #[test]
    fn test_max_axis() {
        let a = Tensor::from_elem([3, 2, 5], 10.0);
        let b = Tensor::from_elem([3, 2, 5], 3.0);

        // (3, 4, 5)
        let c = Tensor::cat(&[a, b], 1).unwrap();

        assert_eq!(
            c.max_axis(1).expand_dims(1),
            Tensor::from_elem([3, 1, 5], 10.0)
        );
    }

    #[test]
    fn test_sum_axis() {
        let a = Tensor::randn([3, 5, 7]);

        assert_eq!(a.sum_axis(1).shape(), &[3, 7]);
        assert_eq!(a.sum_axis(-1), a.fold_axis(-1, 0.0, |&a, &b| a + b));
    }
}
