use crate::tensor::{Tensor};
use crate::shape::{ToShape, ToIndex};

impl Tensor {
    pub fn sum(&self) -> f32 {
        self.logical_iter().sum()
    }

    pub fn mean(&self) -> f32 {
        self.logical_iter().sum::<f32>() / (self.size() as f32)
    }

    pub fn max_axis<I>(&self, axis: I, retain_axis: bool) -> Tensor
        where I: ToIndex
    {
        let axis = axis.to_index(self.rank());
        let max = self.fold_axis(axis, f32::MIN, |&a, &b| if a > b { a } else { b });
        if retain_axis {
            max.expand_dims(axis)
        } else {
            max
        }
    }

    pub fn sum_axis<I>(&self, axis: I, retain_axis: bool) -> Tensor
        where I: ToIndex
    {
        let axis = axis.to_index(self.rank());

        let mut new_shape = self.shape;
        new_shape.remove(axis);

        let mut summed = Tensor::zeros(new_shape);


        for t in self.along_axis(axis) {
            summed = summed + t;
        }

        if retain_axis {
            summed.expand_dims(axis)
        } else {
            summed
        }
    }

    pub fn mean_axis<I>(&self, axis: I, retain_axis: bool) -> Tensor
        where I: ToIndex
    {
        let axis = axis.to_index(self.rank());
        self.sum_axis(axis, retain_axis) / self.shape[axis] as f32
    }

    pub fn softmax<I>(&self, axis: I) -> Tensor
        where I: ToIndex
    {
        let axis = axis.to_index(self.rank());

        let max = self.max_axis(axis, true);

        // for numerical stability
        let mut y = self - max;
        y.mapv_inplace(|x| x.exp());

        let sum = y.sum_axis(axis, true);
        y / sum
    }

    pub fn log_sum_exp<I>(&self, axis: I) -> Tensor
        where I: ToIndex
    {
        let axis = axis.to_index(self.rank());

        let max = self.max_axis(axis, true);

        let mut y = self - &max;

        y.mapv_inplace(|x| x.exp());
        let mut sum = y.sum_axis(axis, true);
        sum.mapv_inplace(|x| x.ln());

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
        let a = Tensor::from_elem([3, 10, 1], 3.0);
        assert_eq!(a.sum(), 90.0_f32);
    }

    #[test]
    fn test_max_axis() {
        let a = Tensor::from_elem([3, 2, 5], 10.0);
        let b = Tensor::from_elem([3, 2, 5], 3.0);

        // (3, 4, 5)
        let c = Tensor::cat(&[a, b], 1).unwrap();

        assert_eq!(
            c.max_axis(1, true),
            Tensor::from_elem([3, 1, 5], 10.0)
        );
    }

    #[test]
    fn test_sum_axis() {
        let a = Tensor::randn([3, 5, 7]);

        assert_eq!(a.sum_axis(1, false).shape(), [3, 7].to_shape());
        assert_eq!(a.sum_axis(-1, false), a.fold_axis(-1, 0.0, |&a, &b| a + b));
    }
}
