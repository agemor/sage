use crate::tensor::shape::ToIndex;
use crate::tensor::Tensor;
use num_traits::FromPrimitive;

impl Tensor {
    pub fn mean(&self) -> f32 {
        self.logical_iter().sum::<f32>() / (self.size() as f32)
    }

    pub fn max_axis<I>(&self, axis: I, retain_axis: bool) -> Tensor
    where
        I: ToIndex,
    {
        let axis = axis.to_index(self.rank());
        let max = self.fold_axis(axis, f32::MIN, |&a, &b| if a > b { a } else { b });
        if retain_axis {
            max.expand_dims(axis)
        } else {
            max
        }
    }

    pub fn min_axis<I>(&self, axis: I, retain_axis: bool) -> Tensor
    where
        I: ToIndex,
    {
        (-self).max_axis(axis, retain_axis)
    }

    pub fn mean_axis<I>(&self, axis: I, retain_axis: bool) -> Tensor
    where
        I: ToIndex,
    {
        let axis = axis.to_index(self.rank());
        self.sum_axis(axis, retain_axis) / self.shape[axis] as f32
    }

    // variance along given axis
    pub fn var_axis<I>(&self, axis: I, retain_axis: bool) -> Tensor
    where
        I: ToIndex,
    {
        let axis = axis.to_index(self.rank());

        (self - self.mean_axis(axis, true))
            .pow(2.0)
            .sum_axis(axis, retain_axis)
            / (self.shape[axis] - 1) as f32
    }

    pub fn argmax<I>(&self, axis: I) -> Tensor
    where
        I: ToIndex,
    {
        let axis = axis.to_index(self.rank());

        let mut folded_shape = self.shape;
        folded_shape.remove(axis);

        let max_val = Tensor::from_elem(folded_shape, f32::MIN);
        let max_idx = Tensor::from_elem(folded_shape, 0.0);

        for (i, t) in self.along_axis(axis).enumerate() {
            t.logical_iter()
                .zip(max_val.random_iter_mut())
                .zip(max_idx.random_iter_mut())
                .for_each(|((x, mx), idx)| {
                    if x > mx {
                        *mx = *x;
                        *idx = f32::from_usize(i).unwrap();
                    }
                });
        }
        max_idx
    }

    pub fn argmin<I>(&self, axis: I) -> Tensor
    where
        I: ToIndex,
    {
        (-self).argmax(axis)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_max_axis() {
        let a = Tensor::from_elem([3, 2, 5], 10.0);
        let b = Tensor::from_elem([3, 2, 5], 3.0);

        // (3, 4, 5)
        let c = Tensor::cat(&[&a, &b], 1).unwrap();

        assert_eq!(c.max_axis(1, true), Tensor::from_elem([3, 1, 5], 10.0));
    }

    #[test]
    fn test_argmax() {
        let a = Tensor::from_slice(
            [3, 5],
            &[
                0.37894, -1.43962, -0.03472, 1.50011, 1.10574, 1.20776, -0.74392, -0.10786,
                0.48039, -0.82024, -0.62761, -0.94768, 0.75950, 1.23026, 1.93393,
            ],
        );

        let b = Tensor::from_slice([5], &[1., 1., 2., 0., 2.]);

        assert!(a.argmax(0).equals(&b, 0.001));
    }
}
