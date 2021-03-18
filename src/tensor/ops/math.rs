use crate::tensor::shape::{ToIndex, ToShape};
use crate::tensor::Tensor;
use num_traits::FromPrimitive;

impl Tensor {
    pub fn softmax<I>(&self, axis: I) -> Tensor
    where
        I: ToIndex,
    {
        let axis = axis.to_index(self.rank());

        let max = self.max_axis(axis, true);

        // for numerical stability
        let mut y = self - max;
        y.mapv_inplace(|x| x.exp());

        let sum = y.sum_axis(axis, true);
        y / sum
    }

    pub fn log_sum_exp<I>(&self, axis: I, retain_axis: bool) -> Tensor
    where
        I: ToIndex,
    {
        let axis = axis.to_index(self.rank());

        // (N, K) -> (N, 1)
        let max = self.max_axis(axis, true);

        // (N, K) - (N, 1) -> (N, K)
        let mut y = self - &max;
        y.mapv_inplace(|x| x.exp());

        // (N, K) -> (N, '1)
        let mut sum = y.sum_axis(axis, retain_axis);

        sum.mapv_inplace(|x| x.ln());

        // (N, '1) + (N, 1)
        if retain_axis {
            sum + max
        } else {
            sum + max.squeeze(axis)
        }
    }

    pub fn recip(&self) -> Tensor {
        self.map(|elem| elem.recip())
    }

    pub fn sqrt(&self) -> Tensor {
        self.map(|elem| elem.sqrt())
    }

    pub fn ln(&self) -> Tensor {
        self.map(|elem| elem.ln())
    }

    pub fn pow(&self, n: f32) -> Tensor {
        self.map(|elem| elem.powf(n))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_softmax() {
        let a = Tensor::from_slice(
            [10, 5],
            &[
                1.3546, -0.6792, -2.7762, 1.0667, 0.2832, -2.2399, -0.2652, -0.3175, 1.3278,
                -0.0646, -1.1600, 0.4844, 0.0892, -0.1862, 0.4903, -0.3444, -0.9608, -0.1954,
                1.8407, -0.4723, 0.4810, -0.1955, 1.2180, 0.9043, -0.0608, 0.1345, 0.9163, -0.5973,
                -0.5241, -1.4359, 0.5486, 1.6099, 0.6888, -0.2405, 0.6788, -0.0528, 1.3258, 0.7182,
                0.8407, -0.1208, -0.1869, 0.1736, -0.4857, -1.7939, 0.8562, 0.4678, 0.4607,
                -0.9925, -1.5066, -0.6263,
            ],
        );

        let b = Tensor::from_slice(
            [10, 5],
            &[
                0.4466, 0.0584, 0.0072, 0.3348, 0.1530, 0.0169, 0.1215, 0.1153, 0.5977, 0.1485,
                0.0571, 0.2955, 0.1990, 0.1511, 0.2973, 0.0802, 0.0433, 0.0931, 0.7129, 0.0706,
                0.1752, 0.0891, 0.3662, 0.2676, 0.1019, 0.2277, 0.4976, 0.1095, 0.1179, 0.0474,
                0.1507, 0.4357, 0.1734, 0.0685, 0.1717, 0.0952, 0.3777, 0.2057, 0.2325, 0.0889,
                0.1609, 0.2308, 0.1194, 0.0323, 0.4567, 0.3705, 0.3679, 0.0860, 0.0514, 0.1241,
            ],
        );

        assert!(a.softmax(-1).equals(&b, 0.001))
    }

    #[test]
    fn test_log_sum_exp() {
        let a = Tensor::from_slice(
            [10, 5],
            &[
                -0.5428, -0.0058, 0.7255, -0.5670, 0.2610, 1.2867, -1.3333, 0.4835, 1.6653, 1.6967,
                0.8039, -0.6021, -0.6879, 1.1528, -2.4406, -1.7149, 0.1154, -0.6681, -0.8789,
                1.0272, -0.3563, 0.3535, -1.8956, 1.7223, 0.2468, 1.4946, -0.2689, 1.6087, 0.5768,
                0.1741, 0.3104, 0.2249, 0.5557, 0.2956, 0.2743, -1.1335, -2.1608, 0.8473, 0.9657,
                0.9091, -0.4426, 0.1262, -0.4767, 0.0135, 1.9762, -1.6753, -0.2830, 1.6229,
                -0.4841, 1.1718,
            ],
        );

        let b = Tensor::from_slice(
            [10],
            &[
                1.7060, 2.7880, 1.8777, 1.6141, 2.2140, 2.5794, 1.9486, 2.0642, 2.3634, 2.2877,
            ],
        );

        assert!(a.log_sum_exp(-1, false).equals(&b, 0.001));
    }
}