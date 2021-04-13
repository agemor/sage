use crate::tensor::{Element, Tensor};
use num_traits::NumOps;
use std::iter::Sum;

impl<T> Tensor<T>
where
    T: Element + NumOps + Sum,
{
    pub fn matmul(&self, other: &Tensor<T>) -> Tensor<T> {
        let rank1 = self.order();
        let rank2 = other.order();

        // ensure at least two dims
        if rank1 < 2 || rank2 < 2 {
            panic!("not a matrix");
        }

        // check last two dims are compatible,
        if self.shape()[rank1 - 1] != other.shape()[rank2 - 2] {
            panic!("matrix not compatible");
        }

        self.contract(other, [rank1 - 1], [rank2 - 2])
    }

    pub fn matvec(&self, other: &Tensor<T>) -> Tensor<T> {
        // (A, B) * (B, 1) = (A, 1)
        let rank1 = self.order();
        let rank2 = other.order();

        // ensure at least two dims
        if rank1 < 2 || rank2 < 1 {
            panic!("not a matrix or vector");
        }

        // check last two dims are compatible,
        if self.shape()[rank1 - 1] != other.shape()[rank2 - 1] {
            panic!("matrix-vector not compatible");
        }

        self.contract(other, [rank1 - 1], [rank2 - 1])
    }
}
