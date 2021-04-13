use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

impl Tensor {
    // * Creates new tensor
    // (broadcast) batch matrix multiplication
    // e.g., (3, 5, 1, 7, 11) matmul (9, 11, 2) = > (3, 5, 9, 7, 2)
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let rank_a = self.order();
        let rank_b = other.order();

        // ensure at least two dims
        if rank_a < 2 || rank_b < 2 {
            panic!("not a matrix");
        }

        // check last two dims are compatible,
        if self.shape[rank_a - 1] != other.shape[rank_b - 2] {
            panic!("matrix not compatible");
        }

        self.contract(other, [rank_a - 1], [rank_b - 2])
    }

    // * Creates new tensor
    // batch operations are allowed
    pub fn matvec(&self, other: &Tensor) -> Tensor {
        // (A, B) * (B, 1) = (A, 1)
        let rank_a = self.order();
        let rank_b = other.order();

        // ensure at least two dims
        if rank_a < 2 || rank_b < 1 {
            panic!("not a matrix or vector");
        }

        // check last two dims are compatible,
        if self.shape[rank_a - 1] != other.shape[rank_b - 1] {
            panic!("matrix-vector not compatible");
        }

        self.contract(other, [rank_a - 1], [rank_b - 1])
    }
}

#[cfg(test)]
mod test {
    use super::*;
}
