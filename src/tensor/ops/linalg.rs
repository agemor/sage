use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

impl Tensor {
    // * Creates new tensor
    // (broadcast) batch matrix multiplication
    // e.g., (3, 5, 1, 7, 11) matmul (9, 11, 2) = > (3, 5, 9, 7, 2)
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let rank_a = self.rank();
        let rank_b = other.rank();

        // ensure at least two dims
        if rank_a < 2 || rank_b < 2 {
            panic!("not a matrix");
        }

        // check last two dims are compatible,
        if self.shape[rank_a - 1] != other.shape[rank_b - 2] {
            panic!("matrix not compatible");
        }

        // if a_dim=2, b_dim =2 return matmul
        if rank_a == 2 && rank_b == 2 {
            // let a = self.to_ndarray();
            // let b = other.to_ndarray();
            //
            // let a2d = a.into_dimensionality::<ndarray::Ix2>().unwrap();
            // let b2d = b.into_dimensionality::<ndarray::Ix2>().unwrap();
            // let c2d = a2d.dot(&b2d);
            // Tensor::from_ndarray(c2d.into_dyn())

            gemm(self, other)
        } else {
            // create a shared shape
            let (a_batch, a_mat) = self.shape.split(rank_a - 2);
            let (b_batch, b_mat) = other.shape.split(rank_b - 2);

            // shape broadcast
            let batch_shape = Shape::union(a_batch, b_batch).unwrap();

            let mut a_shape = batch_shape;
            let mut b_shape = batch_shape;

            a_shape.extend(&a_mat);
            b_shape.extend(&b_mat);

            // real broadcast
            let a = self.upcast(a_shape).unwrap();
            let b = other.upcast(b_shape).unwrap();

            let c = a
                .along_axis(0)
                .zip(b.along_axis(0))
                .map(|(a, b)| a.matmul(&b))
                .collect::<Vec<Tensor>>();

            let c_ref = c.iter().map(|a| a).collect::<Vec<&Tensor>>();

            Tensor::stack(&c_ref, 0).unwrap()
        }
    }

    // * Creates new tensor
    // batch operations are allowed
    pub fn matvec(&self, v: &Tensor) -> Tensor {
        // (A, B) * (B, 1) = (A, 1)
        let v = v.expand_dims(-1);
        self.matmul(&v).squeeze(-1)
    }

    pub fn tensordot(&self) -> Tensor {
        Tensor::null()
    }
}

fn gemm(a: &Tensor, b: &Tensor) -> Tensor {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    //println!("gemm {},  {}", a.shape, b.shape);

    let mut v = Vec::with_capacity(m * n);
    //println!("m, k, n {},  {}, {}", m, k, n);

    unsafe {
        v.set_len(m * n);

        // common parameters for gemm
        let ap = a.arr().as_ptr().add(a.offset);
        let bp = b.arr().as_ptr().add(b.offset);
        let cp = v.as_mut_ptr();
        // println!("ap, bp, cp {}, {}, {}", a.offset, b.offset, 0);

        let ast = a.strides();
        let bst = b.strides();
        let cst = Shape::default_strides([m, n]);
        //println!("ap, bp, cp {:?}, {:?}, {:?}", ast, bst, cst);

        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            ap,
            ast[0] as isize,
            ast[1] as isize,
            bp,
            bst[0] as isize,
            bst[1] as isize,
            0.0,
            cp,
            cst[0] as isize,
            cst[1] as isize,
        );
    }
    Tensor::from_vec([m, n], v)
}

#[cfg(test)]
mod test {
    use super::*;
}
