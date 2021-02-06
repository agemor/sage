// Tensor implementation using ndarray
pub type NdArray = ndarray::Array<f32, ndarray::IxDyn>;

pub struct Tensor {
    pub shape: Shape,
}

#[derive(Clone, Eq, PartialEq)]
pub struct Shape {
    pub dim: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ShapeError;

impl Shape {
    pub fn new(dim: &[usize]) -> Shape {
        Shape { dim: dim.to_vec() }
    }

    pub fn expand_dim(&mut self, axis:usize)  {
        self.dim.insert(axis, 1);
    }

    pub fn broadcast(&self, other: &Shape) -> Result<Shape, ShapeError> {
        let s1 = &self.dim;
        let s2 = &other.dim;

        if s1 == s2 {
            Ok(Shape::new(s1))
        }
        // Do broadcasting
        else {
            let mut sn: Vec<usize>;
            let mut zip;

            if s1.len() > s2.len() {
                sn = s1.clone();
                zip = s1.iter().zip(s2.iter())
            } else {
                sn = s2.clone();
                zip = s2.iter().zip(s1.iter())
            };

            for (i, (d1, d2)) in zip.enumerate() {
                if *d1 == 1 || *d1 == *d2 {
                    sn[i] = *d2;
                } else if *d2 == 1 {
                    sn[i] = *d1;
                } else {
                    return Err(ShapeError);
                }
            }
            Ok(Shape::new(&sn))
        }
    }
}
