use crate::tensor::Tensor;

pub struct AlongAxisIter<'a> {
    t: &'a Tensor,
    axis: usize,
    index: usize,
}

impl<'a> AlongAxisIter<'a> {
    pub fn new(t: &'a Tensor, axis: usize) -> Self {
        AlongAxisIter { t, axis, index: 0 }
    }
}

impl<'a> Iterator for AlongAxisIter<'a> {
    type Item = Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.t.shape[self.axis] {
            self.index += 1;
            Some(
                self.t
                    .index_axis((self.index - 1) as isize, self.axis as isize),
            )
        } else {
            None
        }
    }
}
