use crate::autodiff::var::Var;
use crate::layers::Parameter;
use crate::tensor;
use crate::tensor::Tensor;
use itertools::Itertools;

pub struct Embedding {
    num_embeddings: usize,
    embedding_dim: usize,
    weights: Var,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Embedding {
            num_embeddings,
            embedding_dim,
            weights: Var::with_shape([num_embeddings, embedding_dim]),
        }
    }

    pub fn forward(&self, batch_ids: &[Vec<usize>]) -> Var {
        batch_ids
            .iter()
            .map(|ids| {
                // (W, F)
                ids.iter()
                    .map(|&id| {
                        // (1, F)
                        self.weights.index(id, 0)
                    })
                    .fold1(|acc, feature| acc.concat(feature, 0))
                    .unwrap()
                    // (1, W, F)
                    .unsqueeze(0)
            })
            .fold1(|acc, w| acc.concat(w, 0))
            // (N, W, F)
            .unwrap()
    }
}

impl Parameter for Embedding {
    fn init(&self) {
        self.weights.set_data(Tensor::null());
        //self.weights.set_data(Tensor::zeros(self.weights.shape()));
    }

    fn params(&self) -> Option<Vec<&Var>> {
        Some(vec![&self.weights])
    }
}
