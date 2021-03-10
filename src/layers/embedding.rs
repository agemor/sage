use crate::autodiff::var::Var;
use crate::layers::Parameter;

pub struct Embedding {
    num_embeddings: usize,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Embedding {
            num_embeddings,
            embedding_dim,
        }
    }

    pub fn forward(&self, ids: &[usize]) -> Var {
        unimplemented!();
    }
}

impl Parameter for Embedding {}
