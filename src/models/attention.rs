use crate::autodiff::Var;
use crate::layers::{gather_params, Affine, Layer};
use crate::models::common::{Dropout, LayerNorm, Softmax};

struct MultiHeadAttention {
    embed_dim: usize,
    head_dim: usize,
    num_heads: usize,

    key_proj: Affine,
    query_proj: Affine,
    value_proj: Affine,

    attn_softmax: Softmax,
    dense: Affine,
    dropout: Dropout,
    norm: LayerNorm,
}

impl MultiHeadAttention {
    pub fn new(
        embed_dim: usize,
        head_dim: usize,
        num_heads: usize,
        dropout_prob: f32,
        layer_norm_eps: f32,
    ) -> Self {
        MultiHeadAttention {
            embed_dim,
            head_dim,
            num_heads,
            key_proj: Affine::new(embed_dim, embed_dim),
            query_proj: Affine::new(embed_dim, embed_dim),
            value_proj: Affine::new(embed_dim, embed_dim),
            attn_softmax: Softmax::new(-1),
            dense: Affine::new(embed_dim, embed_dim),
            dropout: Dropout::new(dropout_prob),
            norm: LayerNorm::new([embed_dim], layer_norm_eps),
        }
    }

    pub fn forward(&self, x: &Var, attn_mask: &Var) {
        // (N, L, E) -> (N, L, num_heads * head_dim)
        let query = self.query_proj.forward(x);
        let key = self.key_proj.forward(x);
        let value = self.value_proj.forward(x);

        // (N, L, num_heads * head_dim) -> (N, num_heads, L, head_dim)
        let query = self.seperate_heads(query);
        let key = self.seperate_heads(key);
        let value = self.seperate_heads(value);

        // Calculate the attention scores
        // (N, num_heads, L, head_dim) * (N, num_head, head_dim, L) -> (N, num_head, L, L)
        let attention = matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)

    }

    fn seperate_heads(&self, features: Var) -> Var {
        Var
    }

    fn merge_heads(&self, features: Var) -> Var {
        Var
    }

    fn extend_mask(&self, mask: Var) -> Var {
        Var
    }
}

impl Layer for MultiHeadAttention {
    fn init(&self) {
        // init
    }

    fn forward(&self, x: &Var) -> Var {
        unimplemented!()
    }

    fn params(&self) -> Option<Vec<&Var>> {
        gather_params(vec![
            self.key_proj.params(),
            self.query_proj.params(),
            self.value_proj.params(),
            self.dense.params(),
            self.norm.params(),
        ])
    }
}
