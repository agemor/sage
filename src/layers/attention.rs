use crate::autodiff::var::Var;
use crate::layers::base::{Dense, Dropout, Softmax};
use crate::layers::normalization::LayerNorm;
use crate::layers::{gather_params, Parameter, Stackable};

pub struct MultiHeadAttention {
    embed_dim: usize,
    head_dim: usize,
    num_heads: usize,

    key_proj: Dense,
    query_proj: Dense,
    value_proj: Dense,

    attn_softmax: Softmax,
    dense: Dense,
    dropout: Dropout,
    norm: LayerNorm,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize, dropout_prob: f32, layer_norm_eps: f32) -> Self {
        MultiHeadAttention {
            embed_dim,
            head_dim: embed_dim / num_heads,
            num_heads,
            key_proj: Dense::new(embed_dim, embed_dim),
            query_proj: Dense::new(embed_dim, embed_dim),
            value_proj: Dense::new(embed_dim, embed_dim),
            attn_softmax: Softmax::new(-1),
            dense: Dense::new(embed_dim, embed_dim),
            dropout: Dropout::new(dropout_prob),
            norm: LayerNorm::new([embed_dim], layer_norm_eps),
        }
    }

    pub fn forward(&self, x: &Var, attn_mask: &Var) -> Var {
        // (N, L, E) -> (N, L, num_heads * head_dim)
        let query = self.query_proj.forward(x);
        let key = self.key_proj.forward(x);
        let value = self.value_proj.forward(x);

        // (N, L, num_heads * head_dim) -> (N, num_heads, L, head_dim)
        let query = self.separate_heads(query);
        let key = self.separate_heads(key);
        let value = self.separate_heads(value);

        // Calculate the attention scores
        // (N, num_heads, L, head_dim) * (N, num_head, head_dim, L) -> (N, num_head, L, L)
        let attention = query.matmul(key.transpose(-1, -2)) / self.head_dim.sqrt();

        // Apply softmax to the attention scores
        let attention = self
            .attn_softmax
            .forward(&(attention + self.extend_mask(attn_mask)));

        // Applying attention weights
        // (N, num_heads, L, L) * (N, num_heads, L, head_dim) -> (N, num_heads, L, head_dim)
        let attention_value = attention.matmul(&value);

        // (N, num_heads, L, head_dim) -> (N, L, num_heads * head_dim)
        let attention_value = self.merge_heads(attention_value);

        let y = self.dense.forward(&attention_value);
        self.norm.forward(&(y + x))
    }

    fn separate_heads(&self, features: Var) -> Var {
        // (N, L, num_heads * head_dim) -> (N, L, num_heads, head_dim)
        let batch_size = features.shape().sizes()[0];
        let input_len = features.shape().sizes()[1];

        let features = features.reshape([batch_size, input_len, self.num_heads, self.head_dim]);

        // (N, L, num_heads, head_dim) -> (N, num_heads, L, head_dim)
        features.transpose(2, 1)
    }

    fn merge_heads(&self, features: Var) -> Var {
        //# (N, num_heads, L, head_dim) -> (N, L, num_heads, head_dim)
        let features = features.transpose(2, 1);

        // # (N, L, num_heads, head_dim) -> (N, L, num_heads * head_dim)
        let batch_size = features.shape().sizes()[0];
        let input_len = features.shape().sizes()[1];

        features.reshape([batch_size, input_len, -1])
    }

    fn extend_mask(&self, mask: &Var) -> Var {
        //# (N, L) -> (N, 1, 1, L)

        let batch_size = mask.shape().sizes()[0];
        let input_len = mask.shape().sizes()[1];

        let extended_mask = mask.reshape([batch_size, 1, 1, input_len]);

        //# Adding -1e5 makes masked locations zeroed out during softmax
        (1.0 - extended_mask) * -1e5
    }
}

impl Parameter for MultiHeadAttention {
    fn init(&self) {
        // init
        self.key_proj.init();
        self.query_proj.init();
        self.value_proj.init();
        self.dense.init();
        self.norm.init();
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
