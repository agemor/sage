use crate::autodiff::var::Var;
use crate::layers::activations::Relu;
use crate::layers::attention::MultiHeadAttention;
use crate::layers::base::{Dense, Dropout, Sequential};
use crate::layers::embedding::Embedding;
use crate::layers::normalization::LayerNorm;
use crate::layers::{gather_params, Parameter, Stackable};

#[derive(Copy, Clone)]
pub struct BertConfig {
    embed_dim: usize,

    // Embedding
    num_vocabs: usize,
    max_num_tokens: usize,

    // Encoding
    hidden_dim: usize,
    num_layers: usize,
    num_attention_heads: usize,
    layer_norm_eps: f32,
    dropout_prob: f32,

    // classifier
    num_classes: usize,
}

impl BertConfig {
    pub fn base() -> Self {
        BertConfig {
            embed_dim: 768,
            num_vocabs: 30522,
            max_num_tokens: 512,
            hidden_dim: 3072,
            num_layers: 12,
            num_attention_heads: 12,
            layer_norm_eps: 1e-12,
            dropout_prob: 0.3,
            num_classes: 2,
        }
    }
}

pub struct Bert {
    embedding: BertEmbedding,
    encoder: BertEncoder,
    classifier: Dense,
}

impl Bert {
    pub fn new(config: BertConfig) -> Self {
        Bert {
            embedding: BertEmbedding::new(config),
            encoder: BertEncoder::new(config),
            classifier: Dense::new(config.embed_dim, config.num_classes),
        }
    }

    pub fn forward(&self, token_ids: &[Vec<usize>], attn_mask: &Var) -> Var {
        let embeddings = self.embedding.forward(token_ids);
        let features = self.encoder.forward(embeddings, attn_mask);

        let cls_tokens = features.index(1, 1).squeeze(1);

        let logits = self.classifier.forward(&cls_tokens);

        logits
    }
}

impl Parameter for Bert {
    fn init(&self) {
        self.embedding.init();
        self.encoder.init();
        self.classifier.init();
    }

    fn params(&self) -> Option<Vec<&Var>> {
        gather_params(vec![
            self.embedding.params(),
            self.encoder.params(),
            self.classifier.params(),
        ])
    }
}

struct BertEmbedding {
    word_emb: Embedding,
    pos_emb: Embedding,
    type_emb: Embedding,
    norm: LayerNorm,
    dropout: Dropout,
}

impl BertEmbedding {
    pub fn new(config: BertConfig) -> Self {
        BertEmbedding {
            word_emb: Embedding::new(config.num_vocabs, config.embed_dim),
            pos_emb: Embedding::new(config.max_num_tokens, config.embed_dim),
            type_emb: Embedding::new(2, config.embed_dim),
            norm: LayerNorm::new([config.embed_dim], config.layer_norm_eps),
            dropout: Dropout::new(config.dropout_prob),
        }
    }

    pub fn forward(&self, token_ids: &[Vec<usize>]) -> Var {
        // support 2d token ids
        let pos_ids = (0..token_ids[0].len()).collect::<Vec<usize>>();
        let type_ids = vec![0; token_ids[0].len()];
        let word_embeddings = self.word_emb.forward(token_ids);
        let pos_embeddings = self.pos_emb.forward(&[pos_ids]);
        let type_embeddings = self.word_emb.forward(&[type_ids]);

        self.norm
            .forward(&(word_embeddings + pos_embeddings + type_embeddings))
    }
}

impl Parameter for BertEmbedding {
    fn init(&self) {
        self.word_emb.init();

        self.pos_emb.init();
        self.type_emb.init();
        self.norm.init();
    }

    fn params(&self) -> Option<Vec<&Var>> {
        gather_params(vec![
            self.word_emb.params(),
            self.pos_emb.params(),
            self.type_emb.params(),
            self.norm.params(),
        ])
    }
}

struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn new(config: BertConfig) -> Self {
        BertEncoder {
            layers: (0..config.num_layers)
                .into_iter()
                .map(|_| BertLayer::new(config))
                .collect(),
        }
    }

    pub fn forward(&self, x: Var, attn_mask: &Var) -> Var {
        self.layers
            .iter()
            .fold(x, |x, layer| layer.forward(&x, attn_mask))
    }
}

impl Parameter for BertEncoder {
    fn init(&self) {
        self.layers.iter().for_each(|x| x.init());
    }

    fn params(&self) -> Option<Vec<&Var>> {
        gather_params(self.layers.iter().map(|x| x.params()).collect())
    }
}

struct BertLayer {
    attention: MultiHeadAttention,
    ffn: Sequential,
    norm: LayerNorm,
}

impl BertLayer {
    pub fn new(config: BertConfig) -> Self {
        BertLayer {
            attention: MultiHeadAttention::new(
                config.embed_dim,
                config.num_attention_heads,
                config.dropout_prob,
                config.layer_norm_eps,
            ),
            ffn: Sequential::from(vec![
                box Dense::new(config.embed_dim, config.hidden_dim),
                box Relu,
                box Dense::new(config.hidden_dim, config.embed_dim),
                box Dropout::new(config.dropout_prob),
            ]),
            norm: LayerNorm::new([config.embed_dim], config.layer_norm_eps),
        }
    }

    pub fn forward(&self, x: &Var, attn_mask: &Var) -> Var {
        let interim_features = self.attention.forward(x, attn_mask);
        let out_features = self.ffn.forward(&interim_features);
        self.norm.forward(&(out_features + interim_features))
    }
}

impl Parameter for BertLayer {
    fn init(&self) {
        self.attention.init();
        self.ffn.init();
        self.norm.init();
    }

    fn params(&self) -> Option<Vec<&Var>> {
        gather_params(vec![
            self.attention.params(),
            self.ffn.params(),
            self.norm.params(),
        ])
    }
}
