use crate::autodiff::Var;
use crate::layers::activations::Relu;
use crate::layers::{gather_params, Affine, Layer, Sequential};
use crate::models::attention::MultiHeadAttention;
use crate::models::common::{Dropout, Embedding, LayerNorm};

#[derive(Copy, Clone)]
struct BertConfig {
    embed_dim: usize,

    // Embedding
    num_vocabs: usize,
    max_num_tokens: usize,

    // Encoding
    hidden_dim: usize,
    num_encoding_layers: usize,
    num_attention_heads: usize,
    layer_norm_eps: f32,

    dropout_prob: f32,
}

impl BertConfig {
    pub fn base() -> Self {
        BertConfig {
            embed_dim: 768,
            num_vocabs: 30522,
            max_num_tokens: 512,
            hidden_dim: 3072,
            num_encoding_layers: 12,
            num_attention_heads: 12,
            layer_norm_eps: 1e-12,
            dropout_prob: 0.3,
        }
    }
}

struct Bert {
    embedding: BertEmbedding,
    encoder: BertEncoder,
    classifier: Affine,
}

impl Bert {
    pub fn new(config: BertConfig) -> Self {
        Bert {
            embedding: BertEmbedding::new(config),
            encoder: BertEncoder::new(config),
            classifier: Affine::new(config.embed_dim, 2),
        }
    }

    pub fn forward_with(&self, token_ids: &[usize], attn_mask: &Var) -> Var {
        let embeddings = self.embedding.forward_with(token_ids);
        let features = self.encoder.forward_with(embeddings, attn_mask);

        let cls_tokens = features.select(1, 1);
        let logits = self.classifier.forward(&cls_tokens);

        logits
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

    pub fn forward_with(&self, token_ids: &[usize]) -> Var {
        // support 2d token ids
        unimplemented!();

        let pos_ids = (0..token_ids.len()).collect::<Vec<usize>>();
        let type_ids = vec![0; token_ids.len()];
        let word_embeddings = self.word_emb.forward_with(token_ids);
        let pos_embeddings = self.pos_emb.forward_with(&pos_ids);
        let type_embeddings = self.word_emb.forward_with(&type_ids);

        self.norm
            .forward(&(word_embeddings + pos_embeddings + type_embeddings))
    }
}

impl Layer for BertEmbedding {
    fn init(&self) {
        self.word_emb.init();
        self.pos_emb.init();
        self.type_emb.init();
        self.norm.init();
    }

    fn forward(&self, x: &Var) -> Var {
        unimplemented!()
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
            layers: (0..config.num_encoding_layers)
                .into_iter()
                .map(|_| BertLayer::new(config))
                .collect(),
        }
    }

    pub fn forward_with(&self, x: Var, attn_mask: &Var) -> Var {
        self.layers
            .iter()
            .fold(x, |x, layer| layer.forward_with(&x, attn_mask))
    }
}

impl Layer for BertEncoder {
    fn init(&self) {
        self.layers.iter().for_each(|x| x.init());
    }

    fn forward(&self, x: &Var) -> Var {
        panic!("use forward_with");
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
                config.attention_dropout_prob,
                config.layer_norm_eps,
            ),
            ffn: Sequential::from(vec![
                box Affine::new(config.embed_dim, config.hidden_dim),
                box Relu,
                box Affine::new(config.hidden_dim, config.embed_dim),
                box Dropout::new(config.dropout_prob),
            ]),
            norm: LayerNorm::new([config.embed_dim], config.layer_norm_eps),
        }
    }

    pub fn forward_with(&self, x: &Var, attn_mask: &Var) -> Var {
        let interim_features = self.attention.forward_with(x, attn_mask);
        let out_features = self.ffn.forward(&interim_features);
        self.norm.forward(&(out_features + interim_features))
    }
}

impl Layer for BertLayer {
    fn init(&self) {
        self.attention.init();
        self.ffn.init();
        self.norm.init();
    }

    fn forward(&self, x: &Var) -> Var {
        panic!("use forward_with");
    }

    fn params(&self) -> Option<Vec<&Var>> {
        gather_params(vec![
            self.attention.params(),
            self.ffn.params(),
            self.norm.params(),
        ])
    }
}
