use crate::autodiff::ops::activations::{sigmoid, tanh};
use crate::autodiff::var::Var;
use crate::layers::base::Dense;
use crate::layers::{gather_params, Parameter, Stackable};
use crate::tensor::Tensor;
use itertools::Itertools;

pub struct Lstm {
    input_size: usize,
    hidden_size: usize,

    weights: Dense,
    hidden: Dense,
}

impl Lstm {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Lstm {
            input_size,
            hidden_size,
            weights: Dense::new(input_size, hidden_size * 4),
            hidden: Dense::new(hidden_size, hidden_size * 4),
        }
    }

    pub fn forward_with_states(&self, x: &Var, hidden: &Var, cell: &Var) -> (Var, Var, Var) {
        // bs, seq_sz, _ = x.size()
        // hidden_seq = []
        // if init_states is None:
        //     h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
        //                 torch.zeros(bs, self.hidden_size).to(x.device))
        // else:
        // h_t, c_t = init_states
        //
        // HS = self.hidden_size
        // for t in range(seq_sz):
        //     x_t = x[:, t, :]
        // # batch the computations into a single matrix multiplication
        //     gates = x_t @ self.W + h_t @ self.U + self.bias
        //     i_t, f_t, g_t, o_t = (
        //         torch.sigmoid(gates[:, :HS]), # input
        //         torch.sigmoid(gates[:, HS:HS*2]), # forget
        //         torch.tanh(gates[:, HS*2:HS*3]),
        //         torch.sigmoid(gates[:, HS*3:]), # output
        //     )
        //     c_t = f_t * c_t + i_t * g_t
        //     h_t = o_t * torch.tanh(c_t)
        //     hidden_seq.append(h_t.unsqueeze(0))
        // hidden_seq = torch.cat(hidden_seq, dim=0)
        // # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        // hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        // return hidden_seq, (h_t, c_t)
        //

        let seq_len = x.shape()[1];

        let h = self.hidden_size;

        let mut h_t = hidden.clone();
        let mut c_t = cell.clone();

        let mut hidden_seq = Vec::new();

        for i in 0..seq_len {
            let x_t = x.index(i, 1).squeeze(1);

            let gates = self.weights.forward(&x_t) + self.hidden.forward(&h_t);

            let i_t = sigmoid(gates.slice(h * 0, h, -1));
            let f_t = sigmoid(gates.slice(h * 1, h, -1));
            let g_t = tanh(gates.slice(h * 2, h, -1));
            let o_t = sigmoid(gates.slice(h * 3, h, -1));

            // println!(
            //     "i: {}, f: {}, g: {}, o: {}, c: {}, h: {}",
            //     i_t.shape(),
            //     f_t.shape(),
            //     g_t.shape(),
            //     o_t.shape(),
            //     c_t.shape(),
            //     h_t.shape()
            // );

            c_t = f_t * &c_t + i_t * g_t;
            h_t = o_t * tanh(&c_t);
            hidden_seq.push(h_t.unsqueeze(0));
        }

        let hidden_seq = hidden_seq
            .into_iter()
            .fold1(|acc, v| acc.concat(v, 1))
            .unwrap();

        (hidden_seq, h_t, c_t)
    }
}

impl Parameter for Lstm {
    fn init(&self) {
        self.weights.init();
        self.hidden.init();
    }
    fn params(&self) -> Option<Vec<&Var>> {
        gather_params(vec![self.weights.params(), self.hidden.params()])
    }
}

impl Stackable for Lstm {
    fn forward(&self, x: &Var) -> Var {
        let batch_size = x.shape()[0];

        let h_t = Var::with_data(Tensor::zeros([batch_size, self.hidden_size]));
        let c_t = Var::with_data(Tensor::zeros([batch_size, self.hidden_size]));

        let (h_all, h_t, _) = self.forward_with_states(x, &h_t, &c_t);

        h_all
    }
}
