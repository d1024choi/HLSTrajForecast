from utils.functions import *

ACTIVATION_DERIVATIVES = {
    F.elu: lambda x: torch.ones_like(x) * (x >= 0) + torch.exp(x) * (x < 0),
    torch.tanh: lambda x: 1 - torch.tanh(x) ** 2
}

class PlanarFlow(nn.Module):
    def __init__(self, D, activation=torch.tanh, normalize_u=True):
        super().__init__()

        self.normalize_u = normalize_u
        self.D = D
        self.w = nn.Parameter(torch.empty(D))
        self.b = nn.Parameter(torch.empty(1))
        self.u = nn.Parameter(torch.empty(D))
        self.activation = activation
        self.activation_derivative = ACTIVATION_DERIVATIVES[activation]

        nn.init.normal_(self.w)
        nn.init.normal_(self.u)
        nn.init.normal_(self.b)

    def forward(self, z: torch.Tensor):

        # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
        u_hat = self.u

        if self.normalize_u:
            wtu = (self.w @ self.u.t()).squeeze()
            m_wtu = - 1 + torch.log1p(wtu.exp())
            u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())


        lin = (z @ self.w + self.b).unsqueeze(1)  # shape: (B, 1)
        f = z + u_hat * self.activation(lin)  # shape: (B, D)
        phi = self.activation_derivative(lin) * self.w  # shape: (B, D)
        log_det = torch.log(torch.abs(1 + phi @ u_hat) + 1e-4) # shape: (B,)

        return f, log_det

def make_mlp(dim_list, bias_list, act_list, drop_list):

    num_layers = len(dim_list) - 1

    layers = []
    for i in range(num_layers):

        # layer info
        dim_in = dim_list[i]
        dim_out = dim_list[i+1]
        bias = bias_list[i]
        activation = act_list[i]
        drop_prob = drop_list[i]

        # linear layer
        layers.append(nn.Linear(dim_in, dim_out, bias=bias))

        # add activation
        if (activation == 'relu'):
            layers.append(nn.ReLU())
        elif (activation == 'sigmoid'):
            layers.append(nn.Sigmoid())
        elif (activation == 'tanh'):
            layers.append(nn.Tanh())

        # add dropout
        if (drop_prob > 0):
            layers.append(nn.Dropout(p=drop_prob))

    return nn.Sequential(*layers)

def init_hidden(num_layers, batch, h_dim):
    c = torch.zeros(num_layers, batch, h_dim).cuda()
    h = torch.zeros(num_layers, batch, h_dim).cuda()
    return (h, c)

class PoolingOperation(nn.Module):

    def __init__(self, feat_map_size, x_range, y_range):
        super(PoolingOperation, self).__init__()

        x_range_min = x_range[0]
        x_range_max = x_range[1]
        y_range_min = y_range[0]
        y_range_max = y_range[1]

        # map related params
        self.map_size = feat_map_size
        axis_range_y = y_range_max - y_range_min
        axis_range_x = x_range_max - x_range_min
        self.scale_y = float(self.map_size - 1) / axis_range_y
        self.scale_x = float(self.map_size - 1) / axis_range_x
        self.y_range_max = y_range_max
        self.x_range_max = x_range_max


    def clip(self, array):

        OOB = np.logical_or((array < 0), (array > self.map_size - 1))

        array[array < 0] = 0
        array[array > self.map_size - 1] = self.map_size - 1

        return array, OOB


    def op(self, x, feat_map):

        '''
        ROI-pooling feature vectors from feat map

        Inputs
        x : num_pos x 2
        feat_map : ch x h x w

        Outputs
        pooled_vecs : num_pos x ch
        '''

        ch_dim = feat_map.size(0)

        # from global coordinate system (ego-vehicle coordinate system) to feat_map index
        shift_c = np.trunc(self.y_range_max * self.scale_y)
        shift_r = np.trunc(self.x_range_max * self.scale_x)

        c_pels_f, c_oob = self.clip(-(toNP(x[:, 1]) * self.scale_y) + shift_c)
        r_pels_f, r_oob = self.clip(-(toNP(x[:, 0]) * self.scale_x) + shift_r)
        oob_pels = np.logical_or(c_oob, r_oob)

        # 4 neighboring positions
        '''
        -------|------|
        | lu   | ru   |
        |(cur.)|      |
        |------|------|
        | ld   | rd   |
        |      |      |
        |------|------|
        '''

        c_pels = c_pels_f.astype('int')
        r_pels = r_pels_f.astype('int')

        c_pels_lu = np.copy(c_pels) # j
        r_pels_lu = np.copy(r_pels) # i

        c_pels_ru, _ = self.clip(np.copy(c_pels + 1)) # j + 1
        r_pels_ru, _ = self.clip(np.copy(r_pels)) # i

        c_pels_ld, _ = self.clip(np.copy(c_pels)) # j
        r_pels_ld, _ = self.clip(np.copy(r_pels + 1)) # i + 1

        c_pels_rd, _ = self.clip(np.copy(c_pels + 1)) # j + 1
        r_pels_rd, _ = self.clip(np.copy(r_pels + 1)) # i + 1


        # feats (ch x r x c)
        feat_rd = feat_map[:, r_pels_rd.astype('int'), c_pels_rd.astype('int')]
        feat_lu = feat_map[:, r_pels_lu.astype('int'), c_pels_lu.astype('int')]
        feat_ru = feat_map[:, r_pels_ru.astype('int'), c_pels_ru.astype('int')]
        feat_ld = feat_map[:, r_pels_ld.astype('int'), c_pels_ld.astype('int')]

        # calc weights, debug 210409
        alpha = r_pels_f - r_pels_lu.astype('float')
        beta = c_pels_f - c_pels_lu.astype('float')

        dist_lu = (1 - alpha) * (1 - beta) + 1e-10
        dist_ru = (1 - alpha) * beta + 1e-10
        dist_ld = alpha * (1 - beta) + 1e-10
        dist_rd = alpha * beta

        # weighted sum of features, debug 210409
        w_lu = toTS(dist_lu, dtype=x).view(1, -1).repeat_interleave(ch_dim, dim=0)
        w_ru = toTS(dist_ru, dtype=x).view(1, -1).repeat_interleave(ch_dim, dim=0)
        w_ld = toTS(dist_ld, dtype=x).view(1, -1).repeat_interleave(ch_dim, dim=0)
        w_rd = toTS(dist_rd, dtype=x).view(1, -1).repeat_interleave(ch_dim, dim=0)

        pooled_vecs = (w_lu * feat_lu) + (w_ru * feat_ru) + (w_ld * feat_ld) + (w_rd * feat_rd)
        pooled_vecs[:, oob_pels] = 0

        return pooled_vecs.permute(1, 0)

class AdditiveAttention(nn.Module):
    # Implementing the attention module of Bahdanau et al. 2015 where
    # score(h_j, s_(i-1)) = v . tanh(W_1 h_j + W_2 s_(i-1))
    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None):
        super(AdditiveAttention, self).__init__()

        if internal_dim is None:
            internal_dim = int((encoder_hidden_state_dim + decoder_hidden_state_dim) / 2)

        self.w1 = nn.Linear(encoder_hidden_state_dim, internal_dim, bias=False)
        self.w2 = nn.Linear(decoder_hidden_state_dim, internal_dim, bias=False)
        self.v = nn.Linear(internal_dim, 1, bias=False)

    def score(self, encoder_state, decoder_state):
        # encoder_state is of shape (batch, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        # return value should be of shape (batch, 1)
        return self.v(torch.tanh(self.w1(encoder_state) + self.w2(decoder_state)))

    def forward(self, encoder_states, decoder_state):
        # encoder_states is of shape (batch, num_enc_states, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        score_vec = torch.cat([self.score(encoder_states[:, i], decoder_state) for i in range(encoder_states.shape[1])],
                              dim=1)
        # score_vec is of shape (batch, num_enc_states)

        attention_probs = torch.unsqueeze(F.softmax(score_vec, dim=1), dim=2)
        # attention_probs is of shape (batch, num_enc_states, 1)

        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        # final_context_vec is of shape (batch, enc_dim)

        return final_context_vec, attention_probs

class TemporallyBatchedAdditiveAttention(AdditiveAttention):
    # Implementing the attention module of Bahdanau et al. 2015 where
    # score(h_j, s_(i-1)) = v . tanh(W_1 h_j + W_2 s_(i-1))
    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None):
        super(TemporallyBatchedAdditiveAttention, self).__init__(encoder_hidden_state_dim,
                                                                 decoder_hidden_state_dim,
                                                                 internal_dim)

    def score(self, encoder_state, decoder_state):
        # encoder_state is of shape (batch, num_enc_states, max_time, enc_dim)
        # decoder_state is of shape (batch, max_time, dec_dim)
        # return value should be of shape (batch, num_enc_states, max_time, 1)
        return self.v(torch.tanh(self.w1(encoder_state) + torch.unsqueeze(self.w2(decoder_state), dim=1)))

    def forward(self, encoder_states, decoder_state):
        # encoder_states is of shape (batch, num_enc_states, max_time, enc_dim)
        # decoder_state is of shape (batch, max_time, dec_dim)
        score_vec = self.score(encoder_states, decoder_state)
        # score_vec is of shape (batch, num_enc_states, max_time, 1)

        attention_probs = F.softmax(score_vec, dim=1)
        # attention_probs is of shape (batch, num_enc_states, max_time, 1)

        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        # final_context_vec is of shape (batch, max_time, enc_dim)

        return final_context_vec, torch.squeeze(torch.transpose(attention_probs, 1, 2), dim=3)
