from models.base_modules import *

class Posterior(nn.Module):

    def __init__(self, args):
        super(Posterior, self).__init__()

        self.traj_enc_h_dim = args.traj_enc_h_dim
        self.lane_feat_dim = args.lane_feat_dim
        self.z_dim = args.z_dim

        input_dim = 2*self.lane_feat_dim + 3 * self.traj_enc_h_dim
        self.mean = make_mlp([input_dim, self.lane_feat_dim, self.z_dim], [True, True], ['relu', 'none'], [0, 0])
        self.logvar = make_mlp([input_dim, self.lane_feat_dim, self.z_dim], [True, True], ['relu', 'none'], [0, 0])


    def forward(self, x, y, target_lane_context, ngh_lane_context, ngh_context):

        '''
        x (past traj enc) : batch x dim
        y (future traj enc) : batch x dim
        target_lane_context/ngh_lane_context : batch x  dim
        ngh_context : batch x dim
        '''

        mean = self.mean(torch.cat((x, target_lane_context, ngh_lane_context, ngh_context, y), dim=1))
        log_var = self.logvar(torch.cat((x, target_lane_context, ngh_lane_context, ngh_context, y), dim=1))

        return mean, log_var


class Prior(nn.Module):

    def __init__(self, args):
        super(Prior, self).__init__()

        self.traj_enc_h_dim = args.traj_enc_h_dim
        self.lane_feat_dim = args.lane_feat_dim
        self.z_dim = args.z_dim

        input_dim = 2*self.lane_feat_dim + 2*self.traj_enc_h_dim
        self.mean = make_mlp([input_dim, self.lane_feat_dim, self.z_dim], [True, True], ['relu', 'none'], [0, 0])
        self.logvar = make_mlp([input_dim, self.lane_feat_dim, self.z_dim], [True, True], ['relu', 'none'], [0, 0])


    def forward(self, x, target_lane_context, ngh_lane_context, ngh_context):

        '''
        x (past traj enc) : batch x dim
        target_lane_context/ngh_lane_context : batch x  dim
        ngh_context : batch x dim
        '''

        mean = self.mean(torch.cat((x, target_lane_context, ngh_lane_context, ngh_context), dim=1))
        log_var = self.logvar(torch.cat((x, target_lane_context, ngh_lane_context, ngh_context), dim=1))

        return mean, log_var


class TrajEncoder(nn.Module):

    '''
    Encode past trajectory
    '''

    def __init__(self, args, input_dim=4, onnx=False, is_obs=True):
        super(TrajEncoder, self).__init__()

        self.input_dim = input_dim # (speed, heading, x, y)
        self.h_dim = args.traj_enc_h_dim
        self.emb_dim = args.traj_enc_h_dim
        self.num_layers = 1
        self.onnx = onnx
        self.max_num_agents = args.max_num_agents

        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.seq_len = self.obs_len if is_obs else self.pred_len

        self.pos_emb = make_mlp(dim_list=[self.input_dim, self.emb_dim], bias_list=[True], act_list=['relu'], drop_list=[0])
        self.encoder = nn.LSTM(self.emb_dim, self.h_dim, self.num_layers, dropout=0)


    def forward(self, in_traj_ego):

        """
        in_traj_ego : obs_len x num_total_agents x 4 (speed, heading x, y)
        """

        if (self.onnx == False):
            batch = in_traj_ego.size(1)
            chk_nan = torch.isnan(in_traj_ego)
            in_traj_ego[chk_nan] = 0
        else:
            batch = self.max_num_agents

        # position embedding
        in_traj_embedding = self.pos_emb(in_traj_ego.view(self.seq_len * batch, self.input_dim))
        in_traj_embedding = in_traj_embedding.view(self.seq_len, batch, self.emb_dim)

        # encoding by an LSTM
        state_tuple = init_hidden(self.num_layers, batch, self.h_dim)
        output, state = self.encoder(in_traj_embedding, state_tuple)
        final_h = state[0]

        return final_h


class LaneEncoder(nn.Module):

    def __init__(self, args):
        super(LaneEncoder, self).__init__()

        self.num_max_paths = args.num_max_paths
        self.path_resol = args.path_resol
        self.path_len = int(args.max_path_len_forward / self.path_resol)
        self.lane_feat_dim = args.lane_feat_dim

        self.lane_emb = make_mlp(dim_list=[2+2+1, args.lane_feat_dim], bias_list=[True], act_list=['relu'], drop_list=[0])
        self.encoder = nn.LSTM(args.lane_feat_dim, args.lane_feat_dim, 1, dropout=0)

    def calc_yaw_from_points(self, in_vec):

        '''
        in_vec : seq_len x 2
        '''

        seq_len = in_vec.size(0)

        vec1 = toNP(in_vec).reshape(seq_len, 2)
        vec2 = np.repeat(np.concatenate([np.ones(shape=(1, 1)), np.zeros(shape=(1, 1))], axis=1), seq_len, 0)

        x1 = vec1[:, 0]
        y1 = vec1[:, 1]
        x2 = vec2[:, 0]
        y2 = vec2[:, 1]

        dot = y1 * y2 + x1 * x2  # dot product
        det = y1 * x2 - x1 * y2  # determinant

        heading = np.arctan2(det, dot)  # -1x because of left side is POSITIVE

        return torch.from_numpy(heading).to(in_vec)

    def forward(self, agent_past_motion_context, possible_lanes):

        '''
        agent_past_motion_context : batch x dim
        scene_context_feature : batch x ch x h x w
        possible_lane : seq_len x (batch x num_max_lanes) x 2
        '''

        batch = agent_past_motion_context.size(0)

        possible_lanes_emb = []
        for a in range(batch):

            # possible lanes for the current agent (seq_len x num_max_paths x 2)
            candi_lanes = possible_lanes[:, a*self.num_max_paths:(a+1)*self.num_max_paths, :]

            # for all candidate lanes
            for l in range(self.num_max_paths):

                cur_lane = candi_lanes[:, l, :] # seq_len x 2

                # if exists, extract feature
                cur_lane_ext = torch.zeros(size=(cur_lane.size(0), 5)).to(agent_past_motion_context)
                if (torch.count_nonzero(torch.isnan(cur_lane))==0):
                    cur_lane_ext[:, :2] = cur_lane # position
                    cur_lane_ext[1:, 2:4] = cur_lane[1:, :] - cur_lane[:-1, :] # displacement
                    cur_lane_ext[:, 4] = self.calc_yaw_from_points(cur_lane).view(1, -1) # yaw

                # embedding (seq_len x dim)
                possible_lanes_emb.append(torch.unsqueeze(self.lane_emb(cur_lane_ext), dim=1))
        possible_lanes_emb = torch.cat(possible_lanes_emb, dim=1) # seq_len x (batch x num_max_lanes) x dim

        # path encoding
        state_tuple = init_hidden(1, batch * self.num_max_paths, self.lane_feat_dim)
        output, state = self.encoder(possible_lanes_emb, state_tuple)

        # reshape
        output = []
        for a in range(batch):
            lanes_for_cur_agent = state[0][0][a*self.num_max_paths:(a+1)*self.num_max_paths] # num_max_lanes x dim
            output.append(torch.unsqueeze(lanes_for_cur_agent, dim=0)) # 1 x num_max_lanes x dim

        return torch.cat(output, dim=0) # batch x num_max_lanes x dim


class V2I(nn.Module):

    def __init__(self, args):
        super(V2I, self).__init__()

        self.num_max_paths = args.num_max_paths
        self.lane_feat_dim = args.lane_feat_dim
        self.ngh_dist_thr = args.ngh_dist_thr
        self.batch = args.batch_size
        self.traj_enc_h_dim = args.traj_enc_h_dim

        # GNN
        self.GRU = nn.GRU(self.traj_enc_h_dim, self.traj_enc_h_dim, 1)
        self.message = make_mlp([2*self.traj_enc_h_dim+2, self.traj_enc_h_dim], [True], ['relu'], [0])

    def GNN(self, nodes_pos, nodes):

        '''
        nodes : num_nodes x dim (first node is the  target agent)
        nodes_pos : num_nodes x 2 (first node is the  target agent)
        '''

        num_nodes = nodes.size(0)


        pooled_vectors = []
        for a in range(num_nodes):

            cur_node_pos_repeat = nodes_pos[a].view(1, -1).repeat(num_nodes, 1)
            cur_node_repeat = nodes[a].view(1, -1).repeat(num_nodes, 1)

            message_in = torch.cat(((nodes_pos - cur_node_pos_repeat), cur_node_repeat, nodes), dim=1)
            message = self.message(message_in)

            if (a == 0):
                pooled_vec = torch.sum(message[1:], dim=0).view(1, self.traj_enc_h_dim)
            elif (a == num_nodes - 1):
                pooled_vec = torch.sum(message[:-1], dim=0).view(1, self.traj_enc_h_dim)
            else:
                pooled_vec = torch.sum(torch.cat((message[:a], message[a + 1:]), dim=0), dim=0).view(1, self.traj_enc_h_dim)
            pooled_vectors.append(pooled_vec)

        gru_in = torch.unsqueeze(torch.cat(pooled_vectors, dim=0), dim=0)
        gru_h = torch.unsqueeze(nodes, dim=0)
        O, h = self.GRU(gru_in, gru_h)

        return O[0]

    def forward(self, agent_pos, agent_context, ngh_pos, ngh_context, possible_lanes, lane_context, label, seq_start_end, valid_neighbor):

        '''
        agent_pos (ngh_pos) : batch x 2
        agent_context (ngh_context) : batch x dim
        possible_lanes : seq_len x (batch x num_max_paths) x 2
        lane_context : batch x num_max_paths x dim
        label : batch x num_max_paths
        seq_start_end : batch x 2
        is_valid_neighbor : batch
        '''

        batch, num_max_paths, _ = lane_context.size()
        seq_len = possible_lanes.size(0)

        lane_dicts = [[] for _ in range(batch)]
        for b in range(batch):

            # GT target lane for current agent
            try:
                best_lane_idx = np.argwhere(toNP(label[b, :]) == 1)[0][0]
            except:
                best_lane_idx = 0

            # candidate lanes for current agent
            candi_lanes = possible_lanes[:, b * num_max_paths:(b + 1) * num_max_paths, :] # seq_len x num_max_paths x 2

            for l in range(num_max_paths):

                # target lane flag
                is_best_lane = True if best_lane_idx == l else False

                # current lane and lane_feat
                cur_lane = candi_lanes[:, l, :] # seq_len x dim
                if (torch.count_nonzero(torch.isnan(cur_lane))!=0):
                    cur_lane = torch.zeros(size=(seq_len, 2)).to(agent_pos)
                cur_lane_feat = lane_context[b, l, :].view(1, -1) # 1 x dim


                # calculate neighboring agent context
                num_valid_neighbors = 0
                cur_nghs = []
                if (valid_neighbor[b] == True):
                    start = seq_start_end[b, 0]
                    end = seq_start_end[b, 1]
                    num_neighbors = end - start

                    cur_ngh_poses = ngh_pos[start:end, :].view(num_neighbors, -1)  # num_neighbors x 2
                    cur_ngh_contexts = ngh_context[start:end].view(num_neighbors, -1)  # num_neighbors x dim

                    # for all neighbors
                    for n in range(num_neighbors):
                        cur_ngh_pos = cur_ngh_poses[n].view(1, -1)
                        cur_ngh_context = cur_ngh_contexts[n].view(1, -1)

                        errors = toNP(cur_ngh_pos) - toNP(cur_lane)
                        dist = np.min(np.sqrt(np.sum(errors**2, axis=1)))

                        # neighbors close to a LP
                        if (dist < self.ngh_dist_thr):
                            cur_nghs.append([cur_ngh_pos, cur_ngh_context])
                            num_valid_neighbors += 1

                if (num_valid_neighbors > 0):
                    tngh_pos, tngh_context = [], []
                    for n in range(num_valid_neighbors):
                        tngh_pos.append(cur_nghs[n][0])
                        tngh_context.append(cur_nghs[n][1])
                    tngh_pos = torch.cat(tngh_pos, dim=0)
                    tngh_context = torch.cat(tngh_context, dim=0)

                    zero_pos = torch.zeros(size=(1, 2)).to(agent_context)
                    nodes_pos = torch.cat((zero_pos, tngh_pos), dim=0)
                    nodes = torch.cat((agent_context[b].view(1, -1), tngh_context), dim=0)
                    nodes = self.GNN(nodes_pos, nodes)
                    ngh_context_wrt_lane = torch.sum(nodes[1:].view(num_valid_neighbors, self.traj_enc_h_dim), dim=0).view(1, self.traj_enc_h_dim)
                else:
                    ngh_context_wrt_lane = torch.zeros(size=(1, self.traj_enc_h_dim)).to(agent_context)


                # lane dictionary
                lane_dictionary = dict({'b' : b,
                                        'l' : l,
                                        'is_best_lane' : is_best_lane,
                                        'lane' : cur_lane,
                                        'lane_feat' : cur_lane_feat,
                                        'neighbors' : cur_nghs,
                                        'ngh_context': ngh_context_wrt_lane})
                lane_dicts[b].append(lane_dictionary)

        # back to lane/ngh context tensors (batch x num_max_paths x dim)
        lane_context_recon = torch.zeros_like(lane_context)
        ngh_context_wrt_lane = torch.zeros(size=(self.batch, self.num_max_paths, self.traj_enc_h_dim)).to(agent_context)
        for b in range(self.batch):
            cur_lane_dicts = lane_dicts[b]
            for l in range(self.num_max_paths):
                cur_dict = cur_lane_dicts[l]
                bidx = cur_dict['b']
                lidx = cur_dict['l']
                lane_feat = cur_dict['lane_feat']
                ngh_context = cur_dict['ngh_context']

                lane_context_recon[bidx, lidx, :] += lane_feat.view(self.lane_feat_dim)
                ngh_context_wrt_lane[bidx, lidx, :] += ngh_context.view(self.traj_enc_h_dim)

        return lane_context_recon, ngh_context_wrt_lane


class VLI(nn.Module):

    def __init__(self, args):
        super(VLI, self).__init__()

        self.batch = args.batch_size
        self.num_max_paths = args.num_max_paths
        self.lane_feat_dim = args.lane_feat_dim

        self.att_op = AdditiveAttention(args.lane_feat_dim, args.traj_enc_h_dim)

    def forward(self, agent_context, lane_contexts):
        '''
        agent_context : batch x dim
        lane_contexts : batch x num_max_paths x dim
        lane_label : batch x num_max_paths
        ngh_contexts : batch x num_max_paths x dim

        '''

        batch = agent_context.size(0)
        att_context, scores = self.att_op(lane_contexts, agent_context)  # batch x dim, batch x num_max_paths x 1

        ngh_lane_context = []
        for l in range(self.num_max_paths):
            target_lane = lane_contexts[:, l, :]  # batch x dim
            att_context_sub = att_context - target_lane * scores[:, l].view(batch, 1).repeat(1, self.lane_feat_dim)
            ngh_lane_context.append(torch.unsqueeze(att_context_sub, dim=1))
        ngh_lane_context = torch.cat(ngh_lane_context, dim=1)

        return ngh_lane_context

    def return_ngh_lane_context(self, agent_context, lane_contexts, l):

        '''
        agent_context : batch x dim
        lane_contexts : batch x num_max_lanes x dim
        l : int

        '''

        batch = agent_context.size(0)
        target_lane = lane_contexts[:, l, :]  # batch x dim
        att_context, scores = self.att_op(lane_contexts, agent_context)  # batch x dim, batch x num_max_paths x 1
        ngh_lane_context = att_context - target_lane * scores[:, l].view(batch, 1).repeat(1, self.lane_feat_dim)

        return ngh_lane_context


class ModeSelectionNetwork(nn.Module):

    def __init__(self, args):
        super(ModeSelectionNetwork, self).__init__()

        self.batch = args.batch_size
        self.num_max_paths = args.num_max_paths
        self.lane_feat_dim = args.lane_feat_dim
        self.traj_enc_h_dim = args.traj_enc_h_dim

        self.att_op = AdditiveAttention(args.lane_feat_dim, args.traj_enc_h_dim)

        self.embedder = make_mlp([2*args.lane_feat_dim + 2*args.traj_enc_h_dim, args.lane_feat_dim], [True], ['relu'], [0])
        self.classifier = make_mlp([args.lane_feat_dim * args.num_max_paths, args.num_max_paths], [True], ['none'], [0])

    def forward(self, agent_context, lane_contexts, ngh_lane_context, lane_label, ngh_contexts, isTrain=True):

        '''
        agent_context : batch x dim
        lane_contexts : batch x num_max_paths x dim
        lane_label : batch x num_max_paths
        ngh_contexts : batch x num_max_paths x dim

        '''

        batch = agent_context.size(0)

        '''
        repeat_interleave : 0 0 0 1 1 1 2 2 2
        repeat            : 0 1 2 0 1 2 0 1 2
        
        tensor.repeat_interleave(num_vids_cur, dim=0)
        '''
        # agent context
        agent_context_repeat = agent_context.reshape(batch, 1, -1).repeat_interleave(self.num_max_paths, dim=1) #  batch x num_max_paths x dim
        context_cat = torch.cat((agent_context_repeat, lane_contexts, ngh_lane_context, ngh_contexts), dim=2) #  batch x num_max_paths x dim
        context_emb = self.embedder(context_cat.view(-1, 2*self.lane_feat_dim+2*self.traj_enc_h_dim)).view(batch, self.num_max_paths, self.lane_feat_dim)
        logits = self.classifier(context_emb.view(batch, -1))

        # the best-matched lane by gt-label
        best_lane_contexts = torch.zeros(size=(batch, self.lane_feat_dim)).to(lane_contexts)
        best_ngh_lane_contexts = torch.zeros(size=(batch, self.lane_feat_dim)).to(lane_contexts)
        best_ngh_contexts = torch.zeros(size=(batch, self.traj_enc_h_dim)).to(lane_contexts)
        if (isTrain):
            for b in range(batch):
                best_lane_idx = np.argwhere(toNP(lane_label[b, :]) == 1)[0][0]

                cur_lanes = lane_contexts[b] # num_max_paths x dim
                best_lane_contexts[b, :] += cur_lanes[best_lane_idx, :]

                cur_ngh_lanes = ngh_lane_context[b]
                best_ngh_lane_contexts[b, :] += cur_ngh_lanes[best_lane_idx, :]

                cur_nghs = ngh_contexts[b]
                best_ngh_contexts[b, :] += cur_nghs[best_lane_idx, :]

        return logits, best_lane_contexts, best_ngh_lane_contexts, best_ngh_contexts


class TrajDecoder(nn.Module):


    def __init__(self, args):
        super(TrajDecoder, self).__init__()

        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.best_k = args.best_k
        self.batch = args.batch_size

        self.input_dim = args.input_dim
        self.pos_emb_dim = args.pos_emb_dim # 16
        self.traj_enc_h_dim = args.traj_enc_h_dim # 64
        self.traj_dec_h_dim = args.traj_dec_h_dim
        self.lane_feat_dim = args.lane_feat_dim
        self.z_dim = args.z_dim

        self.pos_emb = make_mlp(dim_list=[self.input_dim, self.pos_emb_dim], bias_list=[True], act_list=['relu'], drop_list=[0])

        lstm_h_dim = self.pos_emb_dim+\
                     2*self.traj_enc_h_dim+\
                     2*self.lane_feat_dim+\
                     self.z_dim
        self.decoder_lstm = nn.LSTM(lstm_h_dim, self.traj_dec_h_dim, 1, dropout=0)

        # traj decoder
        self.decoder_mlp = make_mlp([self.traj_dec_h_dim, self.input_dim], [True], ['None'], [0])


    def forward(self, current_position, agent_motion_context, lane_context, ngh_lane_context, ngh_context, Z):

        '''
        current_position :  (best_k x batch) x dim
        agent_motion_context : (best_k x batch) x dim
        lane_context : (best_k x batch) x dim
        Z : (best_k x batch) x dim

        output : best_k x batch x seq_len x 2
        '''

        batch = current_position.size(0)

        pos_emb = self.pos_emb(current_position) # (best_k x batch) x dim

        input = torch.cat((pos_emb, agent_motion_context, lane_context, ngh_lane_context, ngh_context, Z), dim=1) # (best_k x batch) x dim
        state_tuple = init_hidden(1, batch, self.traj_dec_h_dim)

        future_trajectory, future_offsets = [], []
        for i in range(self.pred_len):

            output, state_tuple = self.decoder_lstm(torch.unsqueeze(input, dim=0), state_tuple)
            h, c = state_tuple # 1 x (best_k x batch) x dim

            # position calc.
            pos = self.decoder_mlp(h.view(batch, self.traj_dec_h_dim)) # (best_k x batch) x 2

            # offset calc.
            if (i == 0):
                future_offsets.append(torch.unsqueeze(pos, dim=1))  # (best_k x batch) x 1 x 2
            else:
                offset = torch.unsqueeze(pos, dim=1) - future_trajectory[-1]
                future_offsets.append(offset)

            # for the next iteration
            pos_emb = self.pos_emb(pos)
            input = torch.cat((pos_emb, agent_motion_context, lane_context, ngh_lane_context, ngh_context, Z), dim=1) # (best_k x batch) x dim

            # save
            future_trajectory.append(torch.unsqueeze(pos, dim=1))  # (best_k x batch) x 1 x 2

        # (best_k x batch) x seq_len x 2
        future_trajectory = torch.cat(future_trajectory, dim=1)
        future_offsets = torch.cat(future_offsets, dim=1)

        return future_trajectory, future_offsets


class Discriminator(nn.Module):


    def __init__(self, args):
        super(Discriminator, self).__init__()


        self.traj_enc_h_dim = args.traj_enc_h_dim
        self.lane_feat_dim = args.lane_feat_dim

        self.input_dim = 2 # (x, y)
        self.num_layers = 1

        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.seq_len = self.obs_len + self.pred_len

        self.pos_emb = make_mlp(dim_list=[2*self.input_dim, self.traj_enc_h_dim],
                                      bias_list=[True], act_list=['relu'], drop_list=[0])
        self.encoder = nn.LSTM(self.traj_enc_h_dim, self.traj_enc_h_dim, self.num_layers, dropout=0)

        input_dim = self.traj_enc_h_dim + self.lane_feat_dim
        # input_dim = self.traj_enc_h_dim
        self.discriminator = make_mlp([input_dim, 1], [True], ['None'], [0])


    def forward(self, future_traj, corr_lane_pos, lane_context):

        '''
        obs_traj or future_traj : seq_len x batch x dim
        corr_lane_pos : seq_len x batch x dim
        lane_context : batch x dim

        output : batch x 1
        '''

        diff = future_traj - corr_lane_pos
        cat = torch.cat((future_traj, diff), dim=2)
        seq_len, batch, input_dim = cat.size()

        sample_emb = self.pos_emb(cat.reshape(seq_len*batch, input_dim)).reshape(seq_len, batch, self.traj_enc_h_dim)
        _, state = self.encoder(sample_emb)
        sample_hidden = state[0][0]

        input_to_dis = torch.cat((sample_hidden, lane_context), dim=1)
        out_scr = self.discriminator(input_to_dis)

        return out_scr


class HLS(nn.Module):

    def __init__(self, args):
        super(HLS, self).__init__()

        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.num_max_paths = args.num_max_paths
        self.lane_feat_dim = args.lane_feat_dim
        self.best_k = args.best_k
        self.batch = args.batch_size
        self.z_dim = args.z_dim
        self.traj_enc_h_dim = args.traj_enc_h_dim
        self.gan_prior_prob = args.gan_prior_prob

        # define past traj encoder
        self.Past_Traj_Enc = TrajEncoder(args=args, input_dim=4, is_obs=True)
        self.Future_Traj_Enc = TrajEncoder(args=args, input_dim=4, is_obs=False)

        # define CVAE encoder
        self.Posterior = Posterior(args=args)
        self.Prior = Prior(args=args)

        # scene context extractor
        self.LE = LaneEncoder(args=args)

        # context preprocessing
        self.V2I = V2I(args=args)
        self.VLI = VLI(args=args)

        # best lane selection
        self.MSN = ModeSelectionNetwork(args=args)

        # trajectory decoder
        self.Traj_Dec = TrajDecoder(args=args)

        print(">> Model is loaded from {%s} " % os.path.basename(__file__))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, feature_map, seq_start_end, valid_neighbor,
                possible_lanes, lane_label, best_k=None):

        '''
        obs_traj : seq_len x batch x 4 (speed, heading, x, y)
        future_traj : seq_len x batch x 4 (speed, heading, x, y)
        obs_traj_ngh : seq_len x num_total_neighbors x 4 (speed, heading, x, y)
        future_traj_ngh : seq_len x num_total_neighbors x 4 (speed, heading, x, y)
        feature_map : batch x ch x h x w
        seq_start_end : batch x 2
        valid_neighbor : batch
        possible_lane : num_pos_f x (max_num_paths x batch) x 2
        lane_label : batch x num_max_paths

        pred_trajs : best_k x batch x seq_len x 2
        '''

        # motion context
        agent_past_motion_context = self.Past_Traj_Enc(obs_traj)[0]
        ngh_past_motion_context = self.Past_Traj_Enc(obs_traj_ngh)[0]
        agent_future_motion_context = self.Future_Traj_Enc(future_traj)[0]

        # lane conext
        lane_context = self.LE(agent_past_motion_context, possible_lanes) # batch x num_max_paths x dim

        # vehicle-vehicle interaction
        lane_context_recon, ngh_context_wrt_lane = self.V2I(obs_traj[-1, :, 2:4], agent_past_motion_context, obs_traj_ngh[-1, :, 2:4],
                             ngh_past_motion_context, possible_lanes, lane_context, lane_label, seq_start_end, valid_neighbor)


        ngh_lane_context = self.VLI(agent_past_motion_context, lane_context)

        # predict best lane index
        logits, best_lane_context, best_ngh_lane_context, best_ngh_context = \
            self.MSN(agent_past_motion_context, lane_context, ngh_lane_context, lane_label, ngh_context_wrt_lane)

        # CAVE encoding : batch x dim
        mean0, log_var0 = self.Posterior(agent_past_motion_context, agent_future_motion_context, best_lane_context, best_ngh_lane_context, best_ngh_context)
        mean1, log_var1 = self.Prior(agent_past_motion_context, best_lane_context, best_ngh_lane_context, best_ngh_context)

        # predict
        if (best_k is None):

            # sample Z : (best_k x batch) x dim
            Z = [self.reparameterize(mean0, log_var0) for _ in range(self.best_k)]
            Z = torch.cat(Z, dim=0)

            start_pos = torch.zeros(size=(self.batch*self.best_k, 2)).to(obs_traj)
            pred_trajs, pred_offsets = self.Traj_Dec(start_pos,
                                       agent_past_motion_context.repeat(self.best_k, 1),
                                       best_lane_context.repeat(self.best_k, 1),
                                       best_ngh_lane_context.repeat(self.best_k, 1),
                                       best_ngh_context.repeat(self.best_k, 1),
                                       Z)

            # reshpae to best_k x batch x seq_len x 2
            pred_trajs_reshape, pred_offsets_reshape = [], []
            for k in range(self.best_k):
                pred_trajs_reshape.append(torch.unsqueeze(pred_trajs[k*self.batch:(k+1)*self.batch], dim=0))
                pred_offsets_reshape.append(torch.unsqueeze(pred_offsets[k * self.batch:(k + 1) * self.batch], dim=0))

            return torch.cat(pred_trajs_reshape, dim=0), torch.cat(pred_offsets_reshape, dim=0), mean0, log_var0, mean1, log_var1, logits, best_lane_context

        else:

            # TODO : prior or posterior
            # sample Z : (best_k x batch) x dim
            if (self.gan_prior_prob < np.random.rand(1)):
                Z = [self.reparameterize(mean1, log_var1) for _ in range(best_k)]
            else:
                Z = [self.reparameterize(mean0, log_var0) for _ in range(best_k)]
            Z = torch.cat(Z, dim=0)

            start_pos = torch.zeros(size=(self.batch * best_k, 2)).to(obs_traj)
            pred_trajs, _ = self.Traj_Dec(start_pos,
                                          agent_past_motion_context.repeat(best_k, 1),
                                          best_lane_context.repeat(best_k, 1),
                                          best_ngh_lane_context.repeat(best_k, 1),
                                          best_ngh_context.repeat(best_k, 1),
                                          Z)

            # reshpae to best_k x batch x seq_len x 2
            pred_trajs_reshape = []
            for k in range(best_k):
                pred_trajs_reshape.append(torch.unsqueeze(pred_trajs[k * self.batch:(k + 1) * self.batch], dim=0))

            return torch.cat(pred_trajs_reshape, dim=0), best_lane_context

    def inference(self, obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, feature_map, seq_start_end, valid_neighbor,
                possible_lanes, lane_label):

        '''
        obs_traj : seq_len x batch x 4 (speed, heading, x, y)
        future_traj : seq_len x batch x 4 (speed, heading, x, y)
        obs_traj_ngh : seq_len x num_total_neighbors x 2
        future_traj_ngh : seq_len x num_total_neighbors x 2
        feature_map : batch x ch x h x w
        seq_start_end : batch x 2
        valid_neighbor : batch
        possible_lane : num_pos_f x (max_num_paths x batch) x 2
        lane_label : batch x num_max_paths

        pred_trajs : best_k x batch x seq_len x 2
        '''

        # agent motion context
        agent_past_motion_context = self.Past_Traj_Enc(obs_traj)[0]
        ngh_past_motion_context = self.Past_Traj_Enc(obs_traj_ngh)[0]

        # lane and neighborhood fusion context
        lane_context = self.LE(agent_past_motion_context, possible_lanes) # batch x num_max_paths x dim

        # lane preprocessing
        lane_context_recon, ngh_context_wrt_lane = self.V2I(obs_traj[-1, :, 2:4], agent_past_motion_context, obs_traj_ngh[-1, :, 2:4],
                             ngh_past_motion_context, possible_lanes, lane_context, lane_label, seq_start_end, valid_neighbor)

        ngh_lane_context = self.VLI(agent_past_motion_context, lane_context)

        # predict best lane index
        logits, _, _, _ = self.MSN(agent_past_motion_context, lane_context_recon, ngh_lane_context, lane_label, ngh_context_wrt_lane, isTrain=False)


        # predict best_k trajectories for each lane
        start_pos = torch.zeros(size=(self.batch*self.best_k, 2)).to(obs_traj)
        pred_trajs_lane = []
        for l in range(self.num_max_paths):

            # for the l-th lane
            target_lane_context = lane_context_recon[:, l, :] # batch x dim
            ngh_lane_context = self.VLI.return_ngh_lane_context(agent_past_motion_context, lane_context_recon, l)
            ngh_context = ngh_context_wrt_lane[:, l, :] # batch x dim

            # sample Z : (best_k x batch) x dim
            mean1, log_var1 = self.Prior(agent_past_motion_context, target_lane_context, ngh_lane_context, ngh_context)
            Z = [self.reparameterize(mean1, log_var1) for _ in range(self.best_k)]
            Z = torch.cat(Z, dim=0)

            # predict best_k trajectories
            pred_trajs, _ = self.Traj_Dec(start_pos,
                                       agent_past_motion_context.repeat(self.best_k, 1),
                                       target_lane_context.repeat(self.best_k, 1),
                                       ngh_lane_context.repeat(self.best_k, 1),
                                       ngh_context.repeat(self.best_k, 1),
                                       Z) # (best_k x batch) x seq_len x 2

            # reshape to (best_k x batch x seq_len x 2)
            pred_trajs_reshape = []
            for k in range(self.best_k):
                pred_trajs_reshape.append(torch.unsqueeze(pred_trajs[k*self.batch:(k+1)*self.batch], dim=0))
            pred_trajs_reshape = torch.cat(pred_trajs_reshape, dim=0)
            pred_trajs_lane.append(torch.unsqueeze(pred_trajs_reshape, dim=0))

        # reshape to (num_max_paths x best_k x batch x seq_len x 2)
        pred_trajs_lane = torch.cat(pred_trajs_lane, dim=0)


        # calc the number of predictions for each lane according to logtis
        prob = toNP(torch.softmax(logits, dim=1))

        # debug ---
        # prob = np.zeros_like(prob)
        # prob[:, 0] = 1
        # debug ---

        num_preds = np.round((prob * self.best_k)).astype('int')

        # select predictions
        pred_trajs = []
        for b in range(self.batch):

            # the number of predictions for each lane
            cur_prob = prob[b]
            cur_num_preds = num_preds[b]
            if (np.sum(cur_num_preds) < self.best_k):
                num_adds = self.best_k - np.sum(cur_num_preds)
                cur_num_preds[np.argmax(cur_prob)] += num_adds
            elif (np.sum(cur_num_preds) > self.best_k):
                num_adds = np.sum(cur_num_preds) - self.best_k
                cnt = 0
                while (num_adds > 0):
                    for i in range(num_adds):
                        cur_num_preds[random.randint(0, self.num_max_paths)-1] -= 1
                        chk = cur_num_preds < 0
                        cur_num_preds[chk] = 0
                    num_adds = np.sum(cur_num_preds) - self.best_k
                    cnt+=1
                    if (cnt>100):
                        break

            # select according to the number of predictions
            cur_pred_trajs = []
            for _, lidx in enumerate(np.argsort(prob[b])[::-1]):
                k = cur_num_preds[lidx]
                if (k < 1):
                    continue
                cur_pred_trajs.append(pred_trajs_lane[lidx, :k, b, :, :].view(k, 1, self.pred_len, 2)) # k x 1 seq_len x 2
            pred_trajs.append(torch.cat(cur_pred_trajs, dim=0))

        return torch.cat(pred_trajs, dim=1)
