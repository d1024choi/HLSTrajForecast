from utils.functions import *
from utils.loss import *
from models.HLS import HLS, Discriminator

class Solver:

    def __init__(self, args, num_train_scenes, dtype):

        # training setting
        self.args = args
        self.dtype = dtype
        self.best_k = args.best_k
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.kappa = args.kappa
        self.batch_size = args.batch_size
        self.grad_clip = args.grad_clip
        self.n_cycle = args.n_cycle
        self.warmup_ratio = args.warmup_ratio
        self.num_max_paths = args.num_max_paths
        self.is_train_dis = args.is_train_dis

        self.num_batches = int(num_train_scenes / args.batch_size)
        self.total_num_iteration = args.num_epochs * self.num_batches

        folder_name = args.dataset_type + '_' + args.model_name + '_model' + str(args.exp_id)
        self.save_dir = os.path.join('./saved_models/', folder_name)

        # training monitoring
        self.iter = 0
        self.l2_losses = 0
        self.kld_losses = 0
        self.bce_losses = 0
        self.g_losses = 0
        self.d_losses = 0
        self.prev_ADE = 1e5
        self.cur_lr = args.learning_rate
        self.min_lr = args.min_learning_rate

        # define models
        self.model = HLS(args)
        self.model.type(dtype)

        self.dis = Discriminator(args)
        self.dis.type(dtype)

        # define optimizer
        if (self.is_train_dis == 1):
            self.opt = optim.Adam(list(self.dis.parameters()) + list(self.model.parameters()), lr=args.learning_rate)
        else:
            self.opt = optim.Adam(self.model.parameters(), lr=args.learning_rate)

        # training schedule
        self.apply_kld_scheduling = args.apply_cyclic_schedule
        self.apply_lr_scheduling = args.apply_lr_scheduling

        self.kld_weight_scheduler = self.create_kld_weight_scheduler()
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.9999)

        # load network params
        if (args.load_pretrained == 1):
            ckp_idx = save_read_latest_checkpoint_num(os.path.join(self.save_dir), 0, isSave=False)
            _ = self.load_pretrained_network_params(ckp_idx)

        print(">> Optimizer is loaded from {%s} " % os.path.basename(__file__))

    def mode_selection(self, isTrain=True):
        if (isTrain):
            self.model.train()
        else:
            self.model.eval()

    def init_loss_tracker(self):
        self.l2_losses = 0
        self.kld_losses = 0
        self.bce_losses = 0
        self.g_losses = 0
        self.d_losses = 0

    def normalize_loss_tracker(self):
        self.l2_losses /= self.num_batches
        self.kld_losses /= self.num_batches
        self.bce_losses /= self.num_batches
        self.g_losses /= self.num_batches
        self.d_losses /= self.num_batches

    def create_kld_weight_scheduler(self):
        scheduler = frange_cycle_linear(self.total_num_iteration, n_cycle=self.n_cycle, ratio=self.warmup_ratio)

        if (self.apply_kld_scheduling == 1):
            return scheduler
        else:
            return np.ones_like(scheduler)

    def learning_rate_step(self):
        self.lr_scheduler.step()
        for g in self.opt.param_groups:
            if (g['lr'] < self.min_lr):
                g['lr'] = self.min_lr
        self.cur_lr = self.opt.param_groups[0]['lr']

    def load_pretrained_network_params(self, ckp_idx, isTrain=False):

        file_name = self.save_dir + '/saved_chk_point_%d.pt' % ckp_idx
        checkpoint = torch.load(file_name)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_state_dict(checkpoint['scratch_state_dict'])
        # self.opt.load_state_dict(checkpoint['opt'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.iter = checkpoint['iter']
        self.prev_ADE = checkpoint['ADE']
        print('>> trained parameters are loaded from {%s}' % file_name)
        print(">> current training settings ...")
        print("   . iteration : %.d" % (self.iter))
        print("   . prev_ADE : %.4f" % (self.prev_ADE))

        return ckp_idx

    def save_trained_network_params(self, e):

        # save trained model
        _ = save_read_latest_checkpoint_num(os.path.join(self.save_dir), e, isSave=True)
        file_name = self.save_dir + '/saved_chk_point_%d.pt' % e
        check_point = {
            'epoch': e,
            # 'model_state_dict': self.model.state_dict(),
            'scratch_state_dict': self.model.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'opt': self.opt.state_dict(),
            'ADE': self.prev_ADE,
            'iter': self.iter}
        torch.save(check_point, file_name)
        print(">> current network is saved ...")
        remove_past_checkpoint(os.path.join('./', self.save_dir), max_num=self.args.max_num_chkpts)

    def print_status(self, e, time_left):
        print("[E %02d, %.2f hrs left] l2: %.4f, kld: %.4f, bce: %.4f, gan: %.4f/%.4f (beta_scheduler: %.2f, cur_lr : %.8f)"
              % (e, time_left, self.l2_losses, self.kld_losses, self.bce_losses, self.g_losses, self.d_losses, self.kld_weight_scheduler[self.iter], self.cur_lr))

    # ------------------------
    # Training
    # ------------------------
    def train(self, batch):

        # Generation step
        self.generation_step(batch)

        # Discrimination step
        if (self.is_train_dis == 1):
            self.discrimination_step(batch)

        if (self.apply_lr_scheduling == 1):
            self.learning_rate_step()

        # increase iteration number
        self.iter += 1

    def generation_step(self, batch):

        '''
        obs_traj : seq_len x batch x 4 (speed, heading, x, y)
        future_traj : seq_len x batch x 4 (speed, heading, x, y)
        obs_traj_ngh : seq_len x num_total_neighbors x 4 (speed, heading, x, y)
        future_traj_ngh : seq_len x num_total_neighbors x 4 (speed, heading, x, y)
        map : batch x dim x h x w
        seq_start_end : batch x 2
        valid_neighbor : batch
        possible_lanes : seq_len x (batch x num_max_paths) x 2
        '''

        # read batch data
        obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, seq_start_end, valid_neighbor, \
        possible_lane, lane_label = batch

        # process map data
        feat_map = 0

        # predict future trajectory
        pred_trajs, pred_offsets, means0, log_vars0, means1, log_vars1, logits, lane_context \
                                                            = self.model(obs_traj[:, :, :4].cuda(),
                                                                           future_traj[:, :, :4].cuda(),
                                                                           obs_traj_ngh[:, :, :4].cuda(),
                                                                           future_traj_ngh[:, :, :4].cuda(),
                                                                           feat_map,
                                                                           seq_start_end,
                                                                           valid_neighbor,
                                                                           possible_lane.cuda(),
                                                                           lane_label.cuda())
        pred_trajs = pred_trajs.permute(0, 2, 1, 3).contiguous()


        # l2 loss
        l2 = torch.zeros(1).to(obs_traj.cuda())
        for i in range(self.best_k):
            l2 += l2_loss(pred_trajs[i], future_traj[:, :, 2:4].cuda())
        l2 = l2 / float(self.best_k)
        self.l2_losses += l2.item()

        # kld loss
        kld = kld_loss(means0, log_vars0, means1, log_vars1)
        self.kld_losses += kld.item()
        if (self.iter > len(self.kld_weight_scheduler) - 1):
            self.iter = len(self.kld_weight_scheduler) - 1

        # bce loss
        bce = cross_entropy_loss(logits, lane_label.cuda())
        self.bce_losses += bce.item()

        # final loss
        loss = l2 + (self.alpha * bce) + \
               (self.beta * self.kld_weight_scheduler[self.iter] * kld)

        # back-propagation
        self.opt.zero_grad()
        loss.backward(retain_graph=True)
        if self.grad_clip > 0.0:
            nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
        self.opt.step()

    def find_corr_lane_position(self, traj, lane):

        '''
        traj : seq_len x 2
        lane : seq_len x 2
        '''

        seq_len = traj.shape[0]
        if (np.count_nonzero(np.isnan(lane)) == 0):
            corr_lane_pos = []
            for t in range(seq_len):
                cur_pos = traj[t, :].reshape(1, 2)
                error = np.sum(np.abs(lane - cur_pos), axis=1)
                minidx = np.argmin(error)
                corr_pos = lane[minidx, :].reshape(1, 2)
                corr_lane_pos.append(corr_pos)
            return np.concatenate(corr_lane_pos, axis=0)

        else:
            return np.copy(traj)


    def discrimination_step(self, batch):

        '''
        obs_traj : seq_len x batch x 4 (speed, heading, x, y)
        future_traj : seq_len x batch x 4 (speed, heading, x, y)
        obs_traj_ngh : seq_len x num_total_neighbors x 4 (speed, heading, x, y)
        future_traj_ngh : seq_len x num_total_neighbors x 4 (speed, heading, x, y)
        map : batch x dim x h x w
        seq_start_end : batch x 2
        valid_neighbor : batch
        possible_lanes : seq_len x (batch x num_max_paths) x 2
        '''

        # read batch data
        obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, seq_start_end, valid_neighbor, \
        possible_lane, lane_label = batch


        # find corresponding lane positions for GT future trajectories
        seq_len, batch, _ = future_traj.size()
        future_traj_np = toNP(future_traj[:, :, 2:4])
        ref_lanes, corr_lane_pos_gt = [], []
        for b in range(batch):
            idx = np.argwhere(toNP(lane_label[b, :]) == 1)[0][0]
            cur_lanes = possible_lane[:, b*self.num_max_paths:(b+1)*self.num_max_paths, :]

            ref_lane = toNP(cur_lanes[:, idx, :])
            ref_lanes.append(ref_lane)

            corr_pos = self.find_corr_lane_position(future_traj_np[:, b, :], ref_lane).reshape(seq_len, 1, 2)
            corr_lane_pos_gt.append(corr_pos)
        corr_lane_pos_gt = np.concatenate(corr_lane_pos_gt, axis=1)
        corr_lane_pos_gt = torch.from_numpy(corr_lane_pos_gt).to(obs_traj)


        # process map data
        feat_map = 0

        # predict future trajectory
        pred_trajs, lane_context = self.model(obs_traj[:, :, :4].cuda(),
                                                future_traj[:, :, :4].cuda(),
                                                obs_traj_ngh[:, :, :4].cuda(),
                                                future_traj_ngh[:, :, :4].cuda(),
                                                feat_map,
                                                seq_start_end,
                                                valid_neighbor,
                                                possible_lane.cuda(),
                                                lane_label.cuda(),
                                                best_k=1)
        pred_trajs = pred_trajs.permute(0, 2, 1, 3).contiguous()


        # find corresponding lane positions for predicted future trajectories
        pred_trajs_np = toNP(pred_trajs)
        corr_lane_pos = []
        for b in range(batch):
            corr_pos = self.find_corr_lane_position(pred_trajs_np[0][:, b, :], ref_lanes[b]).reshape(seq_len, 1, 2)
            corr_lane_pos.append(corr_pos)
        corr_lane_pos = np.concatenate(corr_lane_pos, axis=1)
        corr_lane_pos = torch.from_numpy(corr_lane_pos).to(obs_traj)


        # d-loss
        scr_real = self.dis(future_traj[:, :, 2:4].cuda(), corr_lane_pos_gt.cuda(), lane_context)
        scr_fake_for_D = self.dis(pred_trajs[0].detach(), corr_lane_pos.cuda(), lane_context)
        d_loss = gan_d_loss(scr_real, scr_fake_for_D)
        self.d_losses += d_loss.item()

        # g loss
        scr_fake_for_G = self.dis(pred_trajs[0], corr_lane_pos.cuda(), lane_context)
        g_loss = gan_g_loss(scr_fake_for_G)
        self.g_losses += g_loss.item()

        # final loss
        loss = d_loss + (self.gamma * g_loss)

        # back-propagation
        self.opt.zero_grad()
        loss.backward()
        if self.grad_clip > 0.0:
            nn.utils.clip_grad_value_(self.dis.parameters(), self.grad_clip)
        self.opt.step()


    # ------------------------
    # Validation
    # ------------------------
    def eval(self, data_loader, e):

        # set to evaluation mode
        self.mode_selection(isTrain=False)

        ADE = []
        num_samples = int(len(data_loader.valid_data) / self.args.batch_size) * self.args.batch_size
        for b in range(0, num_samples, self.args.batch_size):

            '''
            obs_traj : seq_len x batch x 4 (speed, heading, x, y)
            future_traj : seq_len x batch x 4 (speed, heading, x, y)
            obs_traj_ngh : seq_len x num_total_neighbors x 4 (speed, heading, x, y)
            future_traj_ngh : seq_len x num_total_neighbors x 4 (speed, heading, x, y)
            map : batch x dim x h x w
            seq_start_end : batch x 2
            valid_neighbor : batch
            possible_lanes : seq_len x (batch x num_max_paths) x 2
            '''

            obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, num_neighbors, valid_neighbor, \
            possible_lanes, lane_label = [], [], [], [], [], [], [], [], []

            for i in range(b, b+self.args.batch_size):

                # data preparation
                _obs_traj, _future_traj, _obs_traj_ngh, _future_traj_ngh, _map, _num_neighbors, _valid_neighbor, \
                _possible_lanes, _lane_label = data_loader.next_sample(i, mode='valid')

                # to tensor
                _obs_traj = torch.from_numpy(_obs_traj).type(self.dtype)
                _future_traj = torch.from_numpy(_future_traj).type(self.dtype)
                _obs_traj_ngh = torch.from_numpy(_obs_traj_ngh).type(self.dtype)
                _future_traj_ngh = torch.from_numpy(_future_traj_ngh).type(self.dtype)
                _map = torch.from_numpy(_map).permute(2, 0, 1).type(self.dtype)
                _map = torch.unsqueeze(_map, dim=0)
                _possible_lanes = torch.from_numpy(_possible_lanes).type(self.dtype)
                _lane_label = torch.from_numpy(_lane_label).type(self.dtype)

                obs_traj.append(_obs_traj)
                future_traj.append(_future_traj)
                obs_traj_ngh.append(_obs_traj_ngh)
                future_traj_ngh.append(_future_traj_ngh)
                map.append(_map)
                num_neighbors.append(_num_neighbors)
                valid_neighbor.append(_valid_neighbor)
                possible_lanes.append(_possible_lanes)
                lane_label.append(_lane_label)

            # concat
            _len = [objs for objs in num_neighbors]
            cum_start_idx = [0] + np.cumsum(_len).tolist()
            seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

            obs_traj = torch.cat(obs_traj, dim=1)
            future_traj = torch.cat(future_traj, dim=1)
            obs_traj_ngh = torch.cat(obs_traj_ngh, dim=1)
            future_traj_ngh = torch.cat(future_traj_ngh, dim=1)
            map = torch.cat(map, dim=0)
            seq_start_end = torch.LongTensor(seq_start_end)
            valid_neighbor = np.array(valid_neighbor)
            possible_lanes = torch.cat(possible_lanes, dim=1)
            lane_label = torch.cat(lane_label, dim=0)

            # process map data
            feat_map = 0

            # predict future trajectory
            pred_trajs = self.model.inference(obs_traj[:, :, :4].cuda(),
                                                future_traj[:, :, :4].cuda(),
                                                obs_traj_ngh[:, :, :4].cuda(),
                                                future_traj_ngh[:, :, :4].cuda(),
                                                feat_map,
                                                seq_start_end,
                                                valid_neighbor,
                                                possible_lanes,
                                                lane_label).permute(0, 2, 1, 3)

            pred_trajs = toNP(pred_trajs)
            for k in range(self.best_k):
                err = np.sqrt(np.sum((pred_trajs[k] - toNP(future_traj[:, :, 2:4]))**2, axis=2))
                ADE.append(err)

            print_current_valid_progress(b, num_samples-self.args.batch_size)
        print(">> evaluation results are created .. {ADE:%.4f}" % np.mean(ADE))


        if (self.prev_ADE > np.mean(ADE)):
            self.prev_ADE = np.mean(ADE)
            self.save_trained_network_params(e)

    def padd(self, tensor, num_padd, dim=1):

        num_dim = len(list(tensor.size()))

        if (num_dim == 3):
            d0, d1, d2 = tensor.size()
            if (dim == 0):
                padd = torch.zeros(size=(num_padd, d1, d2)).to(tensor)
            elif (dim==1):
                padd = torch.zeros(size=(d0, num_padd, d2)).to(tensor)
            else:
                padd = torch.zeros(size=(d0, d1, num_padd)).to(tensor)

        elif (num_dim == 2):
            d0, d1 = tensor.size()
            if (dim == 0):
                padd = torch.zeros(size=(num_padd, d1)).to(tensor)
            elif (dim==1):
                padd = torch.zeros(size=(d0, num_padd)).to(tensor)

        else:
            sys.exit('dim %d is outside of the tensor dimension' % dim)


        return torch.cat((tensor, padd), dim=dim)

    # ------------------------
    # Testing
    # ------------------------
    def test(self, data, dtype, best_k):

        '''
        obs_traj : seq_len x batch x 4 (speed, heading, x, y)
        future_traj : seq_len x batch x 4 (speed, heading, x, y)
        obs_traj_ngh : seq_len x num_total_neighbors x 4 (speed, heading, x, y)
        future_traj_ngh : seq_len x num_total_neighbors x 4 (speed, heading, x, y)
        map : batch x dim x h x w
        seq_start_end : batch x 2
        valid_neighbor : batch
        possible_lanes : seq_len x (batch x num_max_paths) x 2
        '''

        _obs_traj, _future_traj, _obs_traj_ngh, _future_traj_ngh, _map, _seq_start_end, _valid_neighbor, \
        _possible_lanes, _lane_label, _agent_ids, _agent_samples, _scene = data

        pred_trajs = []
        num_agents = _obs_traj.size(1)

        for start in range(0, num_agents, self.batch_size):

            end = start+self.batch_size
            if (end > num_agents):
                end = num_agents

            obs_traj = _obs_traj[:, start:end, :4]
            future_traj = _future_traj[:, start:end, :4]
            possible_lanes = _possible_lanes[:, start*self.num_max_paths:end*self.num_max_paths, :]
            valid_neighbor = _valid_neighbor[start:end]
            lane_label = _lane_label[start:end]

            seq_start_end_copy = _seq_start_end[start:end]
            obs_traj_ngh, future_traj_ngh, seq_start_end = [], [], []

            pivot = seq_start_end_copy[0, 0]
            for i in range(seq_start_end_copy.shape[0]):
                _start = seq_start_end_copy[i, 0].item()
                _end = seq_start_end_copy[i, 1].item()

                seq_start_end.append(np.array([_start-pivot, _end-pivot]).reshape(1, 2))
                obs_traj_ngh.append(_obs_traj_ngh[:, _start:_end, :4])
                future_traj_ngh.append(_future_traj_ngh[:, _start:_end, :4])

            seq_start_end = np.concatenate(seq_start_end, axis=0)
            obs_traj_ngh = torch.cat(obs_traj_ngh, dim=1)
            future_traj_ngh = torch.cat(future_traj_ngh, dim=1)

            remain = end - start
            num_padd = self.batch_size - (end-start)
            if (num_padd > 0):
                obs_traj = self.padd(obs_traj, num_padd)
                future_traj = self.padd(future_traj, num_padd)
                obs_traj_ngh = self.padd(obs_traj_ngh, num_padd)
                future_traj_ngh = self.padd(future_traj_ngh, num_padd)
                possible_lanes = self.padd(possible_lanes, num_padd*self.num_max_paths)
                lane_label = self.padd(lane_label, num_padd, dim=0)


                for i in range(num_padd):
                    seq_start_end = np.concatenate([seq_start_end, seq_start_end[-1].reshape(1, 2)], axis=0)
                    valid_neighbor = np.concatenate([valid_neighbor, np.array([False])])

            # process map data
            feat_map = 0

            # predict future trajectory
            pred_trajs_mini = self.model.inference(obs_traj[:, :, :4].cuda(),
                                           future_traj[:, :, :4].cuda(),
                                           obs_traj_ngh[:, :, :4].cuda(),
                                           future_traj_ngh[:, :, :4].cuda(),
                                           feat_map,
                                           torch.from_numpy(seq_start_end),
                                           valid_neighbor,
                                           possible_lanes.cuda(),
                                           lane_label.cuda()).permute(0, 2, 1, 3)

            pred_trajs.append(toNP(pred_trajs_mini[:, :, :remain, :]))

        pred_trajs = np.concatenate(pred_trajs, axis=2)

        return toNP(_obs_traj[:, :, 2:]), toNP(_future_traj[:, :, 2:]), pred_trajs, _agent_ids, \
               (_agent_samples, _scene), True








