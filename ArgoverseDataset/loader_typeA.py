from utils.functions import read_config
from ArgoverseDataset.visualization import *
from torch.utils.data import Dataset
from ArgoverseDataset.scene import AgentCentricScene
from ArgoverseDataset.argoverse.map_representation.map_api import ArgoverseMap
from ArgoverseDataset.preprocess import DatasetBuilder

class DatasetLoader(Dataset):

    def __init__(self, args, dtype=None, isTrain=True):

        # exp settings
        exp_type = 'train' if isTrain else 'valid'

        config = read_config()
        dataset_path = config['Argoverse_Forecasting']['dataset_path'] + exp_type # update, 211112

        # map loader
        self.map = ArgoverseMap()

        # params
        self.dtype = dtype
        self.min_obs_len = int(args.min_past_horizon_seconds * args.target_sample_period)
        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.sub_step = int(10.0 / args.target_sample_period)
        self.target_sample_period = args.target_sample_period

        self.path_resol = args.path_resol
        self.path_len_f = args.max_path_len_forward
        self.path_len_b = args.max_path_len_backward
        self.num_pos_f = int(args.max_path_len_forward / self.path_resol)
        self.num_pos_b = int(args.max_path_len_backward / self.path_resol)
        self.num_max_paths = args.num_max_paths
        self.is_random_path_order = args.is_random_path_order

        # checks existance of dataset file and create
        save_path = config['Argoverse_Forecasting']['preproc_dataset_path'] + '/%dsec_%dsec' % (args.past_horizon_seconds, args.future_horizon_seconds)
        if (os.path.exists(save_path)==False):
            os.mkdir(save_path)
        file_name = 'argoverse_forecasting_%s.cpkl' % exp_type

        if (os.path.exists(os.path.join(save_path, file_name))==False):
            builder = DatasetBuilder(args=args, dataset_path=dataset_path)
            builder.make_preprocessed_data(os.path.join(save_path, file_name), exp_type)

        # load scenes
        with open(os.path.join(save_path, file_name), 'rb') as f:
            dataset = dill.load(f, encoding='latin1')

        self.train_data, self.valid_data = [], [] # update, 211015 (b/o exp365)
        for _, scene in enumerate(tqdm(dataset[0], desc='refactoring train data')):
            self.train_data += self.refactoring(scene)
        for _, scene in enumerate(tqdm(dataset[1], desc='refactoring valid data')):
            self.valid_data += self.refactoring(scene, isTrain=False)
        self.test_data = dataset[2]

        self.num_train_scenes = len(self.train_data)
        self.num_valid_scenes = len(self.valid_data)
        self.num_test_scenes = len(self.test_data)

        print(">> Loader is loaded from {%s} " % os.path.basename(__file__))
        print(">> num train/valid samples : %d / %d" % (self.num_train_scenes, self.num_valid_scenes))
        print(">> num test scenes : %d " % self.num_test_scenes)


    def __len__(self):
        # return len(self.train_data)
        return 512


    def __getitem__(self, idx):

        '''
        obs_traj_ta : seq_len x 1 x dim
        future_traj_ta : seq_len x 1 x dim
        obs_traj_ngh : seq_len x num_neighbors x dim
        future_traj_ngh : seq_len x num_neighbors x dim
        map : 1 x 3 x map_size x map_size
        num_neighbors : 1
        valid_neighbor :  1 (bool)
        possible_paths : seq_len x num_max_paths x dim
        '''

        # current scene data
        obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, map, num_neighbors, valid_neighbor, possible_paths, \
            lane_label = self.extract_data_from_scene(self.train_data[idx])

        obs_traj_ta = torch.from_numpy(obs_traj_ta).type(self.dtype)
        future_traj_ta = torch.from_numpy(future_traj_ta).type(self.dtype)
        obs_traj_ngh = torch.from_numpy(obs_traj_ngh).type(self.dtype)
        future_traj_ngh = torch.from_numpy(future_traj_ngh).type(self.dtype)
        map = torch.from_numpy(map).permute(2, 0, 1).type(self.dtype)
        map = torch.unsqueeze(map, dim=0)
        possible_paths = torch.from_numpy(possible_paths).type(self.dtype)
        lane_label = torch.from_numpy(lane_label).type(self.dtype)

        return obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, \
               map, num_neighbors, valid_neighbor, possible_paths, lane_label

    def next_sample(self, index, mode):

        '''
        obs_traj_ta : seq_len x 1 x dim
        future_traj_ta : seq_len x 1 x dim
        obs_traj_ngh : seq_len x num_neighbors x dim
        future_traj_ngh : seq_len x num_neighbors x dim
        map : 1 x 3 x map_size x map_size
        num_neighbors : 1
        valid_neighbor :  1 (bool)
        possible_paths : seq_len x num_max_paths x dim
        '''

        # current scene data
        if (mode == 'valid'):
            obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, map, num_neighbors, valid_neighbor, possible_paths, \
                lane_label = self.extract_data_from_scene(self.valid_data[index])
            return obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, \
                   map, num_neighbors, valid_neighbor, possible_paths, lane_label
        else:
            scene = self.test_data[index]
            agent_samples = self.refactoring(scene)

            obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, num_neighbors, valid_neighbor, agent_ids, possible_paths,\
                lane_label = [], [], [], [], [], [], [], [], [], []

            for i in range(len(agent_samples)):

                # agent_index
                agent_id = agent_samples[i].target_agent_index

                # current sample
                _obs_traj, _future_traj, _obs_traj_ngh, _future_traj_ngh, _map, _num_neighbors, _valid_neighbor, _possible_paths,\
                    _lane_label = self.extract_data_from_scene(agent_samples[i])

                _obs_traj = torch.from_numpy(_obs_traj).type(self.dtype)
                _future_traj = torch.from_numpy(_future_traj).type(self.dtype)
                _obs_traj_ngh = torch.from_numpy(_obs_traj_ngh).type(self.dtype)
                _future_traj_ngh = torch.from_numpy(_future_traj_ngh).type(self.dtype)
                _map = torch.from_numpy(_map).permute(2, 0, 1).type(self.dtype)
                _map = torch.unsqueeze(_map, dim=0)
                _possible_paths = torch.from_numpy(_possible_paths).type(self.dtype)
                _lane_label = torch.from_numpy(_lane_label).type(self.dtype)

                obs_traj.append(_obs_traj)
                future_traj.append(_future_traj)
                obs_traj_ngh.append(_obs_traj_ngh)
                future_traj_ngh.append(_future_traj_ngh)
                map.append(_map)
                num_neighbors.append(_num_neighbors)
                valid_neighbor.append(_valid_neighbor)
                agent_ids.append(agent_id)
                possible_paths.append(_possible_paths)
                lane_label.append(_lane_label)

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
            agent_ids = np.array(agent_ids)
            possible_paths = torch.cat(possible_paths, dim=1)
            lane_label = torch.cat(lane_label, dim=0)


            return [obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, seq_start_end, valid_neighbor,
                    possible_paths, lane_label, agent_ids, agent_samples, scene]


    def refactoring(self, scene, isTrain=True):

        '''
        Convert Ego-centric (or AV-centric) data to Agent-centric data
        '''

        samples = []

        # current sample (current time)
        num_total_agents = scene.num_agents

        # all agent's trajectories
        agent_ids = np.zeros(shape=(1, num_total_agents))
        trajectories_g = np.full(shape=(self.obs_len+self.pred_len, num_total_agents, 2), fill_value=np.nan)
        for idx, track_id in enumerate(scene.agent_dict):
            agent_ids[0, idx] = scene.agent_dict[track_id].agent_id
            trajectory = scene.agent_dict[track_id].trajectory
            trajectories_g[:, idx, :] = trajectory[:, 1:3]


        # find agents who has valid observation
        valid_flag = np.sum(trajectories_g[self.obs_len - self.min_obs_len:self.obs_len, :, 0], axis=0) > -1000
        trajectories_g = trajectories_g[:, valid_flag, :]
        agent_ids = agent_ids[:, valid_flag]


        # for all agents
        for a in range(agent_ids.shape[1]):

            agent_track_id = scene.id_2_token_lookup[agent_ids[0, a]]
            agent = scene.agent_dict[agent_track_id]
            agent_type = agent.agent_type

            if (agent_type == 'OTHERS'):
                continue

            if (agent.speed_mps < 1.38 and np.isnan(agent.heading_from_lane) == False):
                Ra2g, Rg2a = agent.rotation_matrix_from_yaw(agent.heading_from_lane)
            else:
                Ra2g, Rg2a = agent.Ra2g, agent.Rg2a
            trans_g = agent.trans_g

            # skip if the target agent doesn't has full trajectory
            FLAG = np.min(trajectories_g[self.obs_len:, a, 0]) > -1000
            if not FLAG:
                continue

            # trajectories
            trajectories_a = np.copy(trajectories_g)
            for aa in range(agent_ids.shape[1]):
                trajectories_a[:, aa, :] = np.matmul(Rg2a, (trajectories_g[:, aa, :] - trans_g).T).T

            agent_sample = AgentCentricScene(scene_id=scene.scene_id, track_id=agent_track_id, city_name=scene.city_name)
            agent_sample.target_agent_index = agent_ids[0, a]
            agent_sample.trajectories = trajectories_a
            agent_sample.R_a2g = Ra2g
            agent_sample.R_g2a = Rg2a
            agent_sample.trans_g = trans_g
            agent_sample.agent_ids = agent_ids
            agent_sample.possible_lanes = agent.possible_lanes
            samples.append(agent_sample)

        return samples


    def extract_data_from_scene(self, scene, isTrain=True):
        '''
        Extract training data from Scene
        '''


        city_name = scene.city_name
        agent_ids = scene.agent_ids
        target_agent_index = scene.target_agent_index
        trajectories = scene.trajectories
        possible_lanes = scene.possible_lanes

        R_g2a = scene.R_g2a
        R_a2g = scene.R_a2g
        trans_g = scene.trans_g

        # split into target agent and neighbors
        idx = np.argwhere(agent_ids[0, :] == target_agent_index)[0][0]

        num_agents = agent_ids.shape[1]
        trajectory_ta = np.full(shape=(self.obs_len+self.pred_len, 1, 2), fill_value=np.nan)
        trajectories_ngh = []
        for l in range(num_agents):
            if (l == idx):
                trajectory_ta[:, :, :] = np.copy(trajectories[:, l, :].reshape(self.obs_len+self.pred_len, 1, 2))
            else:
                trajectories_ngh.append(np.copy(trajectories[:, l, :].reshape(self.obs_len+self.pred_len, 1, 2)))

        num_neighbors = len(trajectories_ngh)
        if (num_neighbors == 0):
            trajectories_ngh = np.full(shape=(self.obs_len+self.pred_len, 1, 2), fill_value=np.nan)
            valid_neighbor = False
            num_neighbors += 1
        else:
            trajectories_ngh = np.concatenate(trajectories_ngh, axis=1)
            valid_neighbor = True


        # calc speed and heading
        trajectory_ta_ext = self.calc_speed_heading(trajectory_ta)
        trajectories_ngh_ext = []
        for n in range(trajectories_ngh.shape[1]):
            trajectories_ngh_ext.append(self.calc_speed_heading(trajectories_ngh[:, n, :].reshape(self.obs_len+self.pred_len, 1, 2)))
        trajectories_ngh_ext = np.concatenate(trajectories_ngh_ext, axis=1)

        # split into observation and future
        obs_traj_ta = np.copy(trajectory_ta_ext[:self.obs_len])
        future_traj_ta = np.copy(trajectory_ta_ext[self.obs_len:])
        obs_traj_ngh = np.copy(trajectories_ngh_ext[:self.obs_len])
        future_traj_ngh = np.copy(trajectories_ngh_ext[self.obs_len:])

        # remove 'nan' in observation
        obs_traj_ta = self.remove_nan(obs_traj_ta)
        obs_traj_ngh = self.remove_nan(obs_traj_ngh)

        # fake topview map
        map = np.zeros(shape=(10, 10, 3))

        # NOTE : must shuffle the order of the possible lanes
        possible_paths, lane_label = self.get_lane_coords(possible_lanes, R_g2a, trans_g, scene.city_name)

        return obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, map, num_neighbors, valid_neighbor, \
               possible_paths, lane_label


    def calc_speed_heading(self, trajectory):

        '''
        trajectory : seq_len x batch x 3
        '''

        # params
        seq_len, batch, dim = trajectory.shape

        # speed (m/s) and heading (rad)
        traj = np.copy(np.squeeze(trajectory)[:, :2])
        pos_diff = np.zeros_like(traj)
        pos_diff[1:, :] = traj[1:] - traj[:-1]

        # speed
        speed_mps = self.target_sample_period * np.sqrt(np.sum(pos_diff ** 2, axis=1)).reshape(seq_len, 1, 1)

        # heading
        heading_rad = np.arctan2(pos_diff[:, 1], pos_diff[:, 0]).reshape(seq_len, 1, 1)

        return np.concatenate([speed_mps, heading_rad, trajectory], axis=2)

    def remove_nan(self, seq):

        '''
        seq : seq_len x batch x 2
        '''

        seq_copy = np.copy(seq)
        for i in range(seq.shape[1]):
            cur_seq = np.copy(seq[:, i, :])
            if (np.count_nonzero(np.isnan(cur_seq[:-self.min_obs_len])) > 0):
                seq_copy[:-self.min_obs_len, i, :] = 0.0

        return seq_copy

    def get_lane_coords(self, possible_lanes, R_g2a, trans_g, location):

        filter = np.array([0.5, 0, 0.5])
        target_spacing = np.arange(0, self.path_len_f, self.path_resol)
        min_path_len = 5.0 # meter

        path_list = self.map.get_cl_from_lane_seq(possible_lanes, location)
        possible_paths = []
        for path in path_list:
            path_agent_centric = np.matmul(R_g2a, (path - trans_g).T).T

            # find target segment
            start_idx = np.argmin(np.sum(np.abs(path_agent_centric[:, :2]), axis=1))
            path_agent_centric = path_agent_centric[start_idx:]

            point_dist = np.zeros(shape=(path_agent_centric.shape[0]))
            point_dist[1:] = np.sqrt(np.sum((path_agent_centric[1:] - path_agent_centric[:-1])**2, axis=1))
            path_len = np.sum(point_dist)
            if (path_len < min_path_len):
                continue


            # sample equally-spaced
            path_agent_centric_intp = []
            for l in range(1, path_agent_centric.shape[0]):
                prev_x, prev_y = path_agent_centric[l-1]
                cur_x, cur_y = path_agent_centric[l]

                resol = int(10 * point_dist[l] / self.path_resol)

                intp_x = np.linspace(prev_x, cur_x, resol, endpoint=False)
                intp_y = np.linspace(prev_y, cur_y, resol, endpoint=False)

                seg_len = intp_x.size
                cur_seg = np.concatenate([intp_x.reshape(seg_len, 1), intp_y.reshape(seg_len, 1)], axis=1)

                if (len(cur_seg) == 0):
                    continue

                path_agent_centric_intp.append(cur_seg)

            path_agent_centric_intp = np.concatenate(path_agent_centric_intp, axis=0)
            point_dist = np.zeros(shape=(path_agent_centric_intp.shape[0]))
            point_dist[1:] = np.sqrt(np.sum((path_agent_centric_intp[1:] - path_agent_centric_intp[:-1])**2, axis=1))


            sorted_index = np.searchsorted(np.cumsum(point_dist), target_spacing, side='right')
            chk = sorted_index < path_agent_centric_intp.shape[0]
            sorted_index = sorted_index[chk]
            path_agent_centric = path_agent_centric_intp[sorted_index]

            # centerline quality
            seq_len = path_agent_centric.shape[0]
            point_dist = self.path_resol * np.ones(shape=seq_len)
            point_dist[1:] = np.sqrt(np.sum((path_agent_centric[1:] - path_agent_centric[:-1]) ** 2, axis=1))

            # lane smoothing
            if (np.max(point_dist) > 1.1*self.path_resol or np.min(point_dist) < 0.9*self.path_resol):
                path_agent_centric_x_avg = np.convolve(path_agent_centric[:, 0], filter, mode='same').reshape(seq_len, 1)
                path_agent_centric_y_avg = np.convolve(path_agent_centric[:, 1], filter, mode='same').reshape(seq_len, 1)
                path_agent_centric_avg = np.concatenate([path_agent_centric_x_avg, path_agent_centric_y_avg], axis=1)

                chk = point_dist > 1.1*self.path_resol
                path_agent_centric[chk] = path_agent_centric_avg[chk]

                chk = point_dist < 0.9*self.path_resol
                path_agent_centric[chk] = path_agent_centric_avg[chk]

            # length of current lane
            path_len = path_agent_centric.shape[0]
            if (path_len < int(min_path_len/self.path_resol)):
                path_agent_centric = np.full(shape=(self.num_pos_f, 2), fill_value=np.nan) # update, 211015 (b/o exp363)
                path_len = path_agent_centric.shape[0]

            # increase length of current lane
            if (path_len < self.num_pos_f):
                num_repeat = self.num_pos_f - path_len
                delta = (path_agent_centric[1:] - path_agent_centric[:-1])[-1].reshape(1, 2)
                delta = np.repeat(delta, num_repeat, axis=0)
                delta[0, :] += path_agent_centric[-1]
                padd = np.cumsum(delta, axis=0)
                path_agent_centric = np.concatenate([path_agent_centric, padd], axis=0)

            possible_paths.append(np.expand_dims(path_agent_centric, axis=1))
            assert(path_agent_centric.shape[0] == self.num_pos_f)

        num_repeat = 0
        if (self.num_max_paths > len(possible_paths)):
            num_repeat = self.num_max_paths - len(possible_paths)
            for i in range(num_repeat):
                possible_paths.append(np.full(shape=(self.num_pos_f, 1, 2), fill_value=np.nan))
        indices = [idx for idx in range(self.num_max_paths)]

        if (self.is_random_path_order == 1):
            random.shuffle(indices)

        possible_paths_random = []
        for _, idx in enumerate(indices):
            possible_paths_random.append(possible_paths[idx])
        possible_paths_random = np.concatenate(possible_paths_random, axis=1)

        # reference lane index
        label = np.zeros(shape=(1, self.num_max_paths))
        if (num_repeat == self.num_max_paths):
            best_match_idx = indices[0]
        else:
            best_match_idx = np.argwhere(np.array(indices) == 0)[0][0]
        label[0, best_match_idx] = 1

        return possible_paths_random, label

    def convert_to_egocentric(self, _obs_traj, _future_traj, _pred_traj, agent_ids, agent_samples):

        best_k = _pred_traj.shape[0]
        num_agents = len(agent_samples)

        # ego-vehicle R & T
        idx = np.argwhere(agent_ids == 0)[0][0]
        assert (idx == 0)
        R_g2e = agent_samples[idx].R_g2a
        trans_g_e = agent_samples[idx].trans_g


        obs_traj, future_traj, pred_traj_k = [], [], []
        for i in range(num_agents):

            # ego-vehicle
            if (agent_ids[i] == 0):
                obs = _obs_traj[:, i, :].reshape(self.obs_len, 1, 2)
                future = _future_traj[:, i, :].reshape(self.pred_len, 1, 2)
                preds = _pred_traj[:, :, i, :].reshape(best_k, self.pred_len, 1, 2)

                obs_traj.append(obs)
                future_traj.append(future)
                pred_traj_k.append(preds)

            else:
                R_a2g = agent_samples[i].R_a2g
                trans_g_a = agent_samples[i].trans_g

                obs = _obs_traj[:, i, :]
                future = _future_traj[:, i, :]
                preds = _pred_traj[:, :, i, :]

                obs = self.to_agent(R_g2e, trans_g_e, self.to_global(R_a2g, trans_g_a, obs))
                future = self.to_agent(R_g2e, trans_g_e, self.to_global(R_a2g, trans_g_a, future))

                preds_k = []
                for k in range(best_k):
                    pred = self.to_agent(R_g2e, trans_g_e, self.to_global(R_a2g, trans_g_a, preds[k, :, :]))
                    preds_k.append(np.expand_dims(pred, axis=0))
                preds = np.concatenate(preds_k, axis=0)

                obs_traj.append(np.expand_dims(obs, axis=1))
                future_traj.append(np.expand_dims(future, axis=1))
                pred_traj_k.append(np.expand_dims(preds, axis=2))

        obs_traj = np.concatenate(obs_traj, axis=1)
        future_traj = np.concatenate(future_traj, axis=1)
        pred_traj_k = np.concatenate(pred_traj_k, axis=2)

        return obs_traj, future_traj, pred_traj_k

    def to_global(self, Rtog, trans_g, input):

        return np.matmul(Rtog, input.T).T + trans_g

    def to_agent(self, Rtoa, trans_g, input):

        return np.matmul(Rtoa, (input - trans_g).T).T


