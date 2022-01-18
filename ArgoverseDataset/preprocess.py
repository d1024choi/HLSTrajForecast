from utils.libraries import *
from utils.functions import read_config, print_voxelization_progress
from ArgoverseDataset.agent import Agent
from ArgoverseDataset.scene import Scene
from utils.trajectory_filter import AverageFilter
from ArgoverseDataset.argoverse.map_representation.map_api import ArgoverseMap
from ArgoverseDataset.argoverse.utils.centerline_utils import remove_overlapping_lane_seq


class DatasetBuilder:

    def __init__(self, args, dataset_path, isTrain=True):

        self.map = ArgoverseMap()

        self.dataset_path = dataset_path + '/data'
        self.scene_list = sorted(glob.glob(os.path.join(self.dataset_path, '*.csv')))

        self.val_ratio = args.val_ratio
        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.scene_accept_prob = args.scene_accept_prob
        self.target_sample_period = args.target_sample_period
        self.sub_step = int(10.0 / args.target_sample_period)
        self.limit_range = args.limit_range

        self.path_resol = args.path_resol
        self.max_path_len_forward = args.max_path_len_forward
        self.max_path_len_backward = args.max_path_len_backward


    def find_best_matched_lane(self, possible_lanes, agent, city_name):

        future_traj = agent.trajectory[self.obs_len:, 1:3]
        centerlines = self.map.get_cl_from_lane_seq(possible_lanes, city_name)

        minDist, target_idx = 1e10, 0
        for idx, centerline in enumerate(centerlines):

            total_dists = 0
            for t in range(self.pred_len):
                cur_pos = future_traj[t, :].reshape(1, 2)
                if (np.count_nonzero(np.isnan(cur_pos)) == 0):
                    dists = np.sqrt(np.sum((centerline - cur_pos)**2, axis=1))
                    total_dists += np.min(dists)

            if (total_dists < minDist):
                minDist = total_dists
                target_idx = idx

        return possible_lanes[target_idx]


    def find_lane_candidates(self, agent, city_name):

        xy = agent.trajectory[:self.obs_len, 1:3]

        # Get all lane candidates within a bubble
        manhattan_threshold = 2.5
        curr_lane_candidates = self.map.get_lane_ids_in_xy_bbox(xy[-1, 0], xy[-1, 1], city_name, manhattan_threshold)

        # Keep expanding the bubble until at least 1 lane is found
        max_search_radius = 10
        while len(curr_lane_candidates) < 1 and manhattan_threshold < max_search_radius:
            manhattan_threshold *= 2
            curr_lane_candidates = self.map.get_lane_ids_in_xy_bbox(xy[-1, 0], xy[-1, 1], city_name, manhattan_threshold)

        if (len(curr_lane_candidates) == 0):
            return [], np.nan

        # remove lanes in opposite direction
        minDist, target_idx, cnt = 1e10, 0, 0
        lane_seg_dicts = []
        for curr_lane in curr_lane_candidates:

            # lane seg coord
            curr_lane_cntline = self.map.get_lane_segment_centerline(curr_lane, city_name)[:, :2]

            # the closest point to agent curr position
            dists = np.sqrt(np.sum((curr_lane_cntline - xy[-1, :].reshape(1, 2))**2, axis=1))
            min_idx = np.argmin(dists)
            if (min_idx == 0):
                min_idx = 1

            # lane yaw at the point
            diff = curr_lane_cntline[min_idx] - curr_lane_cntline[min_idx-1]
            lane_yaw_global = np.arctan2(diff[1], diff[0])

            lane_seg_dict = dict({'id': curr_lane,
                                  'coord' : curr_lane_cntline,
                                  'min_dist' : dists[min_idx],
                                  'min_idx' : min_idx,
                                  'lane_yaw' : lane_yaw_global
                                  })
            lane_seg_dicts.append(lane_seg_dict)

            if (dists[min_idx] < minDist):
                minDist = dists[min_idx]
                target_idx = cnt
            cnt += 1

        target_lane_yaw = lane_seg_dicts[target_idx]['lane_yaw']
        if (agent.speed_mps < 1.38):
            Ra2g, Rg2a = agent.rotation_matrix_from_yaw(target_lane_yaw)
        else:
            Rg2a = agent.Rg2a


        curr_lane_candidates = []
        for ldict in lane_seg_dicts:

            min_idx = ldict['min_idx']
            coord_a = np.matmul(Rg2a, (ldict['coord'] - agent.trans_g).T).T
            diff = coord_a[min_idx] - coord_a[min_idx-1]
            lane_yaw = np.arctan2(diff[1], diff[0])
            lane_yaw_deg = np.rad2deg(lane_yaw)

            if (abs(lane_yaw_deg) > (180.0 / 4)):
                continue

            curr_lane_candidates.append(ldict['id'])


        # Set dfs threshold
        dfs_threshold = self.max_path_len_forward

        # DFS to get all successor and predecessor candidates
        obs_pred_lanes= []
        for lane in curr_lane_candidates:
            candidates_future = self.map.dfs(lane, city_name, 0, dfs_threshold)
            candidates_past = self.map.dfs(lane, city_name, 0, dfs_threshold, True)

            # Merge past and future
            for past_lane_seq in candidates_past:
                for future_lane_seq in candidates_future:
                    assert past_lane_seq[-1] == future_lane_seq[0], "Incorrect DFS for candidate lanes past and future"
                    obs_pred_lanes.append(past_lane_seq + future_lane_seq[1:])

        # Removing overlapping lanes
        obs_pred_lanes = remove_overlapping_lane_seq(obs_pred_lanes)

        # Remove unnecessary extended predecessors
        obs_pred_lanes = self.map.remove_extended_predecessors(obs_pred_lanes, xy, city_name)


        # find best matched lane
        if (len(obs_pred_lanes)>1):
            target_lane = self.find_best_matched_lane(obs_pred_lanes, agent, city_name)
            obs_pred_lanes.remove(target_lane)
            obs_pred_lanes.insert(0, target_lane)

        return obs_pred_lanes, target_lane_yaw


    def make_preprocessed_data(self, file_path, exp_type):

        print('>> Making preprocessed data ..')

        # set seed
        np.random.seed(1)

        scene_list_train, scene_list_val, scene_list_test = [], [], []
        for _, path in enumerate(tqdm(self.scene_list)):

            if (np.random.rand(1) > self.scene_accept_prob):
                continue

            data = pd.read_csv(path)

            city_name = data["CITY_NAME"].values[0]
            ego_data = data.loc[data["OBJECT_TYPE"] == 'AV']
            track_ids_agent = np.unique(data.loc[data["OBJECT_TYPE"] == 'AGENT']["TRACK_ID"])
            track_ids_others = np.unique(data.loc[data["OBJECT_TYPE"] == 'OTHERS']["TRACK_ID"])
            track_ids = track_ids_agent.tolist() + track_ids_others.tolist()
            agent_types = ['AGENT'] + ['OTHERS'] * len(track_ids_others)
            timestamps = np.unique(data["TIMESTAMP"].values)

            # Create agent dictionary for the current scene
            agent_dict = {}
            agent_dict['EGO'] = Agent(track_id='EGO', agent_type='EGO', obs_len=self.obs_len, pred_len=self.pred_len)
            ego_data_np = np.array(ego_data)


            # all agents except for EGO
            agent_datas = []
            num_agents = len(track_ids)
            for _, track_id in enumerate(track_ids):
                agent_datas.append(data.loc[data["TRACK_ID"] == track_id])
            agent_datas_np = [np.array(agent_datas[i]) for i in range(num_agents)]

            ego_x_cur = ego_data_np[2 * self.obs_len - 2, 3]
            ego_y_cur = ego_data_np[2 * self.obs_len - 2, 4]
            agent_x_cur = agent_datas_np[0][2 * self.obs_len - 2, 3]
            agent_y_cur = agent_datas_np[0][2 * self.obs_len - 2, 4]

            # select agents inside the limit range
            agent_datas = []
            for _, track_id in enumerate(track_ids):
                try:
                    x_cur = agent_datas_np[_][2 * self.obs_len - 2, 3]
                    y_cur = agent_datas_np[_][2 * self.obs_len - 2, 4]
                except:
                    x_cur, y_cur = -1000, -1000

                dist2ego = np.sqrt((ego_x_cur - x_cur)**2 + (ego_y_cur - y_cur)**2)
                dist2agent = np.sqrt((agent_x_cur - x_cur) ** 2 + (agent_y_cur - y_cur) ** 2)

                if (dist2ego < self.limit_range or dist2agent < self.limit_range):
                    agent_dict[track_id] = Agent(track_id=track_id, agent_type=agent_types[_], obs_len=self.obs_len, pred_len=self.pred_len)
                    agent_datas.append(data.loc[data["TRACK_ID"] == track_id])
            num_agents = len(agent_datas)
            agent_datas_np = [np.array(agent_datas[i]) for i in range(num_agents)]


            for idx in range(0, len(timestamps), self.sub_step):

                cur_timestamp = ego_data_np[idx, 0]
                ego_x = ego_data_np[idx, 3]
                ego_y = ego_data_np[idx, 4]

                agent_dict['EGO'].trajectory[int(idx / 2), 0] = cur_timestamp
                agent_dict['EGO'].trajectory[int(idx / 2), 1] = ego_x
                agent_dict['EGO'].trajectory[int(idx / 2), 2] = ego_y

                for _, agent_data_np in enumerate(agent_datas_np):

                    cur_agent_data = agent_data_np[agent_data_np[:, 0] == cur_timestamp]

                    if (cur_agent_data.size == 0):
                        continue

                    agent_track_id = cur_agent_data[0, 1]
                    agent_x = cur_agent_data[0, 3]
                    agent_y = cur_agent_data[0, 4]

                    agent_dict[agent_track_id].trajectory[int(idx / 2), 0] = cur_timestamp
                    agent_dict[agent_track_id].trajectory[int(idx / 2), 1] = agent_x
                    agent_dict[agent_track_id].trajectory[int(idx / 2), 2] = agent_y



            for idx, key in enumerate(agent_dict):
                agent_dict[key].agent_id = idx
                agent_dict[key].heading_from_traj()
                agent_dict[key].calc_speed(self.target_sample_period)

                if (agent_dict[key].agent_type == 'OTHERS'):
                    continue

                possible_lanes, target_lane_yaw = self.find_lane_candidates(agent_dict[key], city_name)
                agent_dict[key].possible_lanes += possible_lanes
                agent_dict[key].heading_from_lane = target_lane_yaw



            # save in the scene
            scene_id = path[path.find('/data/')+6:]
            scene = Scene(scene_id=scene_id, agent_dict=agent_dict, city_name=city_name)
            scene.make_id_2_token_lookup()


            if (exp_type == 'train'):
                if (np.random.rand(1) < self.val_ratio):
                    scene_list_val.append(scene)
                else:
                    scene_list_train.append(scene)
            else:
                scene_list_test.append(scene)

        # save
        with open(file_path, 'wb') as f:
            dill.dump([scene_list_train, scene_list_val, scene_list_test], f, protocol=dill.HIGHEST_PROTOCOL)
        print('>> {%s} is created .. ' % file_path)

