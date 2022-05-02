from utils.functions import *
from NuscenesDataset.nuscenes.map_expansion.arcline_path_utils import discretize_lane
from NuscenesDataset.nuscenes.utils.geometry_utils import transform_matrix
from NuscenesDataset.nuscenes.map_expansion.map_api import NuScenesMap

class Map:

    def __init__(self, args, nusc):

        self.nusc = nusc

        # Nuscenes Map loader
        self.nusc_maps = {}
        self.nusc_maps['singapore-onenorth'] = NuScenesMap(dataroot=args.dataset_path, map_name='singapore-onenorth')
        self.nusc_maps['singapore-hollandvillage'] = NuScenesMap(dataroot=args.dataset_path, map_name='singapore-hollandvillage')
        self.nusc_maps['singapore-queenstown'] = NuScenesMap(dataroot=args.dataset_path, map_name='singapore-queenstown')
        self.nusc_maps['boston-seaport'] = NuScenesMap(dataroot=args.dataset_path, map_name='boston-seaport')

        self.centerlines = {}
        self.centerlines['singapore-onenorth'] = self.nusc_maps['singapore-onenorth'].discretize_centerlines(resolution_meters=0.25)
        self.centerlines['singapore-hollandvillage'] = self.nusc_maps['singapore-hollandvillage'].discretize_centerlines(resolution_meters=0.25)
        self.centerlines['singapore-queenstown'] = self.nusc_maps['singapore-queenstown'].discretize_centerlines(resolution_meters=0.25)
        self.centerlines['boston-seaport'] = self.nusc_maps['boston-seaport'].discretize_centerlines(resolution_meters=0.25)

        # params
        self.min_forward_len = args.max_path_len_forward
        self.min_backward_len = args.max_path_len_backward

    def _transform_pc(self, R, translation, pc):
        pc_trans = pc - translation.reshape(1, 3)
        return np.matmul(R, pc_trans.T).T

    def _transform_pc_inv(self, R, translation, pc):
        pc_trans_inv = np.matmul(R, pc.T).T + translation.reshape(1, 3)
        return pc_trans_inv

    def find_best_lane(self, possible_lanes, trajectory, scene_location, obs_len):

        '''
        Input
        possible_lanes : list of 'token sequence list'
        trajectory : global coordinate positions

        Output
        sorted_lanes : list of 'token sequence list' (best matched lane comes first)
        '''

        min_vals = []
        for _, tok_seq in enumerate(possible_lanes):

            path = []
            for __, tok in enumerate(tok_seq):
                lane_record = self.nusc_maps[scene_location].get_arcline_path(tok)
                coords = np.array(discretize_lane(lane_record, resolution_meters=0.25))
                path.append(coords)
            path = np.concatenate(path, axis=0)

            min_val = 0
            for i in range(obs_len, trajectory.shape[0]):
                err = np.sum(np.abs(path[:, :2] - trajectory[i, :2].reshape(1, 2)), axis=1)
                min_val += np.min(err)
            min_vals.append(min_val)

        sorted_lanes = []
        sort_idx = np.argsort(np.array(min_vals))
        for i in range(len(min_vals)):
            sorted_lanes.append(possible_lanes[sort_idx[i]])

        return sorted_lanes

    def find_possible_lanes(self, agent, lidar_now_token, scene_location):


        if (agent.track_id == 'EGO'):
            lidar_now_data = self.nusc.get('sample_data', lidar_now_token)
            pose = self.nusc.get('ego_pose', lidar_now_data['ego_pose_token'])
        else:
            pose = self.nusc.get('sample_annotation', agent.track_id)

        R = transform_matrix(pose['translation'], pyquaternion.Quaternion(pose['rotation']), inverse=False)
        Rinv = transform_matrix(pose['translation'], pyquaternion.Quaternion(pose['rotation']), inverse=True)
        v = np.dot(R[:3, :3], np.array([1, 0, 0]))
        agent_yaw = np.arctan2(v[1], v[0])

        xyz = np.array(pose['translation'])
        x, y, z = xyz

        # find the lanes inside the range
        lanes = self.get_lane_records_in_radius(x, y, scene_location)
        lanes = lanes['lane'] + lanes['lane_connector']

        # remove lane segments with opposite direction [tok, tok, ...]
        target_lanes = self.remove_opposite_directions(copy.deepcopy(lanes), scene_location, xyz, Rinv, agent_yaw)

        # merge connected lanes [[tok,tok, ...], [tok], ...]
        target_path_lists = self.merge_connected_lanes(copy.deepcopy(target_lanes), scene_location)

        # add incoming lane segments
        target_path_lists = self.add_incoming_lanes(copy.deepcopy(target_path_lists), scene_location)

        # find possible outgoing paths
        num_levels = 10
        for l in range(num_levels):
            target_path_lists = self.find_next_level_lanes(copy.deepcopy(target_path_lists), scene_location)

        # prune line segs in paths
        if (len(target_path_lists) > 0):
            target_path_lists = self.prune_lane_segs(copy.deepcopy(target_path_lists), xyz[:2], scene_location)
            target_path_lists = self.remove_overlapped_paths(copy.deepcopy(target_path_lists))


        return target_path_lists

    def prune_lane_segs(self, path_list, xy, scene_location):

        # remove lanes w/o arc path
        path_list_tmp = []
        for i in range(len(path_list)):
            tok_seq = []
            for j in range(len(path_list[i])):
                cur_tok = path_list[i][j]
                if not self.nusc_maps[scene_location].arcline_path_3.get(cur_tok):
                    do_nothing = 0
                else:
                    tok_seq.append(cur_tok)
            if (len(tok_seq) > 0):
                path_list_tmp.append(tok_seq)
        path_list = copy.deepcopy(path_list_tmp)

        out_list = []
        for _, tok_list in enumerate(path_list):

            num_segs = len(tok_list)
            info = np.zeros(shape=(num_segs, 4))
            for __, tok in enumerate(tok_list):

                lane_record = self.nusc_maps[scene_location].get_arcline_path(tok)
                coords = np.array(discretize_lane(lane_record, resolution_meters=0.25))

                # find the closest point
                dist = np.sqrt(np.sum((coords[:, :2] - xy.reshape(1, 2)) ** 2, axis=1))
                min_dist = np.min(dist)
                min_idx = np.argmin(dist)
                max_dist = np.max(dist)
                length = np.sqrt(np.sum((coords[0, :2] - coords[-1, :2]) ** 2))

                info[__, 0] = min_dist
                info[__, 1] = max_dist
                info[__, 2] = length
                info[__, 3] = np.sqrt(np.sum((coords[min_idx, :2] - coords[-1, :2]) ** 2))

            closest_lane_idx = np.argmin(info[:, 0])
            info[closest_lane_idx, 2] = info[closest_lane_idx, 3]

            backward_list = tok_list[:closest_lane_idx]
            forward_list = tok_list[closest_lane_idx:]

            candis = np.argwhere(np.cumsum(info[closest_lane_idx:, 2]) - self.min_forward_len > 0)
            if (candis.shape[0] == 0):
                last_forward_idx = len(forward_list) - 1
            else:
                last_forward_idx = candis[0, 0]

            candis = np.argwhere(np.cumsum(info[:closest_lane_idx, 2][::-1])[::-1] - self.min_backward_len > 0)
            if (candis.shape[0] == 0):
                last_backward_idx = 0
            else:
                last_backward_idx = candis[-1, 0]

            prune_list = backward_list[last_backward_idx:] + forward_list[:last_forward_idx+1]
            out_list.append(prune_list)

        return out_list

    def remove_overlapped_paths(self, path_list):

        out_list = []
        path_list_tmp = copy.deepcopy(path_list)
        while (len(path_list) > 0):
            cp = path_list[0]

            # after this loop, there is no 'cp' in the list
            while(cp in path_list_tmp):
                path_list_tmp.remove(cp)

            out_list.append(cp)
            path_list = copy.deepcopy(path_list_tmp)

        return out_list

    def get_lane_records_in_radius(self, x, y, scene_location, radius=10):
        lanes = self.nusc_maps[scene_location].get_records_in_radius(x, y, radius=radius, layer_names=['lane', 'lane_connector'])
        return lanes

    def remove_opposite_directions(self, lanes, scene_location, xyz, ego_from_global, agent_yaw):


        def _min(a, b):
            if (a >= b):
                return b
            else:
                return a

        def pos_neg_sort(a, b):
            if (a > 0):
                return a, b
            if (b > 0):
                return b, a

        def angle_difference(theta0, theta1):
            '''
            theta in degree
            '''

            if (theta0 == 0):
                theta0 += 0.0001

            if (theta1 == 0):
                theta1 += 0.0001


            # if two have the same sign
            if (theta0*theta1 > 0):
                angle_diff = abs(theta0 - theta1)
                return angle_diff

            else:
                pos_angle, neg_angle = pos_neg_sort(theta0, theta1)
                neg_angle_r = 360 - abs(neg_angle)

                if (pos_angle - neg_angle < 180):
                    angle_diff = pos_angle - neg_angle
                else:
                    angle_diff = abs(pos_angle - neg_angle_r)

                return angle_diff

        def trans_pc(R, pc):
            return np.matmul(R, np.concatenate([pc, np.ones(shape=(1, pc.shape[1]))], axis=0))[:3]

        target_lanes = []
        if (len(lanes) == 0):
            return target_lanes

        for _, tok in enumerate(lanes):

            # get lane record
            lane_record = self.nusc_maps[scene_location].get_arcline_path(tok)

            # get coordinates
            coords = np.array(discretize_lane(lane_record, resolution_meters=0.25))


            # find the closest point
            dist = np.sum((coords[:, :2] - np.array(xyz)[:2].reshape(1, 2))**2, axis=1)
            min_idx = np.argmin(dist)
            if (min_idx == 0):
                min_idx = 1

            coords_ego = trans_pc(ego_from_global, coords.T).T
            vec = (coords_ego[min_idx, :2] - coords_ego[min_idx - 1, :2]).reshape(1, 2)
            lane_yaw = calc_yaw_from_points(vec)[0]
            # TODO 
            # if (lane_yaw < 0):
            #    lane_yaw = 2*np.pi - abs(lane_yaw)
            lane_yaw_deg = np.rad2deg(lane_yaw)

            if (abs(lane_yaw_deg) > (180.0 / 4)):
                continue

            target_lanes.append(tok)

        return target_lanes

    def merge_connected_lanes(self, path_list, scene_location):

        remaining_lanes = []
        if (len(path_list) == 0):
            return remaining_lanes

        while (len(path_list) > 0):

            cur_tok = path_list[0]
            cur_tok_list = [cur_tok]
            while (1):

                prev_tok_list = self.nusc_maps[scene_location].get_incoming_lane_ids(cur_tok_list[0])
                next_tok_list = self.nusc_maps[scene_location].get_outgoing_lane_ids(cur_tok_list[-1])

                num_added_tok = 0
                if (len(prev_tok_list) > 0):
                    if (prev_tok_list[0] in path_list):
                        cur_tok_list.insert(0, prev_tok_list[0])
                        path_list.remove(prev_tok_list[0])
                        num_added_tok+=1

                if (len(next_tok_list) > 0):
                    if (next_tok_list[0] in path_list):
                        cur_tok_list.insert(len(cur_tok_list), next_tok_list[0])
                        path_list.remove(next_tok_list[0])
                        num_added_tok+=1

                if (num_added_tok == 0):
                    remaining_lanes.append(cur_tok_list)
                    path_list.remove(cur_tok)
                    break


        return remaining_lanes

    def find_next_level_lanes(self, path_list, scene_location):

        path_list_ext = []
        if (len(path_list) == 0):
            return path_list_ext

        for _, cur_tok_list in enumerate(path_list):
            if (cur_tok_list[-1] in self.nusc_maps[scene_location].connectivity):
                next_tok_list = self.nusc_maps[scene_location].get_outgoing_lane_ids(cur_tok_list[-1])
                if (len(next_tok_list) > 0):
                    for _, next_tok in enumerate(next_tok_list):
                        path_list_ext.append(cur_tok_list+ [next_tok])
                else:
                    path_list_ext.append(cur_tok_list)
            else:
                path_list_ext.append(cur_tok_list)


        # final connectivity check
        out_list = []
        for _, cur_tok_list in enumerate(path_list_ext):
            if (cur_tok_list[-1] not in self.nusc_maps[scene_location].connectivity):
                cur_tok_list.pop(len(cur_tok_list)-1)
            out_list.append(cur_tok_list)

        return out_list

    def add_incoming_lanes(self, path_list, scene_location):

        out_list = []
        if (len(path_list) == 0):
            return out_list

        for idx in range(len(path_list)):
            tok_list = path_list[idx]

            tok_m1_list = self.nusc_maps[scene_location].get_incoming_lane_ids(tok_list[0])
            if (len(tok_m1_list) > 0):
                tok_list.insert(0, tok_m1_list[0])

            out_list.append(tok_list)

        return out_list

    def __repr__(self):
        return f"Nuscenes Map Helper."


def in_range_points(points, x, y, z, x_range, y_range, z_range):

    points_select = points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], y < y_range[1], z > z_range[0], z < z_range[1]))]
    return np.around(points_select, decimals=2)


def correspondance_check(win_min_max, lane_min_max):

    # four points for window and lane box
    w_x_min, w_y_min, w_x_max, w_y_max = win_min_max
    l_x_min, l_y_min, l_x_max, l_y_max = lane_min_max

    w_TL = (w_x_min, w_y_max)  # l1
    w_BR = (w_x_max, w_y_min)  # r1

    l_TL = (l_x_min, l_y_max)  # l2
    l_BR = (l_x_max, l_y_min)  # r2

    # If one rectangle is on left side of other
    # if (l1.x > r2.x | | l2.x > r1.x)
    if (w_TL[0] > l_BR[0] or l_TL[0] > w_BR[0]):
        return False

    # If one rectangle is above other
    # if (l1.y < r2.y || l2.y < r1.y)
    if (w_TL[1] < l_BR[1] or l_TL[1] < w_BR[1]):
        return False

    return True

def calc_yaw_from_points(vec1):

    '''
    vec : seq_len x 2
    '''

    seq_len = vec1.shape[0]

    vec1 = vec1.reshape(seq_len, 2)
    vec2 = np.repeat(np.concatenate([np.ones(shape=(1, 1)), np.zeros(shape=(1, 1))], axis=1), seq_len, 0)

    x1 = vec1[:, 0]
    y1 = vec1[:, 1]
    x2 = vec2[:, 0]
    y2 = vec2[:, 1]

    dot = y1 * y2 + x1 * x2  # dot product
    det = y1 * x2 - x1 * y2  # determinant

    heading = np.arctan2(det, dot)  # -1x because of left side is POSITIVE

    return heading
