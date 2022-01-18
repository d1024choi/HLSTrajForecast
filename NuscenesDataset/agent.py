from utils.functions import *
from utils.trajectory_filter import LinearModelIntp, AverageFilter, HoleFilling

class Agent:

    def __init__(self, type=None, attribute=None, track_id=None, agent_id=None, obs_len=None, pred_len=None):

        self.type = type
        self.attribute = attribute
        self.track_id = track_id
        self.agent_id = agent_id

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.trajectory = np.full(shape=(obs_len+pred_len, 4), fill_value=np.nan)
        self.trajectory_global_coord = np.full(shape=(obs_len + pred_len, 4), fill_value=np.nan)

        self.wlh = np.full(shape=(3), fill_value=np.nan)
        self.occlusion = 0.0
        self.curvature = 0.0
        self.trajectory_len = 0.0
        self.status = []

        self.heading_traj = 0.0
        self.yaw = 0.0
        self.yaw_prev = 0.0
        self.speed = 0.0

        self.possible_lanes = []
        self.best_lane = -1

        # trajectory preprcoessing
        self.linearmodel = LinearModelIntp()
        self.avgfilter = AverageFilter(filter_size=3)
        self.holefill = HoleFilling()


    def bbox_3d(self):

        w, l, h = self.wlh

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])

        return np.vstack((x_corners, y_corners, z_corners))

    def bbox(self):

        w, l, h = self.wlh
        x, y = 0, 0

        x_front = x + (l / 2)
        x_back = x - (l / 2)
        y_left = y + (w / 2)
        y_right = y - (w / 2)

        bbox = [[x_front, y_left], [x_back, y_left], [x_back, y_right], [x_front, y_right], [x_front, y]]

        return np.array(bbox)

    def calc_speed(self, sample_period):

        '''
        sample period (Hz)
        '''

        obs_seq = self.trajectory[:self.obs_len, 1:3]
        obs_pos_last = obs_seq[-1]
        obs_pos_last_m1 = obs_seq[-2]

        vec1 = obs_pos_last - obs_pos_last_m1
        self.speed = 3.6 * sample_period * np.sqrt(np.sum(vec1 ** 2))

    def heading_from_traj(self):

        obs_seq = self.trajectory[:self.obs_len, 1:3]
        obs_pos_last = obs_seq[-1]
        obs_pos_last_m1 = obs_seq[-2]

        vec1 = (obs_pos_last - obs_pos_last_m1).reshape(1, 2)
        vec2 = np.concatenate([np.ones(shape=(1, 1)), np.zeros(shape=(1, 1))], axis=1)

        x1 = vec1[:, 0]
        y1 = vec1[:, 1]
        x2 = vec2[:, 0]
        y2 = vec2[:, 1]

        dot = y1 * y2 + x1 * x2  # dot product
        det = y1 * x2 - x1 * y2  # determinant

        heading = np.arctan2(det, dot)
        if (y1 < 0):
            heading = 2 * math.pi - 1 * np.abs(heading)  # from 0 to 2*pi

        if (self.track_id == 'EGO'):
            self.heading_traj = 0.0
        else:
            self.heading_traj = heading

    def trajectory_preprocessing(self):

        if (True in np.isnan(self.trajectory[:, 0]).tolist()):
            trajectory_avg = self.avgfilter.op(self.trajectory)
            trajectory_hole = self.holefill.op(trajectory_avg)
            trajectory_lin_forward = self.linearmodel.op(trajectory_hole)
            trajectory_lin_backward = np.flip(self.linearmodel.op(np.flip(trajectory_hole, axis=0)), axis=0)

            nan_forward = np.isnan(trajectory_lin_forward)
            trajectory_lin_forward[nan_forward] = trajectory_lin_backward[nan_forward]

            self.trajectory = np.copy(trajectory_lin_forward)


    def trajectory_curvature(self):

        seq_len = self.obs_len + self.pred_len
        nan_pos = np.isnan(self.trajectory[:, 0])
        num_nan = float(np.count_nonzero(nan_pos))

        if (num_nan / float(seq_len) > 0.3):
            self.curvature = 0.0
            self.trajectory_len = 0.0
        else:

            for i in range(seq_len):
                if (~np.isnan(self.trajectory[i, 0])):
                    first_pos = self.trajectory[i, 1:3]
                    first_idx = i
                    break

            for i in range(seq_len-1, -1, -1):
                if (~np.isnan(self.trajectory[i, 0])):
                    last_pos = self.trajectory[i, 1:3]
                    last_idx = i
                    break

            path_dist = np.sqrt(np.sum((last_pos - first_pos)**2))

            valid_trajectory = self.trajectory[first_idx:last_idx + 1, :]
            point_dist = np.sqrt(np.sum((valid_trajectory[1:, 1:3] - valid_trajectory[:-1, 1:3])**2, axis=1))
            nan_pos = np.isnan(point_dist)
            path_length = np.sum(point_dist[~nan_pos])

            self.curvature = (path_length / (path_dist + 1E-10)) - 1.0
            self.trajectory_len = path_length

    def update_status(self):

        # stop/moving decision
        if (self.curvature > 0.3):
            self.status.append('stop')
        elif (self.curvature > 0.1 and self.curvature <= 0.3 and self.trajectory_len < 2):
            self.status.append('stop')
        else:
            self.status.append('moving')

        # turn/lc action decision
        if ('moving' in self.status and self.curvature > 0.001):
            self.status.append('turn')


    def __repr__(self):
        return self.type + '/' + self.track_id
