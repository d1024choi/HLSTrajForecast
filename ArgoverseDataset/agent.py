from utils.functions import *
from utils.trajectory_filter import LinearModelIntp, AverageFilter, HoleFilling


class Agent:

    def __init__(self, track_id=None, agent_id=None, agent_type=None, obs_len=None, pred_len=None):

        self.track_id = track_id
        self.agent_id = agent_id
        self.agent_type = agent_type

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.trajectory = np.full(shape=(obs_len+pred_len, 4), fill_value=np.nan)

        self.Rg2a = None
        self.trans_g = None
        self.heading = 0.0
        self.heading_from_lane = 0.0
        self.speed_mps = 0.0

        self.possible_lanes = []

        # trajectory filter
        self.avgfilter = AverageFilter(filter_size=3)


    def calc_speed(self, sample_period):

        '''
        sample period (Hz)
        '''

        obs_seq = self.trajectory[:self.obs_len, 1:3]
        obs_pos_last = obs_seq[-1]
        obs_pos_last_m1 = obs_seq[-2]

        vec1 = obs_pos_last - obs_pos_last_m1
        self.speed_mps = sample_period * np.sqrt(np.sum(vec1 ** 2)) # mps


    def rotation_matrix_from_yaw(self, yaw):

        # calculate rotation matrix
        m_cos = np.cos(yaw)
        m_sin = np.sin(yaw)
        Ra2g = np.array([m_cos, -1 * m_sin, m_sin, m_cos]).reshape(2, 2)
        Rg2a = np.linalg.inv(Ra2g)

        return Ra2g, Rg2a


    def heading_from_traj(self):

        # data preprocessing
        avg_traj = self.avgfilter.op(self.trajectory[:, 1:3])
        obs_seq = avg_traj[:self.obs_len, :]

        # calculate heading
        obs_pos_last = obs_seq[-1]
        obs_pos_last_m1 = obs_seq[-2]
        dxdy = (obs_pos_last - obs_pos_last_m1)
        self.heading = np.arctan2(dxdy[1], dxdy[0])  # -1x because of left side is POSITIVE

        # calculate rotation matrix
        self.Ra2g, self.Rg2a = self.rotation_matrix_from_yaw(self.heading)
        # m_cos = np.cos(self.heading)
        # m_sin = np.sin(self.heading)
        # self.Ra2g = np.array([m_cos, -1 * m_sin, m_sin, m_cos]).reshape(2, 2)
        # self.Rg2a = np.linalg.inv(self.Ra2g)

        # translation
        self.trans_g = self.trajectory[self.obs_len-1, 1:3].reshape(1, 2)





    def global_to_agent_centric(self, input):

        '''
        input : seq_len x 2
        '''

        return np.matmul(self.Rg2a, (input - self.trans_g).T).T

    def agent_to_global(self, input):

        '''
        input : seq_len x 2
        '''

        return np.matmul(self.Ra2g, input.T).T + self.trans_g


    def __repr__(self):
        return self.track_id
