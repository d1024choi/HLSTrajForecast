from utils.functions import *

from ArgoverseDataset.argoverse.map_representation.map_api import ArgoverseMap
from ArgoverseDataset.argoverse.utils.frustum_clipping import generate_frustum_planes
from ArgoverseDataset.argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from ArgoverseDataset.argoverse.utils.calibration import CameraConfig, proj_cam_to_uv

class Map:

    def __init__(self, args, exp_type):

        # exp settings
        if (exp_type == 'test'):
            exp_type = 'valid'
        self.exp_type = exp_type
        self.dataset_path = os.path.join(args.dataset_path, self.exp_type)

        # Argoverse data loader
        self.argoverse_loader = ArgoverseTrackingLoader(self.dataset_path)
        self.hdmap = ArgoverseMap()

        # params
        self.map_size = args.map_size
        self.lidar_map_ch_dim = args.lidar_map_ch_dim
        self.num_lidar_sweeps = args.num_lidar_sweeps

        self.min_forward_len = 80
        self.min_backward_len = 10

        self.lidar_sample_period = 10 # Hz

    def transform_pc(self, R, translation, pc):
        pc_trans = pc - translation.reshape(1, 3)
        return np.matmul(R, pc_trans.T).T

    def transform_pc_inv(self, R, translation, pc):
        pc_trans_inv = np.matmul(R, pc.T).T + translation.reshape(1, 3)
        return pc_trans_inv


    def __repr__(self):
        return f"Argoverse Map Helper."


def in_range_points(points, x, y, z, x_range, y_range, z_range):

    points_select = points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], y < y_range[1], z > z_range[0], z < z_range[1]))]
    return np.around(points_select, decimals=2)