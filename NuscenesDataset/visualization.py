from utils.functions import *
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box

from NuscenesDataset.nuscenes.utils.geometry_utils import transform_matrix
import NuscenesDataset.nuscenes.utils.data_classes as dc
from NuscenesDataset.nuscenes.map_expansion.arcline_path_utils import discretize_lane


class Visualizer:

    def __init__(self, args, map, x_range, y_range, z_range, map_size, obs_len, pred_len):

        self.args = args
        self.nusc = map.nusc
        self.map = map
        self.model_mode = args.model_mode

        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.map_size = map_size
        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        self.scale_y = float(map_size - 1) / axis_range_y
        self.scale_x = float(map_size - 1) / axis_range_x

        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)

        self.dpi = 80
        self.color_centerline = (0, 0, 0)
        self.color_bbox = (0.25, 0.25, 0.25)

        self.color_drivable_space = 'lightgray'
        self.color_road_seg = 'lightgray'
        self.color_road_seg_int = 'lightgray'
        self.color_ped_cross = 'yellow'
        self.color_walkway = 'yellowgreen'
        self.color_stop = 'magenta'

        self.palette = make_palette(pred_len)


    # -------------------------------
    # TOPVIEW LIDAR
    # -------------------------------
    def load_point_cloud_file(self, token):
        lidar_sample_data = self.nusc.get('sample_data', token)
        pcl_path = os.path.join(self.nusc.dataroot, lidar_sample_data['filename'])
        pc = dc.LidarPointCloud.from_file(pcl_path).points[:3]
        return pc

    def transform_pc(self, R, pc):
        pc = np.matmul(R, np.concatenate([pc, np.ones(shape=(1, pc.shape[1]))], axis=0))[:3]
        return pc

    def pc_loader(self, token):


        # Current sample data
        current_sd = self.nusc.get('sample_data', token)

        # Load up the pointcloud.
        pc = self.load_point_cloud_file(current_sd['token'])

        # From sensor coordinate frame to ego car frame.
        current_cs = self.nusc.get('calibrated_sensor', current_sd['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs['translation'], pyquaternion.Quaternion(current_cs['rotation']), inverse=False)

        pc = self.transform_pc(car_from_current, pc)

        return pc

    def topview_pc(self, token, IsEmpty=False):

        if (IsEmpty):
            img = 255 * np.ones(shape=(self.map_size, self.map_size, 3))
        else:
            # load point cloud
            pc = self.pc_loader(token)
            x = pc[0, :]
            y = pc[1, :]
            z = pc[2, :]

            # extract in-range points
            x_lim = in_range_points(x, x, y, z, self.x_range, self.y_range, self.z_range)
            y_lim = in_range_points(y, x, y, z, self.x_range, self.y_range, self.z_range)

            # to pixel domain
            col_img = -(y_lim * self.scale_y).astype(np.int32)
            row_img = -(x_lim * self.scale_x).astype(np.int32)

            col_img += int(np.trunc(self.y_range[1] * self.scale_y))
            row_img += int(np.trunc(self.x_range[1] * self.scale_x))

            img = 255 * np.ones(shape=(self.map_size, self.map_size, 3))

            ch0 = np.copy(img[:, :, 0])
            ch1 = np.copy(img[:, :, 1])
            ch2 = np.copy(img[:, :, 2])

            ch0[row_img.astype('int32'), col_img.astype('int32')] = 64 * np.ones_like(x_lim)
            ch1[row_img.astype('int32'), col_img.astype('int32')] = 64 * np.ones_like(x_lim)
            ch2[row_img.astype('int32'), col_img.astype('int32')] = 64 * np.ones_like(x_lim)

            # array to img
            img[:, :, 0] = ch0.astype('uint8')
            img[:, :, 1] = ch1.astype('uint8')
            img[:, :, 2] = ch2.astype('uint8')

        fig, ax = plt.subplots()
        ax.imshow(img.astype('float') / 255.0, extent=[0, self.map_size, 0, self.map_size])

        return fig, ax


    # -------------------------------
    # TOPVIEW HDMAP
    # -------------------------------
    def draw_centerlines(self, ax, Rinv, xyz, x_range, y_range, map_size, scene_location):

        def draw_lines_on_topview(ax, line, x_range, y_range, map_size, color):

            axis_range_y = y_range[1] - y_range[0]
            axis_range_x = x_range[1] - x_range[0]
            scale_y = float(map_size - 1) / axis_range_y
            scale_x = float(map_size - 1) / axis_range_x

            col_pels = -(line[:, 1] * scale_y).astype(np.int32)
            row_pels = -(line[:, 0] * scale_x).astype(np.int32)

            col_pels += int(np.trunc(y_range[1] * scale_y))
            row_pels += int(np.trunc(x_range[1] * scale_x))

            ax.plot(col_pels, self.map_size - row_pels, '-', linewidth=1.0, color=color, alpha=1)

            return ax


        pose_lists = self.map.centerlines[scene_location]

        w_x_min = xyz[0] + x_range[0]
        w_x_max = xyz[0] + x_range[1]
        w_y_min = xyz[1] + y_range[0]
        w_y_max = xyz[1] + y_range[1]
        win_min_max = (w_x_min, w_y_min, w_x_max, w_y_max)

        for l in range(len(pose_lists)):

            cur_lane = pose_lists[l]
            l_x_max = np.max(cur_lane[:, 0])
            l_x_min = np.min(cur_lane[:, 0])
            l_y_max = np.max(cur_lane[:, 1])
            l_y_min = np.min(cur_lane[:, 1])

            lane_min_max = (l_x_min, l_y_min, l_x_max, l_y_max)

            if (correspondance_check(win_min_max, lane_min_max) == True):
                cur_lane = self.transform_pc(Rinv, cur_lane.T).T
                ax = draw_lines_on_topview(ax, cur_lane[:, :2], x_range, y_range, map_size=map_size, color=self.color_centerline)

        return ax

    def draw_centerlines_agents(self, ax, cur_lane, Rinv, xyz, x_range, y_range, map_size, scene_location):

        def draw_lines_on_topview(ax, line, x_range, y_range, map_size, color):

            axis_range_y = y_range[1] - y_range[0]
            axis_range_x = x_range[1] - x_range[0]
            scale_y = float(map_size - 1) / axis_range_y
            scale_x = float(map_size - 1) / axis_range_x

            col_pels = -(line[:, 1] * scale_y).astype(np.int32)
            row_pels = -(line[:, 0] * scale_x).astype(np.int32)

            col_pels += int(np.trunc(y_range[1] * scale_y))
            row_pels += int(np.trunc(x_range[1] * scale_x))

            ax.plot(col_pels, self.map_size - row_pels, '-', linewidth=1.0, color=color, alpha=1)

            return ax

        cur_lane = self.transform_pc(Rinv, cur_lane.T).T
        ax = draw_lines_on_topview(ax, cur_lane[:, :2], x_range, y_range, map_size=map_size,
                                   color='red')


        return ax

    def draw_polygons(self, ax, cur_polygons, scale_x, scale_y, facecolor, alpha):
        col_pels = -(cur_polygons[:, 1] * scale_y).astype(np.int32)
        row_pels = -(cur_polygons[:, 0] * scale_x).astype(np.int32)

        col_pels += int(np.trunc(self.y_range[1] * scale_y))
        row_pels += int(np.trunc(self.x_range[1] * scale_x))

        col_pels[col_pels < 0] = 0
        col_pels[col_pels > self.map_size - 1] = self.map_size - 1

        row_pels[row_pels < 0] = 0
        row_pels[row_pels > self.map_size - 1] = self.map_size - 1

        contours = np.concatenate(
            [col_pels.reshape(cur_polygons.shape[0], 1), self.map_size - row_pels.reshape(cur_polygons.shape[0], 1)], axis=1)

        ax.add_patch(
            patches.Polygon(
                contours,
                closed=True,
                facecolor=facecolor,
                alpha=alpha
            ))

        #lightgray
        return ax

    def draw_drivable_space(self, ax, xyz, yaw, x_range, y_range, scene_location, alpha):

        # target area
        patch_box = (xyz[0], xyz[1], x_range[1]-x_range[0], y_range[1]-y_range[0])
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch_angle = -np.rad2deg(yaw)
        target_patch = self.map.nusc_maps[scene_location].explorer.get_patch_coord(patch_box, patch_angle)

        # corresponding records
        layer_name = 'drivable_area'
        records = getattr(self.map.nusc_maps[scene_location], layer_name)


        polygon_list = []
        for record in records:
            polygons = [self.map.nusc_maps[scene_location].extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]

            for polygon in polygons:
                new_polygon = polygon.intersection(target_patch)
                if not new_polygon.is_empty:

                    new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    new_polygon = affinity.rotate(new_polygon, patch_angle, origin=(0, 0), use_radians=False)

                    if new_polygon.geom_type is 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)


        # for i in range(len(polygon_list)):
        for _, polygon in enumerate(polygon_list):
            # polygon_points = self.map.nusc_maps[scene_location].explorer.polygon_points(polygon_list[i])[0]
            polygon_points = self.map.nusc_maps[scene_location].explorer.polygon_points(polygon)[0]
            ax = self.draw_polygons(ax, polygon_points, self.scale_x, self.scale_y, self.color_drivable_space, alpha)

        return ax

    def draw_road_segment(self, ax, xyz, yaw, x_range, y_range, scene_location, alpha):

        # target area
        patch_box = (xyz[0], xyz[1], x_range[1]-x_range[0], y_range[1]-y_range[0])
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch_angle = -np.rad2deg(yaw)
        target_patch = self.map.nusc_maps[scene_location].explorer.get_patch_coord(patch_box, patch_angle)

        # corresponding records
        layer_name = 'road_segment'
        records = getattr(self.map.nusc_maps[scene_location], layer_name)


        polygon_list = []
        polygon_list_intersection = []
        for record in records:
            polygon = self.map.nusc_maps[scene_location].extract_polygon(record['polygon_token'])

            if polygon.is_valid:
                new_polygon = polygon.intersection(target_patch)
                if not new_polygon.is_empty:

                    new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    new_polygon = affinity.rotate(new_polygon, patch_angle, origin=(0, 0), use_radians=False)

                    if new_polygon.geom_type is 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])

                    if (record['is_intersection']):
                        polygon_list_intersection.append(new_polygon)
                    else:
                        polygon_list.append(new_polygon)


        # for i in range(len(polygon_list)):
        for _, polygon in enumerate(polygon_list):
            polygon_points = self.map.nusc_maps[scene_location].explorer.polygon_points(polygon)[0]
            ax = self.draw_polygons(ax, polygon_points, self.scale_x, self.scale_y, self.color_road_seg, alpha)

        # for i in range(len(polygon_list_intersection)):
        for _, polygon in enumerate(polygon_list_intersection):
            polygon_points = self.map.nusc_maps[scene_location].explorer.polygon_points(polygon)[0]
            ax = self.draw_polygons(ax, polygon_points, self.scale_x, self.scale_y, self.color_road_seg_int, alpha)

        return ax

    def draw_lane(self, ax, xyz, yaw, x_range, y_range, scene_location):

        # target area
        patch_box = (xyz[0], xyz[1], x_range[1]-x_range[0], y_range[1]-y_range[0])
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch_angle = -np.rad2deg(yaw)
        target_patch = self.map.nusc_maps[scene_location].explorer.get_patch_coord(patch_box, patch_angle)

        # corresponding records
        layer_name = 'lane'
        records = getattr(self.map.nusc_maps[scene_location], layer_name)


        polygon_list = []
        polygon_list_car = []
        for record in records:
            polygon = self.map.nusc_maps[scene_location].extract_polygon(record['polygon_token'])

            if polygon.is_valid:
                new_polygon = polygon.intersection(target_patch)
                if not new_polygon.is_empty:

                    new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    new_polygon = affinity.rotate(new_polygon, patch_angle, origin=(0, 0), use_radians=False)

                    if new_polygon.geom_type is 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])

                    if (record['lane_type']=='CAR'):
                        polygon_list_car.append(new_polygon)
                    else:
                        polygon_list.append(new_polygon)


        for i in range(len(polygon_list)):
            polygon_points = self.map.nusc_maps[scene_location].explorer.polygon_points(polygon_list[i])[0]
            ax = self.draw_polygons(ax, polygon_points, self.scale_x, self.scale_y)

        for i in range(len(polygon_list_car)):
            polygon_points = self.map.nusc_maps[scene_location].explorer.polygon_points(polygon_list_car[i])[0]
            ax = self.draw_polygons(ax, polygon_points, self.scale_x, self.scale_y)

        return ax

    def draw_others(self, ax, xyz, yaw, x_range, y_range, scene_location, layer_name, facecolor, alpha):

        # target area
        patch_box = (xyz[0], xyz[1], x_range[1]-x_range[0], y_range[1]-y_range[0])
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch_angle = -np.rad2deg(yaw)
        target_patch = self.map.nusc_maps[scene_location].explorer.get_patch_coord(patch_box, patch_angle)

        # corresponding records
        records = getattr(self.map.nusc_maps[scene_location], layer_name)

        polygon_list = []
        for record in records:
            polygon = self.map.nusc_maps[scene_location].extract_polygon(record['polygon_token'])

            if polygon.is_valid:
                new_polygon = polygon.intersection(target_patch)
                if not new_polygon.is_empty:

                    new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    new_polygon = affinity.rotate(new_polygon, patch_angle, origin=(0, 0), use_radians=False)

                    if new_polygon.geom_type is 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])

                    '''                    
                    Cues for stop_line_type of "PED_CROSSING" or "TURN_STOP" are ped_crossing records.
                    Cues for stop_line_type of TRAFFIC_LIGHT" are traffic_light records.
                    No cues for stop_line_type of "STOP_SIGN" or "YIELD".
                    
                    * https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/map_expansion_tutorial.ipynb
                    '''

                    if (layer_name == 'stop_line'):
                        if (record['stop_line_type'] == 'TRAFFIC_LIGHT' and len(record['traffic_light_tokens']) > 0):
                            polygon_list.append(new_polygon)
                    else:
                        polygon_list.append(new_polygon)

        # for i in range(len(polygon_list)):
        for _, polygon in enumerate(polygon_list):
            polygon_points = self.map.nusc_maps[scene_location].explorer.polygon_points(polygon)[0]
            ax = self.draw_polygons(ax, polygon_points, self.scale_x, self.scale_y, facecolor, alpha)

        return ax

    def topview_hdmap(self, ax, lidar_now_token, scene_location, x_range, y_range, map_size, agent=None, IsAgentOnly=False, BestMatchLaneOnly=False):

        # ego-pose
        lidar_now_data = self.nusc.get('sample_data', lidar_now_token)
        ego_pose = self.nusc.get('ego_pose', lidar_now_data['ego_pose_token'])
        xyz = ego_pose['translation']
        R = transform_matrix(ego_pose['translation'], pyquaternion.Quaternion(ego_pose['rotation']), inverse=False)
        Rinv = transform_matrix(ego_pose['translation'], pyquaternion.Quaternion(ego_pose['rotation']), inverse=True)
        v = np.dot(R[:3, :3], np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])

        # drivable space (map_size x map_size x 1)
        ax = self.draw_drivable_space(ax, xyz, yaw, x_range, y_range, scene_location, 0.3)

        # road segment (map_size x map_size x 2)
        ax = self.draw_road_segment(ax, xyz, yaw, x_range, y_range, scene_location, 0.3)

        # others (map_size x map_size x 1)
        ax = self.draw_others(ax, xyz, yaw, x_range, y_range, scene_location, 'ped_crossing', self.color_ped_cross, 0.3)
        ax = self.draw_others(ax, xyz, yaw, x_range, y_range, scene_location, 'walkway', self.color_walkway, 0.3)
        ax = self.draw_others(ax, xyz, yaw, x_range, y_range, scene_location, 'stop_line', self.color_stop, 0.3)

        # centerlines (map_size x map_size x 1)
        ax = self.draw_centerlines(ax, Rinv, xyz, x_range, y_range, map_size, scene_location)

        if (IsAgentOnly):
            if (BestMatchLaneOnly):
                try:
                    target_lanes = [agent.possible_lanes[0]]
                    # target_lanes = [agent.possible_lanes[2]]
                except:
                    target_lanes = []

            else:
                target_lanes = copy.deepcopy(agent.possible_lanes)

            for i in range(len(target_lanes)):
                possible_lane = target_lanes[i]
                for _, tok in enumerate(possible_lane):
                    lane_record = self.map.nusc_maps[scene_location].get_arcline_path(tok)
                    coords = np.array(discretize_lane(lane_record, resolution_meters=0.25))
                    ax = self.draw_centerlines_agents(ax, coords, Rinv, xyz, x_range, y_range, map_size, scene_location)


        return ax


    # -------------------------------
    # BBOX and TRAJECTORY
    # -------------------------------
    def topview_bbox(self, ax, agent, xy, incolor):


        bbox = agent.bbox()
        R = make_rot_matrix_from_yaw(agent.yaw)
        bbox = np.matmul(R, bbox.T).T + xy  # 4x2

        color = self.color_bbox
        if (incolor != None):
            color = incolor

        # to topview image domain
        col_pels = -(bbox[:, 1] * self.scale_y).astype(np.int32)
        row_pels = -(bbox[:, 0] * self.scale_x).astype(np.int32)

        col_pels += int(np.trunc(self.y_range[1] * self.scale_y))
        row_pels += int(np.trunc(self.x_range[1] * self.scale_x))

        row_pels = self.map_size - row_pels

        line_col = [col_pels[0], col_pels[1]]
        line_row = [row_pels[0], row_pels[1]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[1], col_pels[2]]
        line_row = [row_pels[1], row_pels[2]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[2], col_pels[3]]
        line_row = [row_pels[2], row_pels[3]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[3], col_pels[0]]
        line_row = [row_pels[3], row_pels[0]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[1], col_pels[4]]
        line_row = [row_pels[1], row_pels[4]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[2], col_pels[4]]
        line_row = [row_pels[2], row_pels[4]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        return ax

    def topview_trajectory(self, ax, gt, pred):
        '''
        gt : seq_len x 2
        '''

        if (len(pred) > 0):
            # Predicted trajs ----------------------------------
            col_pels = -(pred[:, 1] * self.scale_y).astype(np.int32)
            row_pels = -(pred[:, 0] * self.scale_x).astype(np.int32)

            col_pels += int(np.trunc(self.y_range[1] * self.scale_y))
            row_pels += int(np.trunc(self.x_range[1] * self.scale_x))
            row_pels = self.map_size - row_pels

            # update, 211007
            if (self.model_mode == 'vehicle'):
                for t in range(self.pred_len):
                    r, g, b = self.palette[t]
                    ax.plot(col_pels[t], row_pels[t], 's', linewidth=1.0, color=(r, g, b), alpha=1.0)
            else:
                ax.plot(col_pels, row_pels, '.-', linewidth=1.0, color=(1, 0, 0),
                        alpha=0.5)

        # GT trajs --------------------------------------
        col_pels = -(gt[:, 1] * self.scale_y).astype(np.int32)
        row_pels = -(gt[:, 0] * self.scale_x).astype(np.int32)

        col_pels += int(np.trunc(self.y_range[1] * self.scale_y))
        row_pels += int(np.trunc(self.x_range[1] * self.scale_x))
        row_pels = self.map_size - row_pels

        # update, 211007
        if (self.model_mode == 'vehicle'):
            ax.plot(col_pels[self.obs_len - 1:], row_pels[self.obs_len - 1:], 'o-', linewidth=1.0, color=(0, 0, 0), alpha=0.5)
            ax.plot(col_pels[:self.obs_len], row_pels[:self.obs_len], 'o', linewidth=1.0, color=(0.5, 0.5, 0.5), alpha=0.5)
        else:
            ax.plot(col_pels[self.obs_len - 1:], row_pels[self.obs_len - 1:], '-', linewidth=1.0, color=(0, 0, 0), alpha=0.5)
            ax.plot(col_pels[:self.obs_len], row_pels[:self.obs_len], '-', linewidth=1.0, color=(0.5, 0.5, 0.5), alpha=0.5)

        return ax

    def fig_to_nparray(self, fig, ax):

        fig.set_size_inches(self.map_size / self.dpi, self.map_size / self.dpi)
        ax.set_axis_off()

        fig.canvas.draw()
        render_fig = np.array(fig.canvas.renderer._renderer)

        final_img = np.zeros_like(render_fig[:, :, :3]) # 450, 1600
        final_img[:, :, 2] = render_fig[:, :, 0]
        final_img[:, :, 1] = render_fig[:, :, 1]
        final_img[:, :, 0] = render_fig[:, :, 2]

        plt.close()

        return final_img


def read_and_resize_img(path, img_size):
    img = cv2.imread(path)
    return cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)

def in_range_points(points, x, y, z, x_range, y_range, z_range):

    points_select = points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], y < y_range[1], z > z_range[0], z < z_range[1]))]
    return np.around(points_select, decimals=2)

def make_palette(pred_len):
    red = np.array([1, 0, 0])
    orange = np.array([1, 0.5, 0])
    yellow = np.array([1, 1.0, 0])
    green = np.array([0, 1.0, 0])
    blue = np.array([0, 0, 1])
    colors = [red, orange, yellow, green, blue]

    palette = []
    for t in range(pred_len):

        cur_pos = 4.0 * float(t) / float(pred_len - 1)  # pred_len -> 0 ~ 4
        prev_pos = int(cur_pos)
        next_pos = int(cur_pos) + 1

        if (next_pos > 4):
            next_pos = 4

        prev_color = colors[prev_pos]
        next_color = colors[next_pos]

        prev_w = float(next_pos) - cur_pos
        next_w = 1 - prev_w

        cur_color = prev_w * prev_color + next_w * next_color
        palette.append(cur_color)

    return palette

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

def make_rot_matrix_from_yaw(yaw):

    '''
    yaw in radian
    '''

    m_cos = np.cos(yaw)
    m_sin = np.sin(yaw)
    m_R = [[m_cos, -1 * m_sin], [m_sin, m_cos]]

    return np.array(m_R)
