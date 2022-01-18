from utils.functions import *


class Visualizer:

    def __init__(self, args, map, x_range, y_range, z_range, map_size, obs_len, pred_len):

        self.map = map

        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.map_size = map_size

        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        self.scale_y = float(map_size - 1) / axis_range_y
        self.scale_x = float(map_size - 1) / axis_range_x

        self.obs_len = obs_len
        self.pred_len = pred_len

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


    # ------------------------------------
    # Point cloud
    # ------------------------------------
    def topview_pc(self, idx=None):

        img = 255 * np.ones(shape=(self.map_size, self.map_size, 3))

        fig, ax = plt.subplots()
        ax.imshow(img.astype('float') / 255.0, extent=[0, self.map_size, 0, self.map_size])

        return fig, ax

    # ------------------------------------
    # HD map
    # ------------------------------------

    def to_global(self, Rtog, trans_g, input):

        return np.matmul(Rtog, input.T).T + trans_g

    def to_agent(self, Rtoa, trans_g, input):

        return np.matmul(Rtoa, (input - trans_g).T).T

    def topview_hdmap(self, ax, agent, city_name):

        Rg2a = agent.Rg2a
        trans_g = agent.trans_g

        # current location of ego-vehicle
        xcenter, ycenter = agent.trans_g[0, 0], agent.trans_g[0, 1]

        # find lanes
        search_range = float(np.max([(self.x_range[1] - self.x_range[0]) / 2, (self.y_range[1] - self.y_range[0]) / 2]))
        local_center_lines, local_center_line_ids = self.map.find_local_lane_centerlines_with_ids(xcenter, ycenter, city_name,  query_search_range_manhattan=search_range * 1.5)

        # current_pose = argoverse_data.get_pose(idx)
        for i in range(local_center_lines.shape[0]):
            cur_lane_id = local_center_line_ids[i]

            # # lane polygons ------------------------
            cur_polygons = self.map.city_to_lane_polygons_ids_dict[city_name][cur_lane_id][:, :2]
            cur_polygons = self.to_agent(Rg2a, trans_g, cur_polygons)
            if (np.count_nonzero(np.isnan(cur_polygons[:, 0])) > 0):
                continue

            ax = self.draw_polygons(ax, cur_polygons, self.scale_x, self.scale_y, self.color_road_seg, 0.3)

            # lane centerlines ---------------------
            cur_lane_seg = local_center_lines[i][:, :2]
            cur_lane_seg = self.to_agent(Rg2a, trans_g, cur_lane_seg)
            if (np.count_nonzero(np.isnan(cur_lane_seg[:, 0])) == 0):
                ax = self.draw_centerlines(ax, cur_lane_seg, self.scale_x, self.scale_y)

        path_list = self.map.get_cl_from_lane_seq(agent.possible_lanes, city_name)
        for path in path_list:
            path_agent_centric = self.to_agent(Rg2a, trans_g, path)
            ax = self.draw_centerlines_agentcentric(ax, path_agent_centric, self.scale_x, self.scale_y)

        return ax

    def draw_centerlines(self, ax, cur_lane_seg, scale_x, scale_y):
        col_pels = -(cur_lane_seg[:, 1] * scale_y).astype(np.int32)
        row_pels = -(cur_lane_seg[:, 0] * scale_x).astype(np.int32)

        col_pels += int(np.trunc(self.y_range[1] * scale_y))
        row_pels += int(np.trunc(self.x_range[1] * scale_x))

        ax.plot(col_pels, self.map_size - row_pels, '-', linewidth=1.0, color=self.color_centerline, alpha=1)

        return ax

    def draw_centerlines_agentcentric(self, ax, cur_lane_seg, scale_x, scale_y):
        col_pels = -(cur_lane_seg[:, 1] * scale_y).astype(np.int32)
        row_pels = -(cur_lane_seg[:, 0] * scale_x).astype(np.int32)

        col_pels += int(np.trunc(self.y_range[1] * scale_y))
        row_pels += int(np.trunc(self.x_range[1] * scale_x))

        ax.plot(col_pels, self.map_size - row_pels, '-', linewidth=1.0, color=(1, 0, 0), alpha=1)

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

        return ax


    # ------------------------------------
    # ETC
    # ------------------------------------

    def topview_bbox(self, ax, agent, incolor):

        obj_3d_bbox = agent.bbox
        color = self.color_bbox

        if (incolor != None):
            color = incolor

        # to topview image domain
        col_pels = -(obj_3d_bbox[:, 1] * self.scale_y).astype(np.int32)
        row_pels = -(obj_3d_bbox[:, 0] * self.scale_x).astype(np.int32)

        col_pels += int(np.trunc(self.y_range[1] * self.scale_y))
        row_pels += int(np.trunc(self.x_range[1] * self.scale_x))

        row_pels = self.map_size - row_pels

        line_col = [col_pels[6], col_pels[2]]
        line_row = [row_pels[6], row_pels[2]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[3], col_pels[2]]
        line_row = [row_pels[3], row_pels[2]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[6], col_pels[7]]
        line_row = [row_pels[6], row_pels[7]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[3], col_pels[7]]
        line_row = [row_pels[3], row_pels[7]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[7], (col_pels[3] + col_pels[2]) / 2]
        line_row = [row_pels[7], (row_pels[3] + row_pels[2]) / 2]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[6], (col_pels[3] + col_pels[2]) / 2]
        line_row = [row_pels[6], (row_pels[3] + row_pels[2]) / 2]
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

            for t in range(self.pred_len):
                r, g, b = self.palette[t]
                ax.plot(col_pels[t], row_pels[t], 's', linewidth=1.0, color=(r, g, b), alpha=0.5)

        # GT trajs --------------------------------------
        col_pels = -(gt[:, 1] * self.scale_y).astype(np.int32)
        row_pels = -(gt[:, 0] * self.scale_x).astype(np.int32)

        col_pels += int(np.trunc(self.y_range[1] * self.scale_y))
        row_pels += int(np.trunc(self.x_range[1] * self.scale_x))
        row_pels = self.map_size - row_pels

        ax.plot(col_pels[self.obs_len - 1:], row_pels[self.obs_len - 1:], 'o-', linewidth=1.0, color=(0, 0, 0), alpha=1)
        ax.plot(col_pels[:self.obs_len], row_pels[:self.obs_len], 'o', linewidth=1.0, color=(0.5, 0.5, 0.5), alpha=1)

        return ax

    def topview_traj_distribution(self, ax, pred):

        best_k, pred_len, _ = pred.shape

        # for displaying images
        axis_range_y = self.y_range[1] - self.y_range[0]
        axis_range_x = self.x_range[1] - self.x_range[0]
        scale_y = float(self.map_size - 1) / axis_range_y
        scale_x = float(self.map_size - 1) / axis_range_x

        # for s in range(pred_len):
        #     cur_t = pred[:, s, :]
        #
        #     col_pels = -(cur_t[:, 1] * scale_y)
        #     row_pels = -(cur_t[:, 0] * scale_x)
        #
        #     col_pels += y_range[1] * scale_y
        #     row_pels += x_range[1] * scale_x
        #     row_pels = map_size - row_pels
        #
        #     sns.kdeplot(x=col_pels, y=row_pels, cmap="Reds", shade=True, bw_adjust=.5, ax=ax)

        pred_ = pred.reshape(best_k * pred_len, 2)
        col_pels = -(pred_[:, 1] * scale_y)
        row_pels = -(pred_[:, 0] * scale_x)

        col_pels += self.y_range[1] * scale_y
        row_pels += self.x_range[1] * scale_x
        row_pels = self.map_size - row_pels
        sns.kdeplot(x=col_pels, y=row_pels, cmap="Reds", shade=True, bw_adjust=.5, ax=ax)

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

