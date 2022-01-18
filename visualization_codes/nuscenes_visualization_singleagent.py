from NuscenesDataset.visualization import Visualizer
from utils.functions import *
from helper import load_datasetloader, load_solvers

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', type=int, default=300)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='HLS')
    parser.add_argument('--start_frm_idx', type=int, default=0)
    parser.add_argument('--end_frm_idx', type=int, default=750)
    parser.add_argument('--best_k', type=int, default=15)
    parser.add_argument('--map_size', type=int, default=1024)
    parser.add_argument('--t_skip', type=int, default=1)
    parser.add_argument('--scene_range', type=float, default=70)
    parser.add_argument('--is_target_only', type=int, default=0)

    args = parser.parse_args()
    test(args)

def test(args):

    # parent of working directory is base
    abspath = os.path.dirname(os.path.realpath(__file__))
    os.chdir(Path(abspath).parent.absolute())

    model_type1 = []
    model_type2 = ['HLS']

    # CUDA setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_num))

    # type definition
    long_dtype, float_dtype = get_dtypes(useGPU=True)

    # path to saved network
    folder_name = 'nuscenes_' + args.model_name + '_model' + str(args.exp_id)
    path = os.path.join('./saved_models/', folder_name)

    # load parameter setting
    with open(os.path.join(path, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    saved_args.best_k = args.best_k
    saved_args.batch_size = 4
    saved_args.limit_range = 200
    print_training_info(saved_args)

    # load test data
    data_loader, _ = load_datasetloader(args=saved_args, isTrain=False, dtype=torch.FloatTensor)

    # define & load network
    solver = load_solvers(saved_args, data_loader.num_test_scenes, float_dtype)
    ckp_idx = save_read_latest_checkpoint_num(os.path.join(solver.save_dir), 0, isSave=False)
    solver.load_pretrained_network_params(ckp_idx)
    solver.mode_selection(isTrain=False)

    # evaluation setting
    t_skip = args.t_skip
    obs_len = int(saved_args.past_horizon_seconds * saved_args.target_sample_period)
    pred_len = int(saved_args.future_horizon_seconds * saved_args.target_sample_period)
    saved_args.best_k = args.best_k
    saved_args.batch_size = 1
    obs_len_ = obs_len

    # sub-sample trajs
    target_index_obs = np.array([-1 * _ for _ in range(0, obs_len, t_skip)])[::-1] + (obs_len-1)
    target_index_pred = np.array([_ for _ in range(t_skip - 1, pred_len, t_skip)])

    obs_len = len(target_index_obs)
    pred_len = len(target_index_pred)

    # scene range
    map_size = args.map_size
    x_range = (-1 * args.scene_range, args.scene_range)
    y_range = (-1 * args.scene_range, args.scene_range)
    z_range = (-3, 2)

    # visualizer
    vs = Visualizer(args=saved_args, map=data_loader.map,  x_range=x_range, y_range=y_range,
                    z_range=z_range, map_size=map_size, obs_len=obs_len, pred_len=pred_len)


    dataset_len = data_loader.num_test_scenes
    if (args.end_frm_idx > dataset_len):
        args.end_frm_idx = dataset_len

    target_sa = create_target_scene_agent_list()
    target_scenes = np.unique(target_sa[:, 0]).tolist()

    for current_frame_idx in range(args.start_frm_idx, args.end_frm_idx):

        if (args.is_target_only == 1 and current_frame_idx not in target_scenes):
            continue

        # data loading
        data = data_loader.next_sample(current_frame_idx, mode='test')

        # inference
        obs_traj, future_traj, pred_trajs, agent_ids, scene, valid_scene_flag = solver.test(data, float_dtype, args.best_k)

        if (valid_scene_flag == False):
            print(">> [Skip] current scene : %d" % current_frame_idx)
            continue
        print(">> current scene : %d" % current_frame_idx)

        if (args.model_name in model_type2):
            agent_samples, scene = scene
            obs_traj, future_traj, pred_trajs = data_loader.convert_to_egocentric(obs_traj, future_traj, pred_trajs, agent_ids, agent_samples)

        obs_traj_valid = obs_traj[target_index_obs, :, :]
        future_traj_valid = future_traj[target_index_pred, :, :]
        overall_traj = np.concatenate([obs_traj_valid, future_traj_valid], axis=0)
        pred_trajs_valid = pred_trajs[:, target_index_pred, :, :]


        num_agents = agent_ids.shape[0]
        for a in range(num_agents):

            if (args.is_target_only == 1 and chk_if_target(target_sa, current_frame_idx, a) == False):
                continue

            a_token = scene.id_2_token_lookup[agent_ids[a]]
            agent = scene.agent_dict[a_token]

            #  draw point cloud topivew
            lidar_now_token = scene.lidar_token_seq[obs_len_ - 1]
            fig, ax = vs.topview_pc(lidar_now_token, IsEmpty=True)

            # draw hdmap
            scene_location = scene.city_name
            ax = vs.topview_hdmap(ax, lidar_now_token, scene_location, x_range, y_range, map_size, agent=agent, IsAgentOnly=True, BestMatchLaneOnly=False)

            # draw bbox
            for n in range(num_agents):
                n_token = scene.id_2_token_lookup[agent_ids[n]]
                neighbor = scene.agent_dict[n_token]

                if (n != a):
                    xy = neighbor.trajectory[obs_len_ - 1, 1:3].reshape(1, 2)
                    ax = vs.topview_bbox(ax, neighbor, xy, (0.5, 0.5, 0.5))
            xy = agent.trajectory[obs_len_ - 1, 1:3].reshape(1, 2)
            ax = vs.topview_bbox(ax, agent, xy, (1, 0, 0))

            # draw traj
            gt_traj = overall_traj[:, a, :]
            for k in range(args.best_k):
                est_traj = pred_trajs_valid[k, :, a, :]
                ax = vs.topview_trajectory(ax, gt_traj, est_traj)
            plt.axis([0, map_size, 0, map_size])

            img = vs.fig_to_nparray(fig, ax)
            text = '[Scene %d - Agent %d]' % (current_frame_idx, a)
            cv2.putText(img, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

            cv2.imshow('', img)
            cv2.waitKey(0)
            plt.close()


def create_target_scene_agent_list():

    target_scene_agent = [[0, 1]]
    return np.array(target_scene_agent)


def chk_if_target(target_sa, sid, aid):

    target_s = target_sa[target_sa[:,0] == sid]
    if (aid in target_s[:, 1].tolist()):
        return True
    else:
        return False

if __name__ == '__main__':
    main()


