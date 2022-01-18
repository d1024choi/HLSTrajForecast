import argparse

parser = argparse.ArgumentParser()

# Exp Info
parser.add_argument('--model_name', type=str, default='')
# -----------------------------------
# 'vehicle' or 'pedestrian'
# -----------------------------------
parser.add_argument('--model_mode', type=str, default='vehicle') # update, 211001
parser.add_argument('--exp_id', type=int, default=300)
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--load_pretrained', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--multi_gpu', type=int, default=1)
parser.add_argument('--num_cores', type=int, default=1)


# Dataset
parser.add_argument('--dataset_path', type=str, default='')
# -----------------------------------
# 'argoverse' or 'nuscenes'
# -----------------------------------
parser.add_argument('--dataset_type', type=str, default='nuscenes')
parser.add_argument('--preprocess_trajectory', type=int, default=0)
parser.add_argument('--num_turn_scene_repeats', type=int, default=0)
parser.add_argument('--input_dim', type=int, default=2)
parser.add_argument('--scene_accept_prob', type=float, default=0.33)
# ----------------------------------------------------
#                         |  argoverse  |  nuscenes  |
# ----------------------------------------------------
# past_horizon_seconds    |     2       |     2      |
# future_horizon_seconds  |     3       |     6      |
# target_sample_period    |     5       |     2      |
# ----------------------------------------------------
parser.add_argument('--past_horizon_seconds', type=float, default=2)
parser.add_argument('--future_horizon_seconds', type=float, default=6)
parser.add_argument('--target_sample_period', type=float, default=2)  # Hz
parser.add_argument('--min_past_horizon_seconds', type=float, default=1.5)
parser.add_argument('--min_future_horizon_seconds', type=float, default=3)
parser.add_argument('--val_ratio', type=float, default=0.05)
parser.add_argument('--max_num_agents', type=int, default=100)
parser.add_argument('--min_num_agents', type=int, default=2)
parser.add_argument('--stop_agents_remove_prob', type=float, default=0)
parser.add_argument('--limit_range_change_prob', type=float, default=0.0)
# -----------------------------------
# 0 : vehicle (Trajectron++)
# 1 : vehicle (Nuscenes benchmark)
# 2 : vehicle and pedestrian (Trajectron++)
# -----------------------------------
parser.add_argument('--category_filtering_method', type=int, default=0)

# Training Env
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--best_k', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--min_learning_rate', type=float, default=0.00001)
parser.add_argument('--learning_rate_cnn', type=float, default=0.00005)
parser.add_argument('--grad_clip', type=float, default=0.0)
parser.add_argument('--n_cycle', type=int, default=4)
parser.add_argument('--warmup_ratio', type=float, default=0.5)

parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--kappa', type=float, default=1.0)

parser.add_argument('--valid_step', type=int, default=1)
parser.add_argument('--save_every', type=int, default=3)
parser.add_argument('--max_num_chkpts', type=int, default=5)

parser.add_argument('--apply_cyclic_schedule', type=int, default=0)
parser.add_argument('--separate_lr_for_cnn', type=int, default=0)
parser.add_argument('--apply_lr_scheduling', type=int, default=0)

# ----------------------------------------------------
#  limit_range     |  argoverse  |  nuscenes  |
# ----------------------------------------------------
#  train           |     200     |     50     |
#  test            |     200     |    200     |
# ----------------------------------------------------
parser.add_argument('--limit_range', type=int, default=30)


def HLS(parser):


    parser.add_argument('--is_random_path_order', type=int, default=1)
    parser.add_argument('--is_train_dis', type=int, default=1)

    parser.add_argument('--path_resol', type=float, default=1.0)
    parser.add_argument('--max_path_len_forward', type=float, default=80)
    parser.add_argument('--max_path_len_backward', type=float, default=10)
    parser.add_argument('--ngh_dist_thr', type=float, default=5)

    parser.add_argument('--num_max_paths', type=int, default=10)
    parser.add_argument('--lane_feat_dim', type=int, default=64)

    parser.add_argument('--pos_emb_dim', type=int, default=16)
    parser.add_argument('--traj_enc_h_dim', type=int, default=16)
    parser.add_argument('--traj_dec_h_dim', type=int, default=128)

    parser.add_argument('--gan_prior_prob', type=float, default=0.5)
    parser.add_argument('--z_dim', type=int, default=16)

    args = parser.parse_args()

    return args
