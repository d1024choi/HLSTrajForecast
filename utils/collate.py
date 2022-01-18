from utils.libraries import *

def seq_collate_typeA(data):


    obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, \
    map, num_neighbors, valid_neighbor, possible_path, lane_label = zip(*data)

    _len = [objs for objs in num_neighbors]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj_ta_cat = torch.cat(obs_traj_ta, dim=1)
    future_traj_ta_cat = torch.cat(future_traj_ta, dim=1)
    obs_traj_ngh_cat = torch.cat(obs_traj_ngh, dim=1)
    future_traj_ngh_cat = torch.cat(future_traj_ngh, dim=1)

    map_cat = torch.cat(map, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    possible_path_cat = torch.cat(possible_path, dim=1)
    lane_label_cat = torch.cat(lane_label, dim=0)

    return tuple([obs_traj_ta_cat, future_traj_ta_cat, obs_traj_ngh_cat, future_traj_ngh_cat,
                  map_cat, seq_start_end, np.array(valid_neighbor), possible_path_cat, lane_label_cat])

