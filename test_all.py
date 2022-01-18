from utils.functions import *
from utils.loss import calc_ED_error
from helper import load_datasetloader, load_solvers

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', type=int, default=1052)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='argoverse')
    parser.add_argument('--model_name', type=str, default='HLS')
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--best_k', type=int, default=12)
    parser.add_argument('--is_test_all', type=int, default=1)
    parser.add_argument('--model_num', type=int, default=0)

    args = parser.parse_args()
    test(args)

def test(args):


    # CUDA setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_num))

    # type definition
    _, float_dtype = get_dtypes(useGPU=True)

    # path to saved network
    folder_name = args.dataset_type + '_' + args.model_name + '_model' + str(args.exp_id)
    path = os.path.join('./saved_models/', folder_name)

    # load parameter setting
    with open(os.path.join(path, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    saved_args.best_k = args.best_k
    saved_args.scene_accept_prob = 1.0 # important
    saved_args.limit_range = 200 # important
    print_training_info(saved_args)

    # load test data
    data_loader, _ = load_datasetloader(args=saved_args, isTrain=False, dtype=torch.FloatTensor)

    # define network
    solver = load_solvers(saved_args, data_loader.num_test_scenes, float_dtype)
    ckp_idx_list = read_all_saved_param_idx(solver.save_dir)

    target_models = []
    if (args.is_test_all == 0):
        target_models.append(args.model_num)
    else:
        target_models += ckp_idx_list

    for _, ckp_id in enumerate(ckp_idx_list):

        if (ckp_id not in target_models):
            print(">> [SKIP] current model %d is not in target model list!" % ckp_id)
            continue

        solver.load_pretrained_network_params(ckp_id)
        solver.mode_selection(isTrain=False)

        ADE1, FDE1, ADE5, FDE5, ADE6, FDE6, ADE10, FDE10, ADEK, FDEK, AADEK, AFDEK = [], [], [], [], [], [], [], [], [], [], [], []
        for b in range(0, len(data_loader.test_data), args.step_size):

            # data loading
            data = data_loader.next_sample(b, mode='test')

            # inference
            obs_traj, future_traj, pred_trajs, agent_ids, scene, valid_scene_flag = solver.test(data, float_dtype, args.best_k)

            if (valid_scene_flag == False):
                continue

            num_agents = pred_trajs.shape[2]
            for o in range(num_agents):

                ADE_k, FDE_k = [], []
                for k in range(args.best_k):
                    error = np.sqrt(np.sum((pred_trajs[k, :, o, :2] - future_traj[:, o, :2])**2, axis=1))
                    ADE_k.append(np.mean(error))
                    FDE_k.append(np.mean(error[-1]))

                    AADEK.append(error)
                    AFDEK.append(error[-1])

                error_ADE, error_FDE = calc_ED_error(o, 1, args.best_k, pred_trajs, future_traj, ADE_k, FDE_k)
                ADE1.append(error_ADE)
                FDE1.append(error_FDE)

                error_ADE, error_FDE = calc_ED_error(o, 5, args.best_k, pred_trajs, future_traj, ADE_k, FDE_k)
                ADE5.append(error_ADE)
                FDE5.append(error_FDE)

                error_ADE, error_FDE = calc_ED_error(o, 6, args.best_k, pred_trajs, future_traj, ADE_k, FDE_k)
                ADE6.append(error_ADE)
                FDE6.append(error_FDE)

                error_ADE, error_FDE = calc_ED_error(o, 10, args.best_k, pred_trajs, future_traj, ADE_k, FDE_k)
                ADE10.append(error_ADE)
                FDE10.append(error_FDE)

                error_ADE, error_FDE = calc_ED_error(o, args.best_k, args.best_k, pred_trajs, future_traj, ADE_k, FDE_k)
                ADEK.append(error_ADE)
                FDEK.append(error_FDE)

            print_current_test_progress(b, len(data_loader.test_data))

        print('--------------------------------------------------------------')
        print(">> ADE1 : %.4f, FDE1 : %.4f" % (np.mean(ADE1), np.mean(FDE1)))
        print(">> ADE5 : %.4f, FDE5 : %.4f" % (np.mean(ADE5), np.mean(FDE5)))
        print(">> ADE6 : %.4f, FDE6 : %.4f" % (np.mean(ADE6), np.mean(FDE6)))
        print(">> ADE10 : %.4f, FDE10 : %.4f" % (np.mean(ADE10), np.mean(FDE10)))
        print(">> ADE%d : %.4f, FDE%d : %.4f" % (args.best_k, np.mean(ADEK), args.best_k, np.mean(FDEK)))
        print(">> AADE%d : %.4f, AFDE%d : %.4f" % (args.best_k, np.mean(AADEK), args.best_k, np.mean(AFDEK)))
        print('--------------------------------------------------------------')

        file_name_txt = 'test_results_%s_model%d_ckpid%d.txt' % (args.dataset_type, args.exp_id, ckp_id)
        file = open(os.path.join(path, file_name_txt), "w")
        file.write('ADE1 :  %s \n' % str(float(int(np.mean(ADE1)*10000))/10000))
        file.write('FDE1 :  %s \n' % str(float(int(np.mean(FDE1) * 10000)) / 10000))
        file.write('ADE5 :  %s \n' % str(float(int(np.mean(ADE5)*10000))/10000))
        file.write('FDE5 :  %s \n' % str(float(int(np.mean(FDE5) * 10000)) / 10000))
        file.write('ADE6 :  %s \n' % str(float(int(np.mean(ADE6)*10000))/10000))
        file.write('FDE6 :  %s \n' % str(float(int(np.mean(FDE6) * 10000)) / 10000))
        file.write('ADE10 :  %s \n' % str(float(int(np.mean(ADE10)*10000))/10000))
        file.write('FDE10 :  %s \n' % str(float(int(np.mean(FDE10) * 10000)) / 10000))
        file.write('ADE%d :  %s \n' % (args.best_k, str(float(int(np.mean(ADEK)*10000))/10000)))
        file.write('FDE%d :  %s \n' % (args.best_k, str(float(int(np.mean(FDEK) * 10000)) / 10000)))
        file.write('AADE%d :  %s \n' % (args.best_k, str(float(int(np.mean(AADEK)*10000))/10000)))
        file.write('AFDE%d :  %s \n' % (args.best_k, str(float(int(np.mean(AFDEK) * 10000)) / 10000)))
        file.close()

if __name__ == '__main__':
    main()


