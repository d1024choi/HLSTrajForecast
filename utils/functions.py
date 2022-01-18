from utils.libraries import *

def read_config():
    with open('./config/config.json', 'r') as f:
        json_data = json.load(f)
    return json_data

def get_dtypes(useGPU=True):

    if (useGPU):
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    else:
        long_dtype = torch.LongTensor
        float_dtype = torch.FloatTensor

    return long_dtype, float_dtype

def init_weights(m):

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def toNP(x):

    return x.detach().to('cpu').numpy()

def toTS(x, dtype):

    return torch.from_numpy(x).to(dtype)

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

def save_read_latest_checkpoint_num(path, val, isSave):

    file_name = path + '/checkpoint.txt'
    index = 0

    if (isSave):
        file = open(file_name, "w")
        file.write(str(int(val)) + '\n')
        file.close()
    else:
        if (os.path.exists(file_name) == False):
            print('[Error] there is no such file in the directory')
            sys.exit()
        else:
            f = open(file_name, 'r')
            line = f.readline()
            index = int(line[:line.find('\n')])
            f.close()

    return index

def read_all_saved_param_idx(path):
    ckp_idx_list = []
    files = sorted(glob.glob(os.path.join(path, '*.pt')))
    for i, file_name in enumerate(files):
        start_idx = 0
        for j in range(-3, -10, -1):
            if (file_name[j] == '_'):
                start_idx = j+1
                break
        ckp_idx = int(file_name[start_idx:-3])
        ckp_idx_list.append(ckp_idx)
    return ckp_idx_list[::-1]

def copy_chkpt_every_N_epoch(args):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index

    root_path = args.model_dir + str(args.exp_id)
    save_directory =  root_path + '/copies'
    if save_directory != '' and not os.path.exists(save_directory):
        os.makedirs(save_directory)

    fname_list = []
    fnum_list = []
    all_file_names = os.listdir(root_path)
    for fname in all_file_names:
        if "saved" in fname:
            chk_index = get_file_number(fname)
            fname_list.append(fname)
            fnum_list.append(chk_index)

    max_idx = np.argmax(np.array(fnum_list))
    target_file = fname_list[max_idx]

    src = root_path + '/' + target_file
    dst = save_directory + '/' + target_file
    shutil.copy2(src, dst)

    print(">> {%s} is copied to {%s}" % (target_file, save_directory))

def remove_past_checkpoint(path, max_num=5):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index


    num_remain = max_num - 1
    fname_list = []
    fnum_list = []

    all_file_names = os.listdir(path)
    for fname in all_file_names:
        if "saved" in fname:
            chk_index = get_file_number(fname)
            fname_list.append(fname)
            fnum_list.append(chk_index)

    if (len(fname_list)>num_remain):
        sort_results = np.argsort(np.array(fnum_list))
        for i in range(len(fname_list)-num_remain):
            del_file_name = fname_list[sort_results[i]]
            os.remove('./' + path + '/' + del_file_name)

def print_current_train_progress(e, b, num_batchs, time_spent):

    if b == num_batchs-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r [Epoch %02d] %d / %d (%.4f sec/sample)' % (e, b, num_batchs, time_spent)),

    sys.stdout.flush()

def print_current_valid_progress(b, num_batchs):

    if b >= num_batchs-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r >> validation process (%d / %d) ' % (b, num_batchs)),

    sys.stdout.flush()

def print_current_test_progress(b, num_batchs):

    if b >= num_batchs-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r >> test process (%d / %d) ' % (b, num_batchs)),

    sys.stdout.flush()

def print_voxelization_progress(b, num_batchs):

    if (b >= num_batchs-1):
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r >> voxelization process (%d / %d) ' % (b, num_batchs)),

    sys.stdout.flush()

def print_training_info(args):

    print("--------- %s / %s (%s) ----------" % (args.dataset_type, args.model_name, args.model_mode))
    print(" Exp id : %d" % args.exp_id)
    print(" Gpu num : %d" % args.gpu_num)
    print(" Num epoch : %d" % args.num_epochs)
    print(" Batch size : %d" % args.batch_size)
    print(" best_k : %d" % args.best_k)
    print(" alpha/beta/gamma : %.4f/%.4f/%.4f" % (args.alpha, args.beta, args.gamma))
    print(" initial learning rate : %.5f " % args.learning_rate)
    print(" gradient clip : %.4f" % (args.grad_clip))
    print("         -----------          ")
    print(" Past horizon seconds (obs_len): %.1f sec (%d)" % (args.past_horizon_seconds, args.past_horizon_seconds*args.target_sample_period))
    print(" Future horizon seconds (pred_len) : %.1f sec (%d) "  % (args.future_horizon_seconds, args.future_horizon_seconds*args.target_sample_period))
    print(" Target sample period : %.1f Hz" % args.target_sample_period)
    print("         -----------          ")
    print(" LR scheduling (min lr : %.5f): %d" % (args.min_learning_rate, args.apply_lr_scheduling))
    print(" Separate LR for CNN (lr: %.5f) : %d" % (args.learning_rate_cnn, args.separate_lr_for_cnn))

    if (args.model_name == 'HLS'):
        print("         -----------          ")
        print(" Limit range (m) : %d" % args.limit_range)
        print(" Neighbor Dist. Thr (m) : %.2f" % args.ngh_dist_thr)
        print(" Num possible paths : %d" % args.num_max_paths)
        print(" Random path order : %d" % args.is_random_path_order)
        print(" Train Dis (GAN prior prob. %.2f) : %d" % (args.gan_prior_prob, args.is_train_dis))
        print(" Forward path length (m) : %.1f" % args.max_path_len_forward)
        print(" Path resolution (m) : %.1f" % args.path_resol)
        print(" Cyclic scheduling for KLD (n_cycle: %d, w_ratio: %.2f) : %d" % (
        args.n_cycle, args.warmup_ratio, args.apply_cyclic_schedule))
    print("----------------------------------")
