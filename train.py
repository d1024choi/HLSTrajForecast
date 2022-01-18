from utils.libraries import *
from utils.functions import get_dtypes, print_training_info, print_current_train_progress, read_config
import argumentparser as ap
from helper import load_datasetloader, load_solvers

def main():

    model_name = read_config()['model_name']
    args = getattr(ap, model_name)(ap.parser)
    args.model_name = model_name

    train(args)


def train(args):

    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_num))

    # dtype define
    _, float_dtype = get_dtypes(useGPU=True)

    # checks if there is a pre-defined training settings
    folder_name = args.dataset_type + '_' + args.model_name + '_model' + str(args.exp_id)
    save_dir = os.path.join('./saved_models/', folder_name)
    if save_dir != '' and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        args.load_pretrained = 0

    # load pre-defined training settings or save current settings
    start_epoch = args.start_epoch
    if (args.load_pretrained == 1):
        with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
            args = pickle.load(f)
            args.load_pretrained = 1
    else:
        with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(args, f)
    print_training_info(args)


    # prepare training data
    dataset_loader, data_loader = load_datasetloader(args=args, dtype=torch.FloatTensor)

    # define network
    solver = load_solvers(args, dataset_loader.num_train_scenes, float_dtype)

    # training and validation
    num_batches = int(dataset_loader.num_train_scenes / args.batch_size)
    for e in range(start_epoch, args.num_epochs):

        # ------------------------------------------
        # Training
        # ------------------------------------------

        # # turn on train mode
        solver.mode_selection(isTrain=True)

        if (solver.apply_lr_scheduling == 1):
            solver.learning_rate_step(e)

        start = time.time()
        for b, data in enumerate(data_loader):

            start_batch = time.time()

            solver.train(data)

            end_batch = time.time()
            print_current_train_progress(e, b, num_batches, (end_batch-start_batch))

        end = time.time()
        time_left = (end - start) * (args.num_epochs - e - 1) / 3600.0

        solver.normalize_loss_tracker()
        solver.print_status(e, time_left)
        solver.init_loss_tracker()


        # ------------------------------------------
        # Evaluation
        # ------------------------------------------
        if (e % int(args.save_every) == 0):
            solver.eval(dataset_loader, e)



if __name__ == '__main__':
    main()

