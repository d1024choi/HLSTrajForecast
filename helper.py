from utils.libraries import *
from torch.utils.data import DataLoader
from utils.collate import *
from utils.functions import read_config

LOADER_TYPE_A = ['HLS']

def load_datasetloader(args, dtype, isTrain=True):

    config = read_config()

    # --------------------------
    # Argoverse
    # --------------------------
    if (args.dataset_type == 'argoverse'):
        if (args.model_name in LOADER_TYPE_A):
            from ArgoverseDataset.loader_typeA import DatasetLoader
            seq_collate = seq_collate_typeA
        else:
            sys.exit("[Error] No loader type exists for '%s' in 'Argoverse' !!" % args.model_name)
        args.dataset_path = config['Argoverse_Forecasting']['dataset_path']


    # --------------------------
    # nuScenes
    # --------------------------
    elif (args.dataset_type == 'nuscenes'):
        if (args.model_name in LOADER_TYPE_A):
            from NuscenesDataset.loader_typeA import DatasetLoader
            seq_collate = seq_collate_typeA
        else:
            sys.exit("[Error] No loader type exists for '%s' in 'Nuscenes' !!" % args.model_name)
        args.dataset_path = config['Nuscenes']['dataset_path']

    else:
        sys.exit("[Error] '%s' dataset is not supported !!" % args.dataset_path)

    # prepare data
    dataset_loader = DatasetLoader(args=args, isTrain=isTrain, dtype=dtype)

    if (isTrain):
        data_loader = DataLoader(dataset_loader, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_cores, drop_last=True, collate_fn=seq_collate)
        return dataset_loader, data_loader
    else:
        return dataset_loader, 0


def load_solvers(args, num_train_scenes, dtype):

    if (args.model_name == 'HLS'):
        from optimization.HLS_solver import Solver
        return Solver(args, num_train_scenes, dtype)
    else:
        sys.exit("[Error] There is no solver for '%s' !!" % args.model_name)
