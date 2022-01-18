# nuScenes dev-kit.
# Code written by Freddy Boulton.

import json
import os
from itertools import chain
from typing import List

from nuscenes.utils.splits import create_splits_scenes

NUM_IN_TRAIN_VAL = 200


def get_prediction_challenge_split(split: str, dataroot: str = '/data/sets/nuscenes') -> List[str]:
    """
    Gets a list of {instance_token}_{sample_token} strings for each split.
    :param split: One of 'mini_train', 'mini_val', 'train', 'val'.
    :param dataroot: Path to the nuScenes dataset.
    :return: List of tokens belonging to the split. Format {instance_token}_{sample_token}.
    """
    if split not in {'mini_train', 'mini_val', 'train', 'train_val', 'val'}:
        raise ValueError("split must be one of (mini_train, mini_val, train, train_val, val)")
    
    if split == 'train':
        split_name = 'train'
    else:
        split_name = 'val'

    path_to_file = os.path.join(dataroot, "maps", "prediction", "prediction_scenes.json")
    prediction_scenes = json.load(open(path_to_file, "r"))
    scenes = create_splits_scenes()
    scenes_for_split = scenes[split_name]

    scene_blacklist = ['scene-0499', 'scene-0515', 'scene-0517']
    for _, err_scene in enumerate(scene_blacklist):
        if (err_scene in scenes_for_split):
            scenes_for_split.remove(err_scene)

    token_list_for_scenes = map(lambda scene: prediction_scenes.get(scene, []), scenes_for_split)

    return list(chain.from_iterable(token_list_for_scenes))
