from utils.libraries import *


class Scene:

    def __init__(self, scene_id=None, agent_dict=None, city_name=None):

        self.scene_id = scene_id
        self.agent_dict = agent_dict
        self.num_agents = len(agent_dict)
        self.city_name = city_name


    def make_id_2_token_lookup(self):
        self.id_2_token_lookup = {}
        for idx, key in enumerate(self.agent_dict):
            self.id_2_token_lookup[self.agent_dict[key].agent_id] = key

    def __repr__(self):
        return f"Scene ID: {self.scene_id}," \
               f" City: {self.city_name}," \
               f" Num agents: {self.num_agents}."


class AgentCentricScene:

    def __init__(self, scene_id = None, track_id = None, city_name=None):

        self.scene_id = scene_id
        self.city_name = city_name
        self.track_id = track_id

        self.trajectories = None
        self.R_a2g = None
        self.R_g2a = None
        self.trans_g = None
        self.agent_ids = None
        self.target_agent_index = None


    def __repr__(self):
        return f"Scene ID: {self.scene_id}," \
               f" City: {self.city_name}," \
               f" Agent ID: {self.track_id}."
