from utils.functions import *

class Scene:

    def __init__(self, sample_token=None, lidar_token_seq=None, agent_dict=None, city_name=None):

        self.sample_token = sample_token
        self.agent_dict = agent_dict
        self.lidar_token_seq = lidar_token_seq
        self.num_agents = len(agent_dict)
        self.city_name = city_name


        self.action_status = ['stop', 'moving', 'turn']
        self.num_stop_agents = 0
        self.num_moving_agents = 0
        self.num_turn_agents = 0

    def make_id_2_token_lookup(self):
        self.id_2_token_lookup = {}
        for idx, key in enumerate(self.agent_dict):
            self.id_2_token_lookup[self.agent_dict[key].agent_id] = key

    def calc_agent_status_statistic(self):

        for idx, key in enumerate(self.agent_dict):
            if (self.action_status[0] in self.agent_dict[key].status):
                self.num_stop_agents+=1

            if (self.action_status[1] in self.agent_dict[key].status):
                self.num_moving_agents+=1

            if (self.action_status[2] in self.agent_dict[key].status):
                self.num_turn_agents+=1


    def __repr__(self):
        return f"Sample ID: {self.sample_token}," \
               f" City: {self.city_name}," \
               f" Num agents: {self.num_agents}."


class AgentCentricScene:

    def __init__(self, sample_token=None, agent_token=None, city_name=None):

        self.sample_token = sample_token
        self.agent_token = agent_token
        self.city_name = city_name

        self.trajectories = None
        self.bboxes = None
        self.R_a2g = None
        self.R_g2a = None
        self.trans_g = None
        self.agent_ids = None
        self.target_agent_index = None
        self.possible_lanes = None
        self.best_lane = -1

    def __repr__(self):
        return f"Scene ID: {self.sample_token}," \
               f" City: {self.city_name}," \
               f" Agent ID: {self.agent_token}."