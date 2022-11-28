import gym
import torch
from envs.environment import TLCEnv
from world import World
from generator import LaneVehicleGenerator, IntersectionVehicleGenerator
from agent import DQNAgent
from metric import TravelTimeMetric, ThroughputMetric, SpeedScoreMetric, MaxWaitingTimeMetric, WaitingCountMeric
import argparse
import os
import numpy as np
import utils as u
import seaborn as sns
import matplotlib.pyplot as plt


def update_analysis_data(num, data, in_env, in_agents):
    w = in_env.world
    lane_vehicles = w.eng.get_lane_vehicles()
    for ii in range(len(in_agents)):
        cur_in_lanes = in_agents[ii].ob_generator[1].all_in_lanes
        v_num = sum([len(lane_vehicles[ll]) for ll in cur_in_lanes])
        inter_name = in_agents[ii].iid
        data[inter_name][num] = v_num
    return data


class FixedTimeAgent:
    def __init__(self, intersection):
        # self.action_space = action_space
        # self.ob_generator = ob_generator
        # self.reward_generator = reward_generator
        self.intersection = intersection
        self.iid = intersection.id
        return

    def get_action(self):
        inter = self.intersection
        phase = inter.current_phase
        next_phase = (phase + 1) % len(inter.phases)
        return next_phase


# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--config_file', type=str, default='./envs/jinan_3_4/config.json', help='path of config file')
parser.add_argument('--thread', type=int, default=8, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--load_model', action="store_true", default=True)
parser.add_argument('--save_dir', type=str, default="model/dqn_backup_271.53",
                    help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/evaluate", help='directory in which logs should be saved')
parser.add_argument('--network', type=str, default="Full", choices=("FC", "CNN", "Full"),
                    help='Network type used in DQN, fully-connected network with queue_size or CNN with map')
parser.add_argument('--parameters', type=str, default="agent/configs_new_dqn/config.json",
                    help='path to the file with informations about the model')
parser.add_argument('--debug', type=u.str2bool, const=True, nargs='?',
                    default=False, help='When in the debug mode, it will not record logs')

args = parser.parse_args()
logger = u.get_logger(args)

# Config File
parameters = u.get_info_file(args.parameters)
parameters['log_path'] = args.log_dir
action_interval = parameters['action_interval']
yellow_phase_time = parameters['yellow_phase_time']
parameters['network'] = args.network
parameters['epsilon_initial'] = 0.0
parameters['epsilon_min'] = 0.0

# create world
yellow_phase_time = 3
world = World(args.config_file, args.thread, action_interval, yellow_phase_time)

# create agents
device = torch.device("cuda:0")
baseline_agents = []
agents = []
for ind, i in enumerate(world.intersections):
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(DQNAgent(
        action_space,
        [LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None, scale=.025),
         IntersectionVehicleGenerator(parameters, world, i, targets=["vehicle_map"])],
        LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
        i.id,
        parameters,
        world,
        device
    ))
    baseline_agents.append(FixedTimeAgent(i))
    if args.load_model:
        agents[ind].load_model(args.save_dir)
agent_nums = len(agents)

# Create metric
metric = [TravelTimeMetric(world), ThroughputMetric(world), SpeedScoreMetric(world),
          WaitingCountMeric(world), MaxWaitingTimeMetric(world)]
# create env
env = TLCEnv(world, agents, metric)
edges_list = u.edges_index_list(agent_nums, world)


def test(args, agents, env):
    last_obs = env.reset()
    env.eng.set_save_replay(True)
    env.eng.set_replay_file("replay_evaluation.txt")
    episodes_rewards = [0 for _ in agents]
    episodes_decision_num = 0
    i = 0
    history_buffer = [[] for _ in range(agent_nums)]

    action_num = 0
    action_nums = args.steps // action_interval
    inter_data = {}
    for agent_id in range(agent_nums):
        i_name = env.world.intersection_ids[agent_id]
        inter_data[i_name] = np.zeros(action_nums)
    inter_data = update_analysis_data(action_num, inter_data, env, agents)

    actions = None
    for _ in range(action_interval):
        obs_list, rewards, dones, _ = env.step(actions)

        obs_map = [item[1] for item in obs_list]
        obs = [item[0] for item in obs_list]
        # obs_phase = [item[1][0] for item in obs_list]
        # obs_map = [item[1][1] for item in obs_list]

        obs = u.aggregate_obs_screen(obs, world)  # (5, 16), [12 + 4 one hot]
        for j in range(agent_nums):
            history_buffer[j].append(obs[j])  # (12, 5, 16)
        i += 1

    last_obs_map = np.array(obs_map)
    last_hist_obs = np.array(history_buffer).transpose(0, 2, 1, 3)  # (12, 10, 5, 16)->(12, 5, 10, 16)
    last_edge_features = u.get_edge_features_sa(last_obs, world)
    while i < args.steps:
        print(i)
        if i % action_interval == 0:
            action_num += 1
            inter_data = update_analysis_data(action_num, inter_data, env, agents)

            actions = []
            att_weights_all = []
            for agent_id, agent in enumerate(agents):
                if i > args.steps:
                # if i > 3000:
                    action, att_weights = agent.get_action(last_hist_obs[agent_id], last_obs_map[agent_id],
                                                           edges_list[agent_id], last_edge_features[agent_id])
                    actions.append(action)
                    att_weights_all.append(att_weights[0])
                else:
                    action = baseline_agents[agent_id].get_action()
                    actions.append(action)
            rewards_list = []
            history_buffer = [[] for _ in range(agent_nums)]
            for _ in range(action_interval):
                # print(f'step {i}')
                obs_list, rewards, dones, _ = env.step(actions)
                obs_map = [item[1] for item in obs_list]
                obs = [item[0] for item in obs_list]

                rewards_list.append(rewards)
                obs = u.aggregate_obs_screen(obs, world)
                for j in range(agent_nums):
                    history_buffer[j].append(obs[j])
                i += 1

            rewards = np.mean(rewards_list, axis=0)
            last_obs = obs.copy()
            last_hist_obs = np.array(history_buffer).transpose(0, 2, 1, 3).copy()  # (12, 10, 5, 16)->(12, 5, 10, 16)
            last_obs_map = np.array(obs_map).copy()
            last_edge_features = u.get_edge_features_sa(last_obs, world).copy()

            for agent_id, agent in enumerate(agents):
                episodes_rewards[agent_id] += rewards[agent_id]
                episodes_decision_num += 1

    for agent_id, agent in enumerate(agents):
        logger.info("\tagent:{}, mean_episode_reward:{}".format(agent_id,
                                                                episodes_rewards[agent_id] / episodes_decision_num))

    for m in env.metric:
        logger.info(f"\t{m.name}: {m.eval()}")

    plt.figure(figsize=(12, 8))
    custom_palette = sns.color_palette("viridis", agent_nums)
    for iid, k in enumerate(inter_data.keys()):
        episode_queue_size = inter_data[k]
        plt.plot(range(action_nums), episode_queue_size, color=custom_palette[iid], label=k)
    # plt.title('Flow distribution of Fixed-time agent')
    plt.title('Flow distribution of our method')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('Action number')
    plt.ylabel('The number of vehicles')
    plt.savefig(f"flow.png", bbox_inches='tight', transparent=True)
    plt.show()

    for k in inter_data.keys():
        print(k, inter_data[k][-1])
    return


if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    test(args, agents, env)
