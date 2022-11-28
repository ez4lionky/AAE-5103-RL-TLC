import random
import gym
import torch
from tqdm import tqdm
from envs.environment import TLCEnv
from world import World
from generator import LaneVehicleGenerator, PressureRewardGenerator, IntersectionVehicleGenerator
from agent import DQNAgent
from metric import TravelTimeMetric, ThroughputMetric, SpeedScoreMetric, MaxWaitingTimeMetric, WaitingCountMeric
import argparse
import os
import numpy as np
import logging
from datetime import datetime
import utils as u

# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
# torch.cuda.set_device(3)
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
torch.autograd.set_detect_anomaly(True)

# set CUDA and seed
seed = 1
# random.seed(seed)
# np.random.seed(seed)
n_training_threads = 1
if torch.cuda.is_available():
    print("choose to use gpu...")
    device = torch.device("cuda")
    torch.set_num_threads(n_training_threads)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    print("choose to use cpu...")
    device = torch.device("cpu")
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(n_training_threads)

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--config_file', type=str, default='./envs/jinan_3_4/config.json', help='path of config file')
parser.add_argument('--thread', type=int, default=4, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument("--save_rate", type=int, default=20,
                    help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir', type=str, default="model/dqn", help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/dqn", help='directory in which logs should be saved')
parser.add_argument('--network', type=str, default="Full", choices=("FC", "CNN", "Full"),
                    help='Network type used in DQN, fully-connected network with queue_size or CNN with map')
parser.add_argument('--parameters', type=str, default="agent/configs_new_dqn/config.json",
                    help='path to the file with informations about the model')
parser.add_argument('--debug', type=u.str2bool, const=True, nargs='?',
                    default=False, help='When in the debug mode, it will not record logs')
parser.add_argument("--other_params",
                    nargs='*',
                    default=[],
                    type=str)

args = parser.parse_args()
logger = u.get_logger(args)

# Config File
parameters = u.get_info_file(args.parameters)
episodes = parameters['episodes']
learning_start = parameters['learning_start']
update_model_freq = parameters['update_model_freq']
parameters['log_path'] = args.log_dir
action_interval = parameters['action_interval']
yellow_phase_time = parameters['yellow_phase_time']
parameters['network'] = args.network

# start wandb
u.wand_init(args, parameters, "TLC - Results", "new_dqn_ma", "new_dqn_ma", debug=args.debug)
logger.info("env: ")
logger.info(args.config_file)
logger.info("other params: ")
logger.info(args.other_params)
logger.info("args: ")
logger.info(args)
logger.info("parameters: ")
logger.info(parameters)

# create world
world = World(args.config_file, args.thread, action_interval, yellow_phase_time)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(DQNAgent(
        action_space,
        [LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None, scale=.025),
         IntersectionVehicleGenerator(parameters, world, i, targets=["vehicle_map"])],
        # IntersectionVehicleGenerator(parameters, world, i, targets=["current_phase", "vehicle_map"])],
        LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
        # PressureRewardGenerator(world, i, scale=0.005, negative=True),
        i.id,
        parameters,
        world,
        device
    ))
    if args.load_model:
        agents[0].load_model(args.save_dir)
agent_nums = len(agents)

# Create metric
metric = [TravelTimeMetric(world), ThroughputMetric(world), SpeedScoreMetric(world),
          WaitingCountMeric(world), MaxWaitingTimeMetric(world)]

# create env
env = TLCEnv(world, agents, metric)

# * add
TARGET_UPDATE_INTERVAL = 20  # unit: update_count
edges_list = u.edges_index_list(agent_nums, world)


# train dqn_agent
def train(args, env):
    epochs_replay = parameters["epochs_replay"]
    epochs_initial_replay = parameters["epochs_initial_replay"]
    first_replay = True

    total_decision_num = 0
    update_count = 0
    for e in range(episodes):
        last_obs = env.reset()

        if e % args.save_rate == args.save_rate - 1:
            env.eng.set_save_replay(True)
            env.eng.set_replay_file("replay_%s.txt" % e)
        else:
            env.eng.set_save_replay(False)
        episodes_rewards = [0 for _ in agents]
        episodes_decision_num = 0
        i = 0
        history_buffer = [[] for _ in range(agent_nums)]
        actions = None
        for _ in range(action_interval):
            obs_list, rewards, dones, _ = env.step(actions)

            obs_map = [item[1] for item in obs_list]
            obs = [item[0] for item in obs_list]

            obs = u.aggregate_obs_screen(obs, world)  # (12, 10, 5, 16)
            for j in range(agent_nums):
                history_buffer[j].append(obs[j])
            i += 1
        last_obs_map = np.array(obs_map)
        last_hist_obs = np.array(history_buffer).transpose(0, 2, 1, 3)
        last_edge_features = u.get_edge_features_sa(last_obs, world)
        while i < args.steps:  # 3600 steps
            if i % action_interval == 0:
                actions = []
                att_weights_all = []
                for agent_id, agent in enumerate(agents):
                    if total_decision_num > agent.learning_start:
                        action, att_weights = agent.get_action(last_hist_obs[agent_id], last_obs_map[agent_id],
                                                               edges_list[agent_id], last_edge_features[agent_id])
                    else:
                        action, att_weights = agent.sample(last_hist_obs[agent_id])
                    actions.append(action)
                    att_weights_all.append(att_weights[0])

                rewards_list = []
                history_buffer = [[] for _ in range(agent_nums)]
                for _ in range(action_interval):
                    obs_list, rewards, dones, _ = env.step(actions)
                    obs_map = [item[1] for item in obs_list]
                    obs = [item[0] for item in obs_list]

                    rewards_list.append(rewards)
                    obs = u.aggregate_obs_screen(obs, world)
                    for j in range(agent_nums):
                        history_buffer[j].append(obs[j])
                    i += 1
                rewards = np.mean(rewards_list, axis=0)
                obs_map = np.array(obs_map)

                hist_obs = np.array(history_buffer).transpose(0, 2, 1, 3)  # (12, 10, 5, 12)->(12, 5, 10, 12)
                # agg_obs = u.aggregate_obs(obs, world)
                # agg_obs = u.aggregate_obs_screen(last_obs, world)
                # agg_rewards = u.aggregate_rewards(rewards, world)  
                cur_edge_features = u.get_edge_features_sa(last_obs, world)
                for agent_id, agent in enumerate(agents):
                    agent.remember(last_hist_obs[agent_id], last_obs_map[agent_id], actions[agent_id],
                                   rewards[agent_id], hist_obs[agent_id], obs_map[agent_id], edges_list[agent_id],
                                   last_edge_features[agent_id], cur_edge_features[agent_id])
                    episodes_rewards[agent_id] += rewards[agent_id]

                episodes_decision_num += 1
                total_decision_num += 1
                last_obs = obs.copy()
                last_hist_obs = hist_obs.copy()
                last_obs_map = obs_map.copy()
                last_edge_features = cur_edge_features.copy()

                if total_decision_num > learning_start and total_decision_num % update_model_freq == update_model_freq - 1:
                    update_count += 1

                    data_list = []
                    for agent in agents:
                        data_list.append(agent.fetch_data())

                    if first_replay:
                        for epoch in tqdm(range(epochs_initial_replay)):
                            for agent_id, agent in enumerate(agents):
                                agent.new_replay(data_list[agent_id], adjust_epsilon=False)
                        for agent in agents:
                            agent.update_target_network()
                        first_replay = False
                    else:
                        for epoch in range(epochs_replay):
                            for agent_id, agent in enumerate(agents):
                                agent.new_replay(data_list[agent_id])

                        torch.cuda.synchronize()

                    if update_count % TARGET_UPDATE_INTERVAL == 0:
                        for agent_id, agent in enumerate(agents):
                            agent.update_target_network()
            if all(dones):
                break

        ## lr scheduler
        if total_decision_num > learning_start:
            for agent in agents:
                agent.learning_rate_decay()
        logger.info(f'episode:{e}, lr:{agents[0].optimizer.param_groups[0]["lr"]}')

        if e % args.save_rate == args.save_rate - 1:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            # agents[0].save_model(args.save_dir)
            for agent in agents:
                agent.save_model(args.save_dir)

        eval_dict = {}

        logger.info(f"episode:{e}/{episodes - 1}, steps:{i}")
        # eval_dict["episode"] = e
        # eval_dict["steps"] = i
        eval_dict["learning rate"] = agents[0].get_lr()

        for agent_id, agent in enumerate(agents):
            logger.info("\tagent:{}, mean_episode_reward:{}".format(agent_id,
                                                                    np.mean(episodes_rewards[
                                                                                agent_id]) / episodes_decision_num))

        for metric in env.metric:
            logger.info(f"\t{metric.name}: {metric.eval()}")
            eval_dict[metric.name] = metric.eval()

        eval_dict["epsilon"] = agents[0].epsilon
        eval_dict["mean_episode_reward"] = np.mean(episodes_rewards) / episodes_decision_num

        if not args.debug:
            u.wand_log(eval_dict)

    # for agent in agents:
    agents[0].save_model(args.save_dir)


if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    train(args, env)
