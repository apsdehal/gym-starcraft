import argparse

from gym_starcraft.envs.starcraft_mnv1 import StarCraftMNv1
import os


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default='nv1',
                        help='Type of starcraft task')
    parser.add_argument('--torchcraft_dir', type=str, default='~/TorchCraft',
                        help='TorchCraft directory')
    parser.add_argument('--bwapi_launcher_path', type=str,
                        default=os.path.join(os.environ["BWAPI_INSTALL_PREFIX"], 'bin/BWAPILauncher'),
                        help='Path to BWAPILauncher')
    parser.add_argument('--config_path', type=str,
                        default='../gym-starcraft/gym_starcraft/envs/config.yml',
                        help='Path to TorchCraft/OpenBW yml config')
    parser.add_argument('--server_ip', type=str, default='127.0.0.1',
                        help='IP of the server')
    parser.add_argument('--server_port', type=int, default=11111,
                        help='Port of the server')
    parser.add_argument('--speed', type=int, default=0,
                        help='Speed')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='Frame skip')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Number of maximum steps in episode')
    parser.add_argument('--nagents', type=int, default=1,
                        help='Number of agents')
    parser.add_argument('--set_gui', action="store_true", default=False,
                        help="Show GUI")
    parser.add_argument('--self_play', action='store_true', default=False,
                        help='Should play with self')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    env = StarCraftMNv1(args, final_init=True)

    agent = RandomAgent(env.action_space)
    episodes = 0

    while episodes < 50:
        obs = env.reset()
        done = False
        while not done:
            actions = []

            for _ in range(args.nagents):
                action = agent.act()[0]
                actions.append(action)
            obs, reward, done, info = env.step(actions)
        episodes += 1
        print(reward)

    env.close()
