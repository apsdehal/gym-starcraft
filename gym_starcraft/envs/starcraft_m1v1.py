import numpy as np

from gym import spaces

import torchcraft.Constants as tcc
import gym_starcraft.utils as utils
import gym_starcraft.envs.starcraft_base_env as  sc

# Note: Somehow starcraft return coordinates in unit's x and y
# are normalized by 8, for e.g. 400, 400 is converted to 50, 50 and 50, 50 to 6, 6
DISTANCE_FACTOR = 8

class StarCraftM1v1(sc.StarCraftBaseEnv):
    def __init__(self, args, final_init):
        super(StarCraftM1v1, self).__init__(args.torchcraft_dir, args.bwapi_launcher_path,
                                              args.config_path, args.server_ip,
                                              args.server_port, args.speed,
                                              args.frame_skip, args.set_gui, args.self_play,
                                              args.max_steps, final_init)
        # TODO: Random initialize later
        self.my_unit_pairs = [(0, 1, 400, 400)]
        self.enemy_unit_pairs = [(0, 1, 50, 50)]

        self.max_frame_skip = 10
        self.vision = 5
        self.frame_count = 0
        self.move_steps = ((0, 1), (0, -1), (-1, 0), (1, 0), (0, 0))

    def _action_space(self):
        # Move up, down, left, right, stay, attack agents i to n
        self.nactions = 5 + 1

        # return spaces.Box(np.array(action_low), np.array(action_high))
        return spaces.MultiDiscrete([self.nactions])

    def _observation_space(self):
        # relative_x, relative_y, my_hp, enemy_hp, my_cooldown, enemy_cooldown
        obs_low = [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        obs_high = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        return spaces.Box(np.array(obs_low), np.array(obs_high), dtype=np.float32)

    def _make_commands(self, action):
        cmds = []
        if self.state is None or action is None:
            return cmds

        my_unit = None
        enemy_unit = None

        for unit in self.state.units[0]:
            my_unit = unit

        for unit in self.state.units[1]:
            enemy_unit = unit

        if my_unit is None or enemy_unit is None:
            return cmds

        # print(utils.get_distance(my_unit.x, -my_unit.y, enemy_unit.x, -enemy_unit.y))
        action = action[0]

        if action < len(self.move_steps):
            x2 = my_unit.x + self.move_steps[action][0]
            y2 = my_unit.y + self.move_steps[action][1]

            cmds.append([
                tcc.command_unit, my_unit.id,
                tcc.unitcommandtypes.Move, -1, int(x2), int(y2), -1
            ])
        else:
            agent_id = action - len(self.move_steps)

            distance = utils.get_distance(my_unit.x, -my_unit.y,
                                          enemy_unit.x, -enemy_unit.y)

            if distance < my_unit.groundRange:
                cmds.append([
                    tcc.command_unit_protected, my_unit.id,
                    tcc.unitcommandtypes.Attack_Unit, enemy_unit.id
                ])
        return cmds

    def _has_step_completed(self):
        check = True
        if self.frame_count < self.max_frame_skip:
            check = False
            self.frame_count += 1
        else:
            self.frame_count = 0
        return check

    def _make_observation(self):
        myself = None
        enemy = None

        for unit in self.state.units[0]:
            myself = unit

        for unit in self.state.units[1]:
            enemy = unit

        obs = np.zeros(self.observation_space.shape)

        if myself is not None and enemy is not None:
            obs[0] = (myself.x - enemy.x) / (self.state.map_size[0])
            obs[1] = (myself.y - enemy.y) / (self.state.map_size[0])
            obs[2] = myself.health / myself.max_health
            obs[3] = enemy.health / enemy.max_health
            obs[4] = myself.groundCD / myself.maxCD
            obs[5] = enemy.groundCD / enemy.maxCD

        return obs

    def _compute_reward(self):
        reward = -0.05

        if self.obs[2] > self.obs[3]:
            reward += 1

        if self.obs[2] <= self.obs[3]:
            reward -= 1

        if self._check_done() and not bool(self.state.battle_won):
            reward += -500

        if self._check_done() and bool(self.state.battle_won):
            reward += 1000
            self.episode_wins += 1

        if self.episode_steps == self.max_episode_steps:
            reward += -500

        return np.array([reward])

    def step(self, action):
        return self._step(action)

    def reset(self):
        return self._reset()
