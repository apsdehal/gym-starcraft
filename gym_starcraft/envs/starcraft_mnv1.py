import numpy as np

from gym import spaces

import torchcraft.Constants as tcc
import gym_starcraft.utils as utils
import gym_starcraft.envs.starcraft_base_env as  sc

# Note: Somehow starcraft return coordinates in unit's x and y
# are normalized by 8, for e.g. 400, 400 is converted to 50, 50 and 50, 50 to 6, 6
DISTANCE_FACTOR = 8

# N vs 1 marines, starcraft environment
class StarCraftMNv1(sc.StarCraftBaseEnv):
    TIMESTEP_PENALTY = -0.05

    def __init__(self, args, final_init):
        super(StarCraftMNv1, self).__init__(args.torchcraft_dir, args.bwapi_launcher_path,
                                              args.config_path, args.server_ip,
                                              args.server_port, args.speed,
                                              args.frame_skip, args.set_gui, args.self_play,
                                              args.max_steps, final_init)
        
        self.nagents = args.nagents
        self.my_unit_pairs = [(0, 1, -1, -1) for _ in range(args.nagents)]

        self.enemy_unit_pairs = [(0, 1, -1, -1)]

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

    def _make_commands(self, actions):
        cmds = []
        if self.state is None or actions is None:
            return cmds

        # Hack for case when map is not purely cleaned for frame
        if len(self.state.units[0]) > self.nagents:
            return cmds

        enemy_units = self.state.units[1]
        enemy_unit = None

        for idx, unit in enumerate(self.state.units[0]):
            my_unit = unit
            action = actions[idx]

            if action < len(self.move_steps):
                x2 = my_unit.x + self.move_steps[action][0]
                y2 = my_unit.y + self.move_steps[action][1]

                cmds.append([
                    tcc.command_unit, my_unit.id,
                    tcc.unitcommandtypes.Move, -1, int(x2), int(y2), -1
                ])
            else:
                agent_id = action - len(self.move_steps)

                if len(enemy_units) > agent_id:
                    enemy_unit = enemy_units[agent_id]
                else:
                    enemy_unit = None
                
                if not enemy_unit:
                    continue

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

        for unit in self.state.units[1]:
            enemy = unit

        full_obs = []
        for idx in range(self.nagents):
            if len(self.state.units[0]) > idx:
                myself = self.state.units[0][idx]
            else:
                myself = None

            curr_obs = np.zeros(self.observation_space.shape)

            if myself is not None and enemy is not None:
                curr_obs[0] = (myself.x - enemy.x) / (self.state.map_size[0])
                curr_obs[1] = (myself.y - enemy.y) / (self.state.map_size[0])
                curr_obs[2] = myself.health / myself.max_health
                curr_obs[3] = enemy.health / enemy.max_health
                curr_obs[4] = myself.groundCD / myself.maxCD
                curr_obs[5] = enemy.groundCD / enemy.maxCD
            full_obs.append(curr_obs)

        return np.stack(full_obs)

    def _compute_reward(self):
        reward = np.full(self.nagents, self.TIMESTEP_PENALTY)

        for idx in range(len(self.obs)):
            reward[idx] += self.obs[idx][3] - self.obs[idx][2]
            reward[idx] += -0.1

            if self._check_done() and not bool(self.state.battle_won):
                reward[idx] += -100

            if self._check_done() and bool(self.state.battle_won):
                reward[idx] += +100
                self.episode_wins += 1

            if self.episode_steps == self.max_episode_steps:
                reward[idx] += -100

        return reward

    def step(self, action):
        return self._step(action)

    def reset(self):
        return self._reset()
