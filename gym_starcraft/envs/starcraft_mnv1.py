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
    TIMESTEP_PENALTY = -0.01

    def __init__(self, args, final_init):
        super(StarCraftMNv1, self).__init__(args.torchcraft_dir, args.bwapi_launcher_path,
                                              args.config_path, args.server_ip,
                                              args.server_port, args.speed,
                                              args.frame_skip, args.set_gui, args.self_play,
                                              args.max_steps, final_init)

        self.nagents = args.nagents
        self.my_unit_pairs = [(0, 1, -1, -1, 100, 150) for _ in range(args.nagents)]

        self.enemy_unit_pairs = [(0, 1, -1, -1, 100, 150)]

        self.vision = 7
        self.move_steps = ((0, 1), (0, -1), (-1, 0), (1, 0), (0, 0))

        self.prev_actions = np.zeros(self.nagents)

    def _action_space(self):
        # Move up, down, left, right, stay, attack agents i to n
        self.nactions = 5 + 1

        # return spaces.Box(np.array(action_low), np.array(action_high))
        return spaces.MultiDiscrete([self.nactions])

    def _observation_space(self):
        # relative_x, relative_y, in_vision, my_hp, enemy_hp, my_cooldown, enemy_cooldown
        obs_low = [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        obs_high = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]

        return spaces.Box(np.array(obs_low), np.array(obs_high), dtype=np.float32)

    def _make_commands(self, actions):
        cmds = []
        if self.state1 is None or actions is None:
            return cmds

        # Hack for case when map is not purely cleaned for frame
        if len(self.state1.units[0]) > self.nagents:
            return cmds

        enemy_unit = None

        for idx in range(self.nagents):
            agent_id = self.agent_ids[idx]

            if agent_id not in self.my_current_units:
                # Agent is probably dead
                continue

            my_unit = self.my_current_units[agent_id]
            action = actions[idx]
            prev_action = self.prev_actions[idx]

            if action < len(self.move_steps):
                x2 = my_unit.x + self.move_steps[action][0]
                y2 = my_unit.y + self.move_steps[action][1]

                cmds.append([
                    tcc.command_unit, my_unit.id,
                    tcc.unitcommandtypes.Move, -1, int(x2), int(y2), -1
                ])
            else:
                enemy_id = action - len(self.move_steps)

                if len(self.enemy_ids) > enemy_id:
                    enemy_id = self.enemy_ids[enemy_id]
                    enemy_unit = self.enemy_current_units[enemy_id]
                else:
                    enemy_unit = None

                if not enemy_unit:
                    continue

                distance = utils.get_distance(my_unit.x, -my_unit.y,
                                              enemy_unit.x, -enemy_unit.y)

                unit_command = tcc.command_unit_protected

                if prev_action < len(self.move_steps):
                    unit_command = tcc.command_unit

                if distance <= my_unit.groundRange:
                    cmds.append([
                        unit_command, my_unit.id,
                        tcc.unitcommandtypes.Attack_Unit, enemy_unit.id
                    ])
        self.prev_actions = actions
        return cmds

    def _has_step_completed(self):
        check = True
        return check

    def _make_observation(self):
        myself = None
        enemy = None

        for enemy_id in self.enemy_ids:
            if enemy_id in self.enemy_current_units:
                enemy = self.enemy_current_units[enemy_id]

        full_obs = []

        for idx in range(self.nagents):
            agent_id = self.agent_ids[idx]

            if agent_id in self.my_current_units:
                myself = self.my_current_units[agent_id]
            else:
                myself = None

            curr_obs = np.zeros(self.observation_space.shape)

            if myself is not None and enemy is not None:
                distance = utils.get_distance(myself.x, myself.y, enemy.x, enemy.y)

                if distance < self.vision:
                    curr_obs[0] = (myself.x - enemy.x) / (self.vision)
                    curr_obs[1] = (myself.y - enemy.y) / (self.vision)
                    curr_obs[2] = 0
                else:
                    curr_obs[0] = 0
                    curr_obs[1] = 0
                    curr_obs[2] = 1

                curr_obs[3] = myself.health / myself.max_health
                curr_obs[4] = enemy.health / enemy.max_health
                curr_obs[5] = myself.groundCD / myself.maxCD
                curr_obs[6] = enemy.groundCD / enemy.maxCD
            full_obs.append(curr_obs)

        return np.stack(full_obs)

    def _compute_reward(self):
        reward = np.full(self.nagents, self.TIMESTEP_PENALTY)

        for idx in range(len(self.obs)):
            if self.episode_steps == self.max_episode_steps:
                reward[idx] += 0 - self.obs_pre[idx][3]
            else:
                reward[idx] += self.obs[idx][3] - self.obs_pre[idx][3]
                reward[idx] += self.obs_pre[idx][4] - self.obs[idx][4]

            if self._check_done() and self._has_won() == 1:
                reward[idx] += +10

        if self._check_done() and self._has_won() == 1:
            self.episode_wins += 1
        return reward

    def _get_info(self):
        alive_mask = np.ones(self.nagents)

        for idx in range(self.nagents):
            agent_id = self.agent_ids[idx]

            if agent_id not in self.my_current_units:
                alive_mask[idx] = 0

        return {'alive_mask': alive_mask}

    def step(self, action):
        return self._step(action)

    def reset(self):
        return self._reset()
