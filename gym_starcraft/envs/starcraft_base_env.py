import gym

import torchcraft.Constants as tcc
import random
import yaml
import subprocess
import sys
import time
import socket, errno
import os
import signal
import atexit
import uuid
import gym_starcraft.utils as utils
import tempfile


DISTANCE_FACTOR = 8
class StarCraftBaseEnv(gym.Env):
    def __init__(self, torchcraft_dir='~/TorchCraft',
                 bwapi_launcher_path=os.path.join(os.environ["BWAPI_INSTALL_PREFIX"], 'bin/BWAPILauncher'),
                 config_path='~/gym-starcraft/gym_starcraft/envs/config.yml',
                 server_ip='127.0.0.1',
                 server_port=11111,
                 ai_type='builtin',
                 speed=0, frame_skip=1, set_gui=0, self_play=0,
                 max_episode_steps=200, final_init=True):

        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

        if not final_init:
            return

        self.config_path = config_path
        self.torchcraft_dir = torchcraft_dir
        self.bwapi_launcher_path = bwapi_launcher_path
        self.server_ip = server_ip
        self.self_play = self_play
        self.frame_skip = frame_skip
        self.speed = speed
        self.max_episode_steps = max_episode_steps
        self.set_gui = set_gui
        self.frame_count = 0
        self.ai_type = ai_type


        config = None
        with open(self.config_path, 'r') as f:
            try:
                config = yaml.load(f)
            except yaml.YAMLError as err:
                print('Config yaml error', err)
                sys.exit(0)

        cmds = []

        tmpfile = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        options = dict(os.environ)
        for key, val in config['options'].items():
            options[key] = str(val)


        options['BWAPI_CONFIG_AUTO_MENU__GAME_TYPE'] = "USE MAP SETTINGS"
        options['BWAPI_CONFIG_AUTO_MENU__AUTO_RESTART'] = "ON"
        options['BWAPI_CONFIG_AUTO_MENU__AUTO_MENU'] = "LAN"
        options['OPENBW_LAN_MODE'] = "LOCAL"
        options['OPENBW_LOCAL_PATH'] = tmpfile

        cmds.append(self.bwapi_launcher_path)
        print(options)

        proc1 = subprocess.Popen(cmds,
                                cwd=os.path.expanduser(self.torchcraft_dir),
                                env=options,
                                stdout=subprocess.PIPE
                                )
        self._register_kill_at_exit(proc1)

        proc2 = subprocess.Popen(cmds,
                                cwd=os.path.expanduser(self.torchcraft_dir),
                                env=options,
                                stdout=subprocess.PIPE
                                )
        self._register_kill_at_exit(proc2)

        matchstr = b"TorchCraft server listening on port "
        for line in iter(proc1.stdout.readline, ''):
            if len(line) != 0:
                print(line.rstrip().decode('utf-8'))
            if line[:len(matchstr)] == matchstr:
                self.server_port1 = int(line[len(matchstr):].strip())
                break
        for line in iter(proc2.stdout.readline, ''):
            if len(line) != 0:
                print(line.rstrip().decode('utf-8'))
            if line[:len(matchstr)] == matchstr:
                self.server_port2 = int(line[len(matchstr):].strip())
                break

        self.episodes = 0
        self.episode_wins = 0
        self.episode_steps = 0
        self.map = 'maps/micro/micro-empty2.scm'
        self.first_reset = True

        # TODO: Move them in proper setters and getters
        # NOTE: These should be overrided in derived class
        # Should be a list of pairs where each pair is
        # (quantity, unit_type, x, y, start_coordinate, end_coordinate)
        # So (1, 0, -1, -1, 100, 150) will instantiate 0 type unit
        # at a random place between x = (100, 150) and y = (100, 150)
        # Leave empty if you want to instantiate anywhere in whole map
        self.nagents = 1
        self.vision = 3
        self.nenemies = 1
        self.my_unit_pairs = []
        self.enemy_unit_pairs = []

        self.my_current_units = {}
        self.enemy_current_units = {}
        self.agent_ids = []
        self.enemy_ids = []
        self.state1 = None
        self.obs = None
        self.obs_pre = None
        self.stat = {}

    def init_conn(self):

        import torchcraft as tc
        self.client1 = tc.Client()
        self.client1.connect(self.server_ip, self.server_port1)
        self.state1 = self.client1.init()

        self.client2 = tc.Client()
        self.client2.connect(self.server_ip, self.server_port2)
        self.state2 = self.client2.init()

        setup = [[tcc.set_combine_frames, 1],
                 [tcc.set_speed, self.speed],
                 [tcc.set_gui, self.set_gui],
                 # NOTE: We use custom frameskip method now
                 # Skip frame below
                 [tcc.set_frameskip, 1],
                 [tcc.set_cmd_optim, 1]]

        self.client1.send(setup)
        self.state1 = self.client1.recv()
        self.client2.send(setup)
        self.state2 = self.client2.recv()

    def __del__(self):
        if hasattr(self, 'client') and self.client1:
            self.client1.close()

    def _register_kill_at_exit(self, proc):
        atexit.register(proc.kill)

    def _kill_child(self, child_pid):
        if child_pid is None:
            pass
        else:
            os.kill(child_pid, signal.SIGTERM)

    def _step(self, action):

        # Stop stepping if map config has come into play
        if len(self.state1.aliveUnits.values()) > self.nagents + self.nenemies:
            reward = self._compute_reward()
            self.my_current_units = {}
            self.obs = self._make_observation()
            done = True
            info = {}
            return self.obs, reward, done, info

        self.episode_steps += 1

        self.client1.send(self._make_commands(action))
        self.state1 = self.client1.recv()
        self.client2.send(self._get_enemy_commands())
        self.state2 = self.client2.recv()

        self._skip_frames()

        while not self._has_step_completed():
            self._empty_step()

        self.obs = self._make_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = self._get_info()

        self._update_stat()
        self.obs_pre = self.obs
        return self.obs, reward, done, info

    def _empty_step(self):
        self.client1.send([])
        self.state1 = self.client1.recv()
        self.client2.send([])
        self.state2 = self.client2.recv()

    def _skip_frames(self):
        count = 0
        while count < self.frame_skip:
            self._empty_step()
            count += 1

    def _get_enemy_commands(self):
        cmds = []
        func = lambda *args: None

        if self.ai_type == 'attack_closest':
            func = utils.get_closest
        elif self.ai_type == 'attack_weakest':
            func = utils.get_weakest

        for unit in self.state2.units[self.state2.player_id]:
            opp_unit = func(unit, self.state2, self.state1.player_id)
            dist = utils.get_distance(opp_unit.x, opp_unit.y, unit.x, unit.y)
            vision = tcc.staticvalues['sightRange'][unit.type] / DISTANCE_FACTOR

            if dist > vision:
                continue
            if opp_unit is not None:
                cmds.append([
                    tcc.command_unit_protected, unit.id,
                    tcc.unitcommandtypes.Attack_Unit, opp_unit.id
                ])

        return cmds

    def try_killing(self):
        if not self.state1:
            return

        while len(self.state1.units[self.state1.player_id]) != 0 \
              or len(self.state2.units[self.state2.player_id]) != 0:
            c1units = self.state1.units[self.state1.player_id]
            c2units = self.state2.units[self.state2.player_id]

            self.client1.send(self.kill_units(c1units))
            self.state1 = self.client1.recv()

            self.client2.send(self.kill_units(c2units))
            self.state2 = self.client2.recv()

            for i in range(10):
                self._empty_step()


    def _reset(self):
        wins = self.episode_wins
        episodes = self.episodes


        # print("Episodes: %4d | Wins: %4d | WinRate: %1.3f" % (
        #         episodes, wins, wins / (episodes + 1E-6)))

        self.episodes += 1
        self.episode_steps = 0

        if self.first_reset:
            self.init_conn()
            self.first_reset = False

        self.try_killing()

        c1 = []
        c2 = []

        for unit_pair in self.my_unit_pairs:
            c1 += self._get_create_units_command(self.state1.player_id, unit_pair)

        for unit_pair in self.enemy_unit_pairs:
            c2 += self._get_create_units_command(self.state2.player_id, unit_pair)

        self.client1.send(c1)
        self.state1 = self.client1.recv()
        self.client2.send(c2)
        self.state2 = self.client2.recv()

        while len(self.state1.units.get(self.state1.player_id, [])) == 0 \
              and len(self.state2.units.get(self.state2.player_id, [])) == 0:
            self._empty_step()

        # This adds my_units and enemy_units to object.
        self.my_current_units = self._parse_units_to_unit_dict(self.state1.units[self.state1.player_id])
        self.enemy_current_units = self._parse_units_to_unit_dict(self.state2.units[self.state2.player_id])

        # This adds my and enemy's units' ids as incrementing list
        self.agent_ids = list(self.my_current_units)
        self.enemy_ids = list(self.enemy_current_units)
        self.stat = {}

        self.obs = self._make_observation()
        self.obs_pre = self.obs
        return self.obs

    def _get_create_units_command(self, player_id, unit_pair):
        defaults = [1, 100, 100, 0, self.state1.map_size[0] - 10][len(unit_pair) - 1:]
        unit_type, quantity, x, y, start, end = (list(unit_pair) + defaults)[:6]

        return self.create_units(player_id, quantity, x=x, y=y,
                                 unit_type=unit_type, start=start,
                                 end=end)

    def create_units(self, player_id, quantity, unit_type=0, x=100, y=100, start=0, end=256):
        if player_id == self.state1.player_id:
            max_coord = (end - start) // 2 - self.vision // 4
            min_coord = 0
        else:
            max_coord = (end - start)
            min_coord = (end - start) // 2 + self.vision // 4

        if x < 0:
            x = (random.randint(min_coord, max_coord) + start) * DISTANCE_FACTOR

        if y < 0:
            y = (random.randint(min_coord, max_coord) + start) * DISTANCE_FACTOR
        commands = []

        for _ in range(quantity):
            command = [
                tcc.command_openbw,
                tcc.openbwcommandtypes.SpawnUnit,
                player_id,
                unit_type,
                x,
                y,
            ]
            commands.append(command)

        return commands

    def is_empty(self, data):
        return data is not None and len(data) == 0

    def kill_units(self, units):
        commands = []

        for u in units:
            command = [
                tcc.command_openbw,
                tcc.openbwcommandtypes.KillUnit,
                u.id
            ]
            commands.append(command)
        return commands


    def _parse_units_to_unit_dict(self, units, units_type='my_units'):
        unit_dict = dict()

        for unit in units:
            unit_dict[unit.id] = unit

        return unit_dict

    def _action_space(self):
        """Returns a space object"""
        raise NotImplementedError

    def _observation_space(self):
        """Returns a space object"""
        raise NotImplementedError

    def _make_commands(self, action):
        """Returns a game command list based on the action"""
        raise NotImplementedError

    def _make_observation(self):
        """Returns a observation object based on the game state"""
        raise NotImplementedError

    def _has_step_completed(self):
        """Returns a boolean to tell whether the current step has
        actually completed in the game"""
        raise NotImplementedError

    def _compute_reward(self):
        """Returns a computed scalar value based on the game state"""
        raise NotImplementedError

    def _check_done(self):
        """Returns true if the episode was ended"""
        return (
                # bool(self.state1.game_ended) or
                # self.state1.battle_just_ended or
                len(self.state1.units[self.state1.player_id]) == 0 or \
                len(self.state2.units[self.state2.player_id]) == 0 or \
                self.episode_steps == self.max_episode_steps)

    def _has_won(self):
        return (
            len(self.state1.units[self.state1.player_id]) > 0 and \
            len(self.state2.units[self.state2.player_id]) == 0
        )

    def _get_info(self):
        """Returns a dictionary contains debug info"""
        return {}

    def _update_stat(self):
        if self._check_done():
            if self._has_won():
                self.stat['success'] = 1
            else:
                self.stat['success'] = 0

            self.stat['steps_taken'] = self.episode_steps

        return self.stat

    def render(self, mode='human', close=False):
        pass
