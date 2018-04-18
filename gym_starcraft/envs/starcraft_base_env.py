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

DISTANCE_FACTOR = 8
class StarCraftBaseEnv(gym.Env):
    def __init__(self, torchcraft_dir='~/TorchCraft',
                 bwapi_launcher_path='../bwapi/bin/BWAPILauncher',
                 config_path='~/gym-starcraft/gym_starcraft/envs/config.yml',
                 server_ip='127.0.0.1',
                 server_port=11111,
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


        self.server_port = self._find_available_port(server_ip, server_port)

        config = None
        with open(self.config_path, 'r') as f:
            try:
                config = yaml.load(f)
            except yaml.YAMLError as err:
                print('Config yaml error', err)
                sys.exit(0)

        cmds = []

        options = dict(os.environ)
        for key, val in config['options'].items():
            options[key] = str(val)

        options['BWAPI_CONFIG_AUTO_MENU__GAME_TYPE'] = "USE MAP SETTINGS"
        options['BWAPI_CONFIG_AUTO_MENU__AUTO_RESTART'] = "ON"
        options['TORCHCRAFT_PORT'] = str(self.server_port)

        cmds.append(self.bwapi_launcher_path)

        proc = subprocess.Popen(cmds, cwd=os.path.expanduser(self.torchcraft_dir),
                                env=options)
        self._register_kill_at_exit(proc)

        time.sleep(5)

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
        self.my_unit_pairs = []
        self.enemy_unit_pairs = []

        self.my_current_units = {}
        self.enemy_current_units = {}
        self.agent_ids = []
        self.enemy_ids = []
        self.state = None
        self.obs = None
        self.obs_pre = None

    def init_conn(self):

        import torchcraft as tc
        self.client = tc.Client()
        self.client.connect(self.server_ip, self.server_port)
        self.state = self.client.init(micro_battles=True)


        setup = [[tcc.set_combine_frames, 1],
                 [tcc.set_speed, self.speed],
                 [tcc.set_gui, self.set_gui],
                 # NOTE: We use custom frameskip method now
                 # Skip frame below
                 [tcc.set_frameskip, 1],
                 [tcc.set_cmd_optim, 1]]

        self.client.send(setup)
        self.state = self.client.recv()

    def __del__(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()

    def _find_available_port(self, server_ip, server_port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        while True:
            try:
                s.bind((server_ip, server_port))
            except socket.error as e:
                if e.errno == errno.EADDRINUSE:
                    server_port += 1
                    continue
                else:
                    # something else raised the socket.error exception
                    print(e)
            break

        s.close()
        return server_port

    def _register_kill_at_exit(self, proc):
        atexit.register(proc.kill)

    def _kill_child(self, child_pid):
        if child_pid is None:
            pass
        else:
            os.kill(child_pid, signal.SIGTERM)


    def _step(self, action):
        self.episode_steps += 1

        # Stop stepping if map config has come into play
        if len(self.state.aliveUnits.values()) > len(self.my_unit_pairs) + len(self.enemy_unit_pairs):
            reward = self._compute_reward()
            self.my_current_units = {}
            self.obs = self._make_observation()
            done = True
            info = {}
            return self.obs, reward, done, info


        self.client.send(self._make_commands(action))
        self.state = self.client.recv()

        self._skip_frames()

        while not self._has_step_completed():
            self.client.send([])
            self.state = self.client.recv()

        self.my_current_units = self._parse_units_to_unit_dict(self.state.units[0])
        self.enemy_current_units = self._parse_units_to_unit_dict(self.state.units[1])

        self.obs = self._make_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = self._get_info()
        self.obs_pre = self.obs
        return self.obs, reward, done, info

    def _skip_frames(self):
        count = 0
        while count < self.frame_skip:
            self.client.send([])
            self.state = self.client.recv()
            count += 1

    def try_killing(self):
        if not self.state:
            return

        while len(self.state.aliveUnits.values()) != 0:
            command = []
            my_units = self.state.units[0]
            enemy_units = self.state.units[1]
            if len(my_units):
                command += self.kill_units(my_units, len(my_units))

            if len(enemy_units):
                command += self.kill_units(enemy_units, len(enemy_units))

            if len(command):
                self.client.send(command)
                self.state = self.client.recv()


    def _reset(self):
        wins = self.episode_wins
        episodes = self.episodes


        # print("Episodes: %4d | Wins: %4d | WinRate: %1.3f" % (
        #         episodes, wins, wins / (episodes + 1E-6)))

        self.episodes += 1
        self.episode_steps = 0

        command = []

        if self.first_reset:
            self.init_conn()
            self.first_reset = False

        self.try_killing()

        for unit_pair in self.my_unit_pairs:
            command += self._get_create_units_command(0, unit_pair)

        for unit_pair in self.enemy_unit_pairs:
            command += self._get_create_units_command(1, unit_pair)

        if len(command):
            self.client.send(command)
            self.state = self.client.recv()

        while bool(self.state.waiting_for_restart):
            self.client.send([])
            self.state = self.client.recv()

        # This adds my_units and enemy_units to object.
        self.my_current_units = self._parse_units_to_unit_dict(self.state.units[0])
        self.enemy_current_units = self._parse_units_to_unit_dict(self.state.units[1])

        # This adds my and enemy's units' ids as incrementing list
        self.agent_ids = list(self.my_current_units)
        self.enemy_ids = list(self.enemy_current_units)

        self.obs = self._make_observation()
        self.obs_pre = self.obs
        return self.obs

    def _get_create_units_command(self, player_id, unit_pair):
        defaults = [1, 100, 100, 0, self.state.map_size[0] - 10][len(unit_pair) - 1:]
        unit_type, quantity, x, y, start, end = (list(unit_pair) + defaults)[:6]

        return self.create_units(player_id, quantity, x=x, y=y,
                                 unit_type=unit_type, start=start,
                                 end=end)

    def create_units(self, player_id, quantity, unit_type=0, x=100, y=100, start=0, end=256):
        if x < 0:
            x = (random.randint(0, end - start) + start) * DISTANCE_FACTOR

        if y < 0:
            y = (random.randint(0, end - start) + start) * DISTANCE_FACTOR
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

    def kill_units(self, units, quantity):
        commands = []
        random.shuffle(units)

        for i in range(quantity):
            u = units[i]
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
        return (len(self.state.units[0]) == 0 or \
                len(self.state.units[1]) == 0 or \
                self.episode_steps == self.max_episode_steps)

    def _get_info(self):
        """Returns a dictionary contains debug info"""
        return {}

    def render(self, mode='human', close=False):
        pass
