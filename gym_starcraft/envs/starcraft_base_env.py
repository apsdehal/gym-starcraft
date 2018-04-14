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


class StarCraftBaseEnv(gym.Env):
    def __init__(self, torchcraft_dir='~/TorchCraft',
                 bwapi_launcher_path='../bwapi/bin/BWAPILauncher',
                 config_path='~/gym-starcraft/gym_starcraft/envs/config.yml',
                 server_ip='127.0.0.1',
                 server_port=11111,
                 speed=0, frame_skip=10000, set_gui=0, self_play=0,
                 max_episode_steps=1000, final_init=True):

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
        options['BWAPI_CONFIG_AUTO_MENU__AUTO_RESTART'] = "OFF"
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


        # NOTE: These should be overrided in derived class
        # Should be a list of pairs where each pair is (number, unit_type)
        self.my_unit_pairs = []
        self.enemy_unit_pairs = []

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
                 [tcc.set_frameskip, self.frame_skip],
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

        self.client.send(self._make_commands(action))
        self.state = self.client.recv()

        while not self._has_step_completed():
            self.client.send([])
            self.state = self.client.recv()

        self.obs = self._make_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = self._get_info()
        self.obs_pre = self.obs
        return self.obs, reward, done, info

    def try_killing(self):
        if not self.state:
            return


        while len(self.state.units[0]) + len(self.state.units[1]) != 0:
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

        self.try_killing()

        # print("Episodes: %4d | Wins: %4d | WinRate: %1.3f" % (
        #         episodes, wins, wins / (episodes + 1E-6)))

        self.episodes += 1
        self.episode_steps = 0

        command = []

        if self.first_reset:
            self.init_conn()
            self.first_reset = False

        for unit_pair in self.my_unit_pairs:
            command += self.create_units(0, unit_pair[1], x=unit_pair[2],
                                         y=unit_pair[3], unit_type=unit_pair[0])

        for unit_pair in self.enemy_unit_pairs:
            command += self.create_units(1, unit_pair[1], x=unit_pair[2],
                                         y=unit_pair[3], unit_type=unit_pair[0])

        if len(command):
            self.client.send(command)
            self.state = self.client.recv()

        while bool(self.state.waiting_for_restart):
            self.client.send([])
            self.state = self.client.recv()


        self.obs = self._make_observation()
        self.obs_pre = self.obs
        return self.obs


    def create_units(self, player_id, quantity, x=100, y=100, unit_type=0):
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
        return bool(self.state.game_ended) or self.state.battle_just_ended

    def _get_info(self):
        """Returns a dictionary contains debug info"""
        return {}

    def render(self, mode='human', close=False):
        pass
