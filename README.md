# gym-starcraft

This repository provides an OpenAI Gym interface to StarCraft: BroodWars online multiplayer game.

## Features

- Includes a base environment which can be easily extended to various specific environment and doesn't assume anything about the downstream task.
- Both enemy and our unit's commands can be overriden for scenarios in which either of units need to be controlled for deterministic settings.
- Includes an example derived class for `M vs N` environment which can be further extended for specific cases of general `M vs N` scenarios.
- In `M vs N` environment, any unit type with any quantity can be initialized anywhere on map or within a specific bounding box. This environment can also be used in case of StarCraft community-building tasks as buildings themselves are units.
- Includes an explore mode environment, where one can test exploration mode for agents in big and dynamic map of StarCraft.
- Attack closest and random agent included as an example agent implementation to be used with environment.
- MvN example supports partial observable setting in which vision is limited as in fog of war.
- Supports built-in, attack-closest and attack-weakest AI strategies.

### Combat Mode
![Combat Mode](https://media.giphy.com/media/2fUIxtNqWQMEdhmfTm/giphy.gif)

### Explore Mode
![Explore Mode](https://media.giphy.com/media/nEujB3aD9HWAWaEnSO/giphy.gif)

## Prerequisites and Installation

- First, we need to install TorchCraft with OpenBW support. We have written down a [wiki](https://github.com/apsdehal/gym-starcraft/wiki/Installation) for this.
- Make sure to install python bindings for TorchCraft as well.
- Run `python setup.py develop` for installing this repo.

## Running

- First, update the [config](https://github.com/apsdehal/gym-starcraft/blob/paper/gym_starcraft/envs/config.yml) present in `gym-starcraft/envs/config.yml` as per your requirements. Explanation is present in comments.
- Make sure all prerequisites are completed.
- To run a sample `attack_closest` match between 10 marines and 3 zealots in bounding box of (100, 100) to (150, 150) with GUI, run the following command:

```
 python examples/attack_closest.py --server_ip 127.0.0.1 --torchcraft_dir=TORCHCRAFT_DIR --set_gui --nagents 10 --max_steps 200 --frame_skip 8 --nenemies 3 --our_unit_type 0 --enemy_unit_type 65 --init_range_end 150 --full_vision --unlimited_attack_range --initialize_enemy_together --step_size 16
```

Most of the other flags are self explanatory.

Use `python examples/attack_closest.py -h` for other options that are available.

## Custom Environment Development

## Credits

Initial implementation of this package was based out of Alibaba's [gym-starcraft](https://github.com/alibaba/gym-starcraft) which didn't work properly with latest TorchCraft version. To handle customized needs for our NIPS 2018 submitted paper (not public yet), we developed this version. Without contributions of [@ebetica](https://github.com/ebetica) (Zeming Lin), [@tesatory](https://github.com/tesatory) (Sainbayar Sukhbaatar) and [@tshrjn](https://github.com/tshrjn) (Tushar Jain) and others at Facebook AI Research (FAIR) this package won't be possible.

## License

Code for this project is available under MIT license.

## TODO

- Support for heterogenuous agents requires some changes in vision range calculation (Vision range is calculated only once)