# gym-starcraft

This repository provides an OpenAI Gym interface to StarCraft: BroodWars online multiplayer game.

## Prerequisites

- First, we need to install TorchCraft with OpenBW support. We have written down a [wiki](https://github.com/apsdehal/gym-starcraft/wiki/Installation) for this.
- Make sure to install python bindings for TorchCraft as well.
- Run `python setup.py develop` for installing this repo.

## Features

- Includes a base environment which can be easily extended to various specific environment and doesn't assume anything about the downstream task.
- Both enemy and our unit's commands can be overriden for scenarios in which either of units need to be controlled for deterministic settings.
- Includes an example derived class for `M vs N` environment which can be further extended for specific cases of general `M vs N` scenarios.
- In `M vs N` environment, any unit type with any quantity can be initialized anywhere on map or within a specific bounding box. This environment can also be used in case of StarCraft community-building tasks as buildings themselves are units.
- Includes an explore mode environments, where we can test exploration mode for agents in big and dynamic map of StarCraft.
- Attack closest and random agent included as an example agent implementation to be used with environment

## Running