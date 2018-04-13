from gym.envs.registration import register

register(
    id='Starcraft-M1v1-v0',
    entry_point='gym_starcraft.envs:StarCraftM1v1',
)
