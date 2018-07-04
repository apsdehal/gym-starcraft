from gym.envs.registration import register

register(
    id='Starcraft-MvN-v0',
    entry_point='gym_starcraft.envs:StarCraftMvN',
)
