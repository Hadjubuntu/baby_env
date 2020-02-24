from gym.envs.registration import register


register(
    id='baby-v0',
    entry_point='baby.envs.baby_env:BabyEnv',
)


register(
    id='topk-baby-v0',
    entry_point='baby.envs.topk_baby_env:TopkBabyEnv',
)
