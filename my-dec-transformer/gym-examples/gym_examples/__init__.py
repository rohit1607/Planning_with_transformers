from gym.envs.registration import register

register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
)

register(
    id="gym_examples/GridWorld-v1",
    entry_point="gym_examples.envs:GridWorldEnvCopy",
)

register(id='gym_examples/contGrid-v5',entry_point='gym_examples.envs:ContGridWorld_v5',)

register(id='gym_examples/contGrid-v6',entry_point='gym_examples.envs:ContGridWorld_v6',)


register(id='gym_examples/contGrid-v5_1',entry_point='gym_examples.envs:ContGridWorld_v5_1',)
register(id='gym_examples/contGrid-v6_1',entry_point='gym_examples.envs:ContGridWorld_v6_1',)

register(id='gym_examples/contGrid-v7_1',entry_point='gym_examples.envs:ContGridWorld_v7_1',)

