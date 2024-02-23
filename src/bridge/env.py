from jaxrl_m.envs.bridge.env_wrappers import *

class DummyEnv():
    def __init__(self):
        super().__init__()
        obs_dict = dict()
        obs_dict['pixels'] = Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        if variant.add_states:
            obs_dict['state'] = Box(low=-100000, high=100000, shape=(7,), dtype=np.float32)
        if num_tasks > 1:
            obs_dict['task_id'] = Box(low=0, high=1, shape=(num_tasks,), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.spec = None
        self.action_space = Box(
            np.asarray([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.]),
            np.asarray([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0]),
            dtype=np.float32)

    def seed(self, seed):
        pass


def get_dummy_env():
    def wrap(env):
        ic('WRAP env')
        # assert not (variant.normalize_actions and variant.rescale_actions)
        # if variant.add_prev_actions:
        #     print('Adding prev actions')
        #     # Only do this to the extent that there is one less frame to be
        #     # stacked, since this only concats the previous actions, not the
        #     # current action....
        #     if variant.frame_stack == 1:
        #         num_action_stack = 1
        #     else:
        #         num_action_stack = variant.frame_stack - 1
        #     env = PrevActionStack(env, num_action_stack)
        env = RescaleActions(env)
        if variant.add_states:
            env = StateStack(env, variant.frame_stack)
        if not variant.from_states:
            env = FrameStack(env, variant.frame_stack)
        env = gym.wrappers.TimeLimit(env, variant.episode_timelimit)
        return env
