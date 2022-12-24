import numpy as np
import inspect
import functools


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def make_env(args):
    from environment2.Area import Area
    from environment2.Constant import N_user, N_ETUAV, N_DPUAV

    env = Area()

    args.n_agents = N_ETUAV + N_DPUAV
    args.public_obs_shape = env.public_state_dim
    args.private_obs_shape = env.private_state_dim
    args.overall_obs_shape = env.overall_state_dim
    # action维数
    args.action_shape = [env.action_dim] * args.n_agents  # 每一维代表该agent的act维度
    # 输出上下限
    args.high_action = 1
    args.low_action = -1

    return env, args
