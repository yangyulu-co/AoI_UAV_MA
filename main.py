from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch
def execute_ai_solution():
    """加载强化学习模型，执行强化学习方法求解充电问题"""
    # get the params
    _args = get_args()
    _env, _args = make_env(_args)
    _runner = Runner(_args, _env)
    return _runner.evaluate()


if __name__ == '__main__':
    # get the params
    args = get_args()
    env, args = make_env(args)
    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()
