import numpy as np
from environment2.Constant import N_user, N_ETUAV
from generate_location import generate_poistion_txt
from Area import Area
from runner import Runner
from common.arguments import get_args
from common.utils import make_env
from Solution import execute_solution
if __name__ == '__main__':
    compare_loop = 10
    average_ai_energy,average_weight_energy = 0,0
    for _ in range(compare_loop):
        # 生成UE和ETUAV位置
        generate_poistion_txt()
        #  强化学习求解

        # get the params
        args = get_args()
        env, args = make_env(args)
        runner = Runner(args, env)
        ai_return = runner.evaluate()
        average_ai_energy += ai_return
        print('ai returns is', ai_return)

        # 重心方法求解
        weight_return = execute_solution()
        average_weight_energy += weight_return
        print('weight return is', weight_return)

    print('ai',average_ai_energy/compare_loop,'weight',average_weight_energy/compare_loop)