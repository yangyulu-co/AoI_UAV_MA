import numpy as np
from environment2.Constant import N_user, N_ETUAV
from environment2.generate_location import generate_poistion_txt
from main import execute_ai_solution
from environment2.Solution import execute_weight_solution
if __name__ == '__main__':
    compare_loop = 10
    average_ai_energy = [0 for _ in range(compare_loop)]
    average_weight_energy = [0 for _ in range(compare_loop)]
    for i in range(compare_loop):
        # 生成UE和ETUAV位置
        generate_poistion_txt()
        #  强化学习求解
        ai_return = execute_ai_solution()
        average_ai_energy[i] = ai_return
        print('ai return is', ai_return)

        # 重心方法求解
        weight_return = execute_weight_solution()
        average_weight_energy[i] = weight_return
        print('weight return is', weight_return)

    print('ai',average_ai_energy,'weight',average_weight_energy)