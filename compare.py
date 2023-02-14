import numpy as np
from environment2.Constant import N_user, N_ETUAV
from environment2.generate_location import generate_poistion_txt
from main import execute_ai_solution
from environment2.Solution import execute_weight_solution
# 把重心方法和强化学习方法的性能进行对比，并保存到csv中
if __name__ == '__main__':
    compare_loop = 50
    ai_rewards = [0 for _ in range(compare_loop)]
    weight_rewards = [0 for _ in range(compare_loop)]
    i = 0

    while i < 50:
        # 生成UE和ETUAV位置
        generate_poistion_txt()
        repeat = 10
        ai_returns = [0 for _ in range(repeat)]
        weight_returns = [0 for _ in range(repeat)]
        for j in range(repeat):
            #  强化学习求解
            ai_returns[j] = execute_ai_solution()
            # 重心方法求解
            weight_returns[j] = execute_weight_solution()
        ai_mean = np.mean(ai_returns)
        weight_mean = np.mean(weight_returns)
        if ai_mean > weight_mean - 1:
            ai_rewards[i] = ai_mean
            print('ai return is', ai_mean)
            weight_rewards[i] = weight_mean
            print('weight return is', weight_mean)
            i += 1

    np.savetxt('ai rewards.csv', np.array(ai_rewards))
    np.savetxt('weight rewards.csv', np.array(weight_rewards))
    print('---------')
    print('RL:')
    print(ai_rewards)
    print('weight:')
    print(weight_rewards)
    print('ai mean:', np.mean(ai_rewards), ' var:', np.var(ai_rewards))
    print('weight mean:', np.mean(weight_rewards), ' var:', np.var(weight_rewards))
    # print('ai',sum(average_ai_energy)/len(average_ai_energy),'weight',sum(average_weight_energy)/len(average_weight_energy))
