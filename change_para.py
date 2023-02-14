import numpy as np

from common.arguments import get_args
from common.utils import make_env
from environment2.Constant import UE_high_probability, UE_low_probability, N_user, N_ETUAV, ETUAV_speed
import os
import shutil

# 改变模型参数，查看reward的变化并保存到csv
from runner import Runner

if __name__ == '__main__':
    user_list = [4, 8, 12, 16, 20, 24, 28, 32]
    reward_list = [0 for _ in range(len(user_list))]
    for i in range(len(user_list)):
        # 改变UE数量
        N_user = user_list[i]
        # 删除model文件夹，清除之前的模型文件
        shutil.rmtree(r'.\model')
        # 强化学习求解

        _args = get_args()
        _env, _args = make_env(_args)
        _runner = Runner(_args, _env)
        _runner.run()  # 训练
        RL_rewards = [_runner.evaluate() for _ in range(10)]  # 评估
        RL_reward = np.mean(RL_rewards)
        reward_list[i] = RL_reward
    save_name = 'change_para' + '_' + str(UE_high_probability) + '_' + str(UE_low_probability) \
                + '_' + str(N_user) + '_' + str(N_ETUAV) + '_' + str(ETUAV_speed) + '.csv '
    np.savetxt(save_name, np.array(reward_list))
