import matplotlib.pyplot as plt
import numpy as np

from common.arguments import get_args
from common.utils import make_env
from environment2.Constant import UE_high_probability, UE_low_probability, N_user, N_ETUAV, ETUAV_speed

if __name__ == '__main__':
    user_list = [4, 8, 12, 16, 20, 24, 28, 32]
    plt.figure()

    save_name = 'change_para' + '_' + str(UE_high_probability) + '_' + str(UE_low_probability) \
                + '_' + str(N_user) + '_' + str(N_ETUAV) + '_' + str(ETUAV_speed) + '.csv '
    data = np.loadtxt(save_name)
    plt.plot(user_list,data,label='test',c='#87CEEB')

    plt.xlabel("Number of UEs")
    plt.ylabel("Accumulated reward")
    plt.title("Accumulated reward vs number of UEs")
    plt.legend()
    plt.show()