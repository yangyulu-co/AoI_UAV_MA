import numpy as np
from environment2.Constant import N_user, N_ETUAV

def generate_poistion_txt():
    """生成UE和ETUAV的位置，并保存到txt中"""
    # 生成UE和ETUAV位置
    horizontal_ue_loc = (np.random.rand(N_user, 2) - 0.5) * 2
    horizontal_et_loc = (np.random.rand(N_ETUAV, 2) - 0.5) * 2
    # 保存到TXT中
    np.savetxt('horizontal_ue_loc.txt', horizontal_ue_loc)
    np.savetxt('horizontal_et_loc.txt', horizontal_et_loc)
