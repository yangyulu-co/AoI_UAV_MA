import random
from collections import defaultdict
from copy import copy

import numpy as np
import matplotlib.pyplot as plt

from environment2.Constant import N_user, N_ETUAV, N_DPUAV, eta_1, eta_2, eta_3, DPUAV_height, ETUAV_height, time_slice
from environment2.DPUAV import DPUAV, max_compute
from environment2.ETUAV import ETUAV
from environment2.Position import Position
from environment2.UE import UE


def get_link_dict(ues: [UE], dpuavs: [DPUAV]):
    """返回UEs和DAPUAVs之间的连接情况,返回一个dict,key为dpuav编号，value为此dpuav能够连接的ue组成的list"""

    link_dict = defaultdict(list)
    for i, ue in enumerate(ues):
        near_dpuav = None
        near_distance = None
        for j, dpuav in enumerate(dpuavs):
            if ue.if_link_DPUAV(dpuav) and ue.task is not None:  # 如果在连接范围内且存在task需要卸载
                distance = ue.distance_DPUAV(dpuav)
                if near_dpuav is None or near_distance > distance:
                    near_dpuav = j
                    near_distance = distance
        if near_distance is not None:
            link_dict[near_dpuav].append(i)

    return link_dict


def calcul_target_function(aois: [float], energy_dpuavs: [float], energy_etuavs: [float]) -> float:
    """计算目标函数的值"""
    return eta_1 * sum(aois) + eta_2 * sum(energy_dpuavs) + eta_3 * sum(energy_etuavs)


def generate_solution(ue_num: int) -> list:
    """根据输入的UE数量，返回所有的可行的卸载决策"""
    max_count = 3 ** ue_num
    possible_solutions = []
    for i in range(max_count):
        code = [0 for _ in range(ue_num)]
        for j in range(ue_num):
            code[j] = (i // (3 ** j)) % 3
        if code.count(1) <= max_compute:
            possible_solutions.append(code)

    return possible_solutions


class Area:
    """模型所在的场地范围"""

    def __init__(self, x_range=500.0, y_range=500.0):

        self.agent_num = N_ETUAV
        self.action_dim = 2  # 角度和rate
        self.overall_state_dim = N_user + self.agent_num * 2 * (N_user + self.agent_num - 1)
        self.public_state_dim = N_user  # 用户的AoI、lambda、队列状况是公有部分
        self.private_state_dim = 2 * (N_user + self.agent_num - 1)  # 与其他的位置关系是私有部分

        self.limit = np.empty((2, 2), np.float32)
        """场地限制"""
        self.limit[0, 0] = -x_range / 2
        self.limit[1, 0] = x_range / 2
        self.limit[0, 1] = -y_range / 2
        self.limit[1, 1] = y_range / 2

        # 生成ue,etuav,dpuav
        self.UEs = self.generate_UEs(N_user)
        """所有ue组成的列表"""
        self.ETUAVs = self.generate_ETUAVs(N_ETUAV)
        """所有ETUAV组成的列表"""

        self.aoi = [0.0 for _ in range(N_user)]
        """UE的aoi"""

    def reset(self):
        # 生成ue,etuav,dpuav
        self.UEs = self.generate_UEs(N_user)
        """所有ue组成的列表"""
        self.ETUAVs = self.generate_ETUAVs(N_ETUAV)
        """所有ETUAV组成的列表"""

        self.aoi = [0.0 for _ in range(N_user)]
        """UE的aoi"""

        state = self.calcul_etuav_state()
        return state

    def step(self, actions):  # action是每个agent动作向量(ndarray[0-2pi, 0-1])的列表，DP在前ET在后
        # UE产生数据
        for ue in self.UEs:
            ue.generate_task()

        # ETUAV充电
        for etuav in self.ETUAVs:
            etuav.charge_all_ues(self.UEs)

        # 由强化学习控制，ETUAV开始运动
        etuav_move_energy = [0.0 for _ in range(N_ETUAV)]
        """ETUAV运动的能耗"""
        for i, etuav in enumerate(self.ETUAVs):
            etuav_move_energy[i] = etuav.move_by_radian_rate_2(actions[i][0], actions[i][1])
        # 计算状态
        state = self.calcul_etuav_state()

        # 计算目标函数
        target = self.calcul_etuav_target()
        reward = [-target] * N_ETUAV
        done = False

        return state, reward, done, ''

    # def calcul_state(self):
    #     """计算所有UAV的状态信息，以[narray]格式返回"""
    #     # 公共的环境信息
    #     # 所有用户的AOI
    #     dpuav_aoi = copy(self.aoi)
    #     # 得到UE生成数据的速率
    #     ue_probability = [ue.get_lambda() for ue in self.UEs]
    #     # 得到UE是否有数据
    #     ue_if_task = [0 if ue.task is None else 1 for ue in self.UEs]
    #     public_state = np.array(dpuav_aoi + ue_probability + ue_if_task)
    #
    #     state = [None for _ in range(N_DPUAV + N_ETUAV)]
    #     for i in range(N_DPUAV):
    #         state[i] = np.append(public_state, self.calcul_relative_horizontal_positions("dpuav", i))
    #     for i in range(N_ETUAV):
    #         state[i + N_DPUAV] = np.append(public_state, self.calcul_relative_horizontal_positions("etuav", i))
    #     return state

    def calcul_etuav_target(self)->float:
        """计算etuav的目标函数值"""
        sum_energy = sum([ue.get_energy() for ue in self.UEs]) / N_user
        """用户平均电量"""
        punish = sum([ue.get_energy_state() for ue in self.UEs])
        """低电量惩罚"""
        weight = 2 * (10 ** (-6))
        """低电量惩罚权重"""
        return sum_energy - punish * weight


    def calcul_etuav_state(self):
        """计算所有etuav的状态信息，包含电量和相对位置"""
        ue_energy = [ue.get_energy() for ue in self.UEs]
        state = [None for _ in range(N_ETUAV)]
        for i in range(N_ETUAV):
            state[i] = np.array(ue_energy + self.calcul_relative_horizontal_positions('etuav', i))
        return state

    def calcul_relative_horizontal_positions(self, type: str, index: int):
        """计算DPUAV或者ETUAV与除自生外所有的UE,ETUAV,DPUAV的相对水平位置"""
        relative_positions = []
        if type == 'etuav':
            center_position = self.ETUAVs[index].position
            for ue in self.UEs:
                rel_position = center_position.relative_horizontal_position(ue.position)
                relative_positions += rel_position
            for i, etuav in enumerate(self.ETUAVs):
                if i != index:
                    rel_position = center_position.relative_horizontal_position(etuav.position)
                    relative_positions += rel_position
            # for dpuav in self.DPUAVs:
            #     rel_position = center_position.relative_horizontal_position(dpuav.position)
            #     relative_positions += rel_position

        else:
            return False
        return relative_positions

    # def calcul_single_dpuav_single_ue_energy_aoi(self, dpuav_index: int, ue_index: int, offload_choice):
    #     """计算单个dpuav单个ue的卸载决策下的能量消耗和aoi"""
    #     energy = self.DPUAVs[dpuav_index].calcul_single_compute_and_offloading_energy(self.UEs[ue_index],
    #                                                                                   offload_choice)
    #     aoi = self.DPUAVs[dpuav_index].calcul_single_compute_and_offloading_aoi(self.UEs[ue_index], offload_choice)
    #     if aoi is None:
    #         aoi = self.aoi[ue_index] + 1
    #     return energy, aoi

    def find_single_dpuav_best_offload(self, dpuav_index: int, ue_index_list: list):
        """穷举查找单个DPUAV下多个用户的最优卸载决策,返回数据格式为[dpuav_index,ue_index,{0,1,2}]组成的list"""
        solutions = generate_solution(len(ue_index_list))
        best_target = float('inf')
        best_solution = None

        for solution in solutions:
            solution_energy = 0.0
            solution_aoi = 0.0

            for i in range(len(ue_index_list)):
                energy, aoi = self.calcul_single_dpuav_single_ue_energy_aoi(dpuav_index, ue_index_list[i], solution[i])
                solution_energy += energy
                solution_aoi += aoi
            target = solution_energy * eta_2 + solution_aoi * eta_1

            if target < best_target:
                best_solution = copy(solution)
                best_target = target

        ans = []
        for i in range(len(ue_index_list)):
            ans.append([dpuav_index, ue_index_list[i], best_solution[i]])
        return ans

    def find_best_offload(self, link: dict):
        """穷举查找多个DPUAV下多个用户的最优卸载决策,返回数据格式为[dpuav_index,ue_index,{0,1,2}]组成的list"""
        ans = []
        for dpuav in link.keys():
            single_ans = self.find_single_dpuav_best_offload(dpuav, link[dpuav])
            ans += single_ans
        return ans

    def if_in_area(self, position) -> bool:
        """判断位置是否在场地里"""
        for i in range(2):
            if not self.limit[0, i] <= position.data[0, i] <= self.limit[1, i]:
                return False
        return True

    def generate_single_UE_position(self) -> Position:
        """随机生成一个UE在区域里的点"""

        x = random.uniform(self.limit[0, 0], self.limit[1, 0])
        y = random.uniform(self.limit[0, 1], self.limit[1, 1])
        return Position(x, y, 0)

    def generate_single_ETUAV_position(self) -> Position:
        """随机生成一个ETUAV在区域里的点"""

        x = random.uniform(self.limit[0, 0], self.limit[1, 0])
        y = random.uniform(self.limit[0, 1], self.limit[1, 1])
        return Position(x, y, ETUAV_height)

    def generate_single_DPUAV_position(self) -> Position:
        """随机生成一个DPUAV在区域里的点"""

        x = random.uniform(self.limit[0, 0], self.limit[1, 0])
        y = random.uniform(self.limit[0, 1], self.limit[1, 1])
        return Position(x, y, DPUAV_height)

    def generate_UEs(self, num: int) -> [UE]:
        """生成指定数量的UE，返回一个list"""
        return [UE(self.generate_single_UE_position()) for _ in range(num)]

    def generate_ETUAVs(self, num: int) -> [ETUAV]:
        """生成指定数量ETUAV，返回一个list"""
        return [ETUAV(self.generate_single_ETUAV_position()) for _ in range(num)]

    def generate_DPUAVs(self, num: int) -> [DPUAV]:
        """生成指定数量DPUAV，返回一个list"""
        return [DPUAV(self.generate_single_DPUAV_position()) for _ in range(num)]


if __name__ == "__main__":
    area = Area()
    # area.step([np.array([0, 0.1]), np.array([0.2, 0.3]), np.array([0.4, 0.5]), np.array([0.6, 0.7])])
    print(area.step([np.array([0, 0.1]), np.array([0.2, 0.3]), np.array([0.4, 0.5]), np.array([0.6, 0.7])]))
