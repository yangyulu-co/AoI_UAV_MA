import math
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
        # code = [0 for _ in range(ue_num)]
        # for j in range(ue_num):
        #     code[j] = (i // (3 ** j)) % 3
        code = [(i // (3 ** j)) % 3 for j in range(ue_num)]
        if code.count(1) <= max_compute:  # 如果在DPUAV上计算的没有超出DPUAV的计算上限
            possible_solutions.append(code)

    return possible_solutions


class Area:
    """模型所在的场地范围"""

    def __init__(self, x_range=500.0, y_range=500.0):

        self.agent_num = N_ETUAV
        self.action_dim = 2  # 角度和rate
        self.overall_state_dim = N_user + self.agent_num * 2 * (N_user + self.agent_num - 1)
        self.public_state_dim = N_user  # 用户的电量
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
        # self.UEs = self.load_UEs()
        # """所有ue组成的列表"""
        # self.ETUAVs = self.load_ETUAVs()
        # """所有ETUAV组成的列表"""
        self.aoi = [0.0 for _ in range(N_user)]
        """UE的aoi"""

        state = self.calcul_etuav_state()
        return state

    def render(self,title:str):
        # 画user离散点
        user_x = []
        user_y = []
        for i in range(N_user):
            user_x.append(self.UEs[i].position.data[0, 0])
            user_y.append(self.UEs[i].position.data[0, 1])
        plt.scatter(user_x, user_y, c='#696969',marker='.',label='UE')
        # 画出ETUAV轨迹
        for i in range(N_ETUAV):
            color = '#1f77b4' if i == 0 else '#ff7f0e'
            plt.scatter([self.ETUAVs[i].position.data[0, 0]], [self.ETUAVs[i].position.data[0, 1]],marker='o',c=color,label='UAV'+str(i)+' end')
            plt.plot(self.ETUAVs[i].position.tail[:,0],self.ETUAVs[i].position.tail[:,1],c=color,label='UAV'+str(i))
            plt.scatter([self.ETUAVs[i].position.tail[0,0]],[self.ETUAVs[i].position.tail[0,1]], marker='x',c=color,label='UAV'+str(i)+' start')
        plt.xlim((-250,250))
        plt.ylim((-250,250))
        plt.xlabel("x(m)")
        plt.ylabel("y(m)")
        plt.title(title)
        plt.legend()
        plt.show()


    def step(self, actions):  # action是每个agent动作向量(ndarray[0-2pi, 0-1])的列表，DP在前ET在后

        # 由强化学习控制，ETUAV开始运动
        etuav_move_energy = [etuav.move_by_xy_rate(actions[i][0], actions[i][1]) for i, etuav in
                             enumerate(self.ETUAVs)]
        """ETUAV运动的能耗"""
        # print(etuav_move_energy)
        # ETUAV充电,并记录充入的电量
        etuav_charge_energy = [etuav.charge_all_ues(self.UEs) for etuav in self.ETUAVs]
        """ETUAV给用户冲入的电量"""

        # 计算目标函数
        target = self.calcul_etuav_target()
        reward = [target] * N_ETUAV
        # 加入能量消耗惩罚
        for i in range(N_ETUAV):
            reward[i] -= etuav_move_energy[i] * 0.0001
        # UE产生数据
        for ue in self.UEs:
            ue.generate_task()
        # 计算状态
        state = self.calcul_etuav_state()

        done = False

        return state, reward, done, ''

    def calcul_etuav_target(self) -> float:
        """计算etuav的目标函数值"""
        sum_energy = sum([ue.get_energy_percent() for ue in self.UEs]) / N_user
        """用户平均百分比电量"""
        punish = sum([ue.get_energy_state() - 1 for ue in self.UEs])
        """低电量惩罚（是负数）"""
        weight1 = 1
        weight2 = 0
        """低电量惩罚权重"""
        bias = 0.0
        """为强化学习方便的一个偏置"""
        return sum_energy * weight1 + punish * weight2 - bias

    def calcul_etuav_target_2(self)->float:
        """计算etuav的目标函数值，增加边界外惩罚"""
        """计算etuav的目标函数值"""
        sum_energy = sum([ue.get_energy() for ue in self.UEs]) / N_user
        """用户平均电量"""
        punish = sum([ue.get_energy_state() - 1 for ue in self.UEs])
        """低电量惩罚（是负数）"""
        weight1 = 2 * 10 ** 6
        weight2 = 0
        """低电量惩罚权重"""
        ans = [sum_energy * weight1 + punish * weight2 for _ in range(N_ETUAV)]
        out_punish = 100
        """etuav出界惩罚"""
        out_count = 0
        for et in self.ETUAVs:
            if not self.if_in_area(et.position):
                out_count += 1

        return -(sum_energy * weight1 + punish * weight2 - out_punish*out_count)



    def if_in_area(self, position) -> bool:
        """判断位置是否在场地里"""
        for i in range(2):
            if not self.limit[0, i] <= position.data[0, i] <= self.limit[1, i]:
                return False
        return True

    def calcul_etuav_state(self):
        """计算所有etuav的状态信息，包含百分比电量和百分比相对位置"""
        ue_energy = [ue.get_energy_percent() for ue in self.UEs]
        state = [None for _ in range(N_ETUAV)]
        for i in range(N_ETUAV):
            state[i] = np.array(ue_energy + self.calcul_relative_horizontal_positions('etuav', i))
        return state

    def calcul_relative_horizontal_positions(self, type: str, index: int):
        """计算DPUAV或者ETUAV与除自生外所有的UE,ETUAV,DPUAV的百分比相对水平位置"""
        relative_positions = []
        if type == 'etuav':
            center_position = self.ETUAVs[index].position
            for ue in self.UEs:
                rel_position = center_position.relative_horizontal_position_percent(ue.position,self.limit[1,0],self.limit[1,1])
                relative_positions += rel_position
            for i, etuav in enumerate(self.ETUAVs):
                if i != index:
                    rel_position = center_position.relative_horizontal_position_percent(etuav.position,self.limit[1,0],self.limit[1,1])
                    relative_positions += rel_position
            # for dpuav in self.DPUAVs:
            #     rel_position = center_position.relative_horizontal_position(dpuav.position)
            #     relative_positions += rel_position

        else:
            return False
        return relative_positions


    def calcul_relative_horizontal_positions_radian_length(self, type: str,index:int):
        """计算DPUAV或者ETUAV与除自生外所有的UE,ETUAV,DPUAV的相对水平位置,极坐标系形式"""
        relative_positions = self.calcul_relative_horizontal_positions(type,index)
        ans = [0 for _ in range(len(relative_positions))]
        for i in range(len(relative_positions)//2):

            radian = math.atan2(relative_positions[2*i+1],relative_positions[2*i])
            length = (relative_positions[2 * i + 1]**2+relative_positions[2*i]**2) ** 0.5
            ans[2*i] = radian
            ans[2*i+1] = length
        return ans


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

    def load_UEs(self) -> [UE]:
        """从txt文件加载UE的位置，返回一个list"""
        data = np.loadtxt('environment2\horizontal_ue_loc.txt')
        return [UE(Position(loc[0] * self.limit[1, 0], loc[1] * self.limit[1, 1], 0)) for loc in data]

    def load_ETUAVs(self) -> [ETUAV]:
        """从txt文件加载ETUAV的位置，返回一个list"""
        data = np.loadtxt('environment2\horizontal_et_loc.txt')
        return [ETUAV(Position(loc[0] * self.limit[1, 0], loc[1] * self.limit[1, 1], ETUAV_height)) for loc in data]




if __name__ == "__main__":
    area = Area()
    print(area.generate_UEs())
    # area.step([np.array([0, 0.1]), np.array([0.2, 0.3]), np.array([0.4, 0.5]), np.array([0.6, 0.7])])
    # print(area.step([np.array([0, 0.1]), np.array([0.2, 0.3]), np.array([0.4, 0.5]), np.array([0.6, 0.7])]))
