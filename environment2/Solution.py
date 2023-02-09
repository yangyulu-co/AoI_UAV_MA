import numpy as np

from environment2.Area import Area
from environment2.Constant import ETUAV_speed, time_slice, N_ETUAV, N_user, time_length
import math

debug_print = False


def get_gravity_center(position: [float], weight: [float]) -> [float]:
    """计算出重心"""
    # if len(weight) * 2 != len(position):
    #     print('位置权重尺寸不匹配')
    center_x, center_y = 0, 0
    sum_weight = sum(weight)
    for i in range(len(weight)):
        center_x += position[2 * i] * weight[i]
        center_y += position[2 * i + 1] * weight[i]
    center_x /= sum_weight
    center_y /= sum_weight
    return [center_x, center_y]


def get_action(start_position: [float], target_position: [float]) -> [float]:
    """输入出发位置和希望去的位置，然后返回角度和速度率的action"""
    dx = (target_position[0] - start_position[0]) * 250
    dy = (target_position[1] - start_position[1]) * 250
    distance = (dx ** 2 + dy ** 2) ** 0.5  # 直线距离
    radian = math.atan2(dy, dx)
    speed_rate = min(distance / (ETUAV_speed * time_slice), 1)
    return [radian, speed_rate]


def get_action_2(start_position: [float], target_position: [float]) -> [float]:
    """输入出发位置和希望去的位置，然后返回dxdy的action"""
    dx = (target_position[0] - start_position[0]) * 250
    dy = (target_position[1] - start_position[1]) * 250
    dx_rate = max(min(dx / (ETUAV_speed * time_slice), 1), -1)
    dy_rate = max(min(dy / (ETUAV_speed * time_slice), 1), -1)
    return [dx_rate, dy_rate]


class Solution:
    """加权重心解法"""

    def __init__(self, problem: Area):
        self.problem = problem

        self.state = problem.reset()

        self.count = 0
        """用户做出的action计数"""
        self.sum_reward = 0.0
        """累计的reward"""

    def step(self):
        # 计数加1
        self.count += 1
        # 切分state为位置和权重
        position_state = [self.state[_][N_user:] for _ in range(N_ETUAV)]
        weight_state = [self.state[_][0:N_user] for _ in range(N_ETUAV)]

        if debug_print:
            print('1', weight_state)
        for etuav in range(N_ETUAV):
            weight_state[etuav] = [1 if _ < 0.5 else 0 for _ in weight_state[etuav]]
        if debug_print:
            print('2', weight_state)
        # 计算action

        action = [get_action_2([0, 0], get_gravity_center(position_state[_], weight_state[_])) for _ in range(N_ETUAV)]

        # print('position:',position_state)
        # print('action:', action)
        # 执行action,得到新的state
        self.state, reward, done, t = self.problem.step(action)
        self.sum_reward += reward[0]

    def render(self):
        print('reward', self.sum_reward)
        self.problem.render()

    def get_sum_reward(self) -> float:
        """返回累计的reward值"""
        return self.sum_reward

    def get_step_count(self) -> int:
        """返回进行的step数"""
        return self.count


def execute_solution():
    """执行重心方法求解充电问题"""
    area = Area()
    solution = Solution(area)
    for s in range(time_length):
        solution.step()
    return solution.get_sum_reward()


if __name__ == "__main__":

    average_reward = 0
    test_count = 20
    for test in range(test_count):
        average_reward += execute_solution()
    print(average_reward / test_count)
