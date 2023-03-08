import numpy as np
import matplotlib.pyplot as plt
#  画图相关代码
if __name__ == '__main__':
    # 画训练中reward和
    reward = np.loadtxt(r"model\returns.csv")

    episodes = [_*500 for _ in range(np.size(reward, 0))]
    # 计算平滑曲线
    smoothed_reward = np.convolve(reward,[1,1,1],mode='same')
    plt.figure()
    plt.plot(episodes,reward,c='#a8d3f0',label="Accumulated reward",linewidth=1)
    plt.plot(episodes, reward, c='#98F5FF', label="Accumulated reward", linewidth=1)
    # plt.plot(episodes, smoothed_reward, c='#1f77b4', linewidth=2, label="Smoothed reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Accumulated reward versus training episodes")
    plt.legend()
    plt.show()

    # 画比较的图
    ai_rewards = np.loadtxt(r'ai rewards.csv')
    weight_rewards = np.loadtxt(r'weight rewards.csv')
    print('ai mean:', np.mean(ai_rewards), ' var:', np.var(ai_rewards))
    print('weight mean:', np.mean(weight_rewards), ' var:', np.var(weight_rewards))
    test_count = [_ for _ in range(np.size(ai_rewards,0))]
    plt.figure()
    plt.plot(test_count,ai_rewards,label='M2DDPG rewards',c='#87CEEB')
    plt.plot(test_count,weight_rewards,label='Weight strategy rewards',c='#9AFF9A')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    # plt.title("Accumulated rewards compare")
    plt.legend()
    plt.show()
