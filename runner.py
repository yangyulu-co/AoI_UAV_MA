from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        returns = []
        actor_loss = dict()
        critic_loss = dict()
        for i in range(self.args.n_agents):
            actor_loss['agent_%d' % i] = np.array([])
            critic_loss['agent_%d' % i] = np.array([])

        s = self.env.reset()
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            u = []
            actions = []

            # 选择动作
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)  # 存在大量重复的信息
                    u.append(action)
                    actions.append(action)
            s_next, r, done, info = self.env.step(actions)

            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent_id, agent in enumerate(self.agents):
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    critic_loss_temp, actor_loss_temp = agent.learn(transitions, other_agents)
                    critic_loss['agent_%d' % agent_id] = np.append(critic_loss['agent_%d' % agent_id], critic_loss_temp)
                    actor_loss['agent_%d' % agent_id] = np.append(actor_loss['agent_%d' % agent_id], actor_loss_temp)

            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                evaluate_reward = self.evaluate()
                print('Returns is', evaluate_reward)
                returns.append(evaluate_reward)

            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
        np.savetxt(self.save_path + '/returns.csv', returns)
        plt.figure()
        plt.plot(range(len(returns)), returns)
        plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
        plt.ylabel('average returns')
        plt.savefig(self.save_path + '/reward.png', format='png')
        plt.close()
        if self.args.save_loss:
            for agent_id, agent in enumerate(self.agents):
                self.save_actor_critic_loss(agent_id, critic_loss['agent_%d' % agent_id], actor_loss['agent_%d' % agent_id])

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):

                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                # print(actions)
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next

            self.env.render()
            # print('rewards=')
            # print(rewards)
            returns.append(rewards)
            # print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes

    def save_actor_critic_loss(self, agent_id, critic_loss_single, actor_loss_single):
        np.savetxt(self.save_path + '/agent_%d/critic_loss.csv' % agent_id,
                   critic_loss_single)
        np.savetxt(self.save_path + '/agent_%d/actor_loss.csv' % agent_id,
                   actor_loss_single)
        plt.figure()
        plt.plot(range(len(critic_loss_single)), critic_loss_single)
        plt.xlabel('episode * ' + str(self.args.batch_size))
        plt.ylabel('agent_%d critic loss' % agent_id)
        plt.savefig(self.save_path + '/agent_%d/critic_loss.png' % agent_id, format='png')
        plt.close()
        plt.figure()
        plt.plot(range(len(actor_loss_single)), actor_loss_single)
        plt.xlabel('episode * ' + str(self.args.batch_size))
        plt.ylabel('agent_%d actor loss' % agent_id)
        plt.savefig(self.save_path + '/agent_%d/actor_loss.png' % agent_id, format='png')
        plt.close()