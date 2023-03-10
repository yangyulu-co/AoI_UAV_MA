import torch
import os
from maddpg.actor_critic import Actor, Critic
import numpy as np
from common.utils import to_tensor_var


class MADDPG:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.use_cuda = args.use_cuda and torch.cuda.is_available()

        # create the network
        self.actor_network = Actor(args, agent_id)
        self.critic_network = Critic(args)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id)
        self.critic_target_network = Critic(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # use cuda
        if self.use_cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))
        # else:
        #     print('not found')

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            if self.use_cuda:
                transitions[key] = torch.tensor(transitions[key], dtype=torch.float32, device='cuda')
                # transitions[key] = to_tensor_var(transitions[key], self.use_cuda, 'float')
            else:
                transitions[key] = torch.tensor(transitions[key], dtype=torch.float32, device='cpu')
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])  # 有agent_num个索引，每个索引中是batch_size条记录，每条是该agent的观测值

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.n_agents):
                # 计算所有agent下一步的动作
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            # 得到去重后的环境
            overall_state_next = self.generate_overall_state(o_next, self.use_cuda)

            # 估计next_q
            q_next = self.critic_target_network(overall_state_next, u_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        overall_state = self.generate_overall_state(o, self.use_cuda)
        q_value = self.critic_network(overall_state, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = - self.critic_network(overall_state, u).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = self.args.save_dir
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + 'actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + 'critic_params.pkl')

    def generate_overall_state(self, state, use_cuda=True):
        # 这里state有agent_num个索引，每个索引中是batch_size条记录，每条是该agent的观测值
        # 需要处理成size条，每条由overall_state组成
        # 输入每个智能体的观测值组成的列表，生成一个总的观测值（实现去重的工作）
        if use_cuda:
            overall_state = torch.empty([self.args.batch_size, self.args.overall_obs_shape], dtype=torch.float32, device='cuda')
            if state:
                temp = torch.empty([self.args.batch_size, self.args.overall_obs_shape - self.args.public_obs_shape], dtype=torch.float32, device='cuda')
                for i in range(self.args.batch_size):
                    overall_state[i][:self.args.public_obs_shape] = state[0][i][:self.args.public_obs_shape]
                    for j in range(self.args.n_agents):
                        temp[i][j * self.args.private_obs_shape:(j + 1) * self.args.private_obs_shape] = \
                            state[j][i][self.args.public_obs_shape:]
                    overall_state[i][self.args.public_obs_shape:] = temp[i][:]

            return overall_state

        else:
            overall_state = np.empty([self.args.batch_size, self.args.overall_obs_shape])
            if state:
                temp = np.empty([self.args.batch_size, self.args.overall_obs_shape - self.args.public_obs_shape])
                for i in range(self.args.batch_size):
                    overall_state[i][:self.args.public_obs_shape] = state[0][i][:self.args.public_obs_shape]
                    for j in range(self.args.n_agents):
                        temp[i][j*self.args.private_obs_shape:(j+1)*self.args.private_obs_shape] = \
                            state[j][i][self.args.public_obs_shape:]
                    overall_state[i][self.args.public_obs_shape:] = temp[i][:]

            return torch.tensor(overall_state, dtype=torch.float32)

