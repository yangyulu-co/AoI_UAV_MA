clc;
clear;
%Reward

b=load('returns.csv');


reward_size=size(b);%reward的矩阵size
step_count = reward_size(1,1);%step总数

d=zeros(step_count,1);
c=zeros(step_count,1);

evalute_interval = 200;%强化学习训练中评估的间隔
step=1:evalute_interval:step_count*evalute_interval;



plot(step,b(:,1),'Color','#C6D9F1','linewidth',1.5);
hold on;
plot(step,smooth(b(:,1),10,'rlowess'),'Color','#1F497D','linewidth',2);
hold on;
plot(step,b(:,2),'Color','#F2DCDB','linewidth',1.5)
hold on;
plot(step,smooth(b(:,2),10,'rlowess'),'Color','#A61C00','linewidth',2)
hold on;
plot(step,b(:,3),'Color','#D9EAD3','linewidth',1.5)
hold on;
plot(step,smooth(b(:,3),10,'rlowess'),'Color','#006100','linewidth',2)
hold on;
plot(step,b(:,4),'Color','#FDE9D9','linewidth',1.5)
hold on;
plot(step,smooth(b(:,4),10,'rlowess'),'Color','#F79646','linewidth',2)
hold on;


% set(gca,'linewidth',1);


legend('DO-UAV 1 reward','DO-UAV 1 smoothed reward','DO-UAV 2 reward','DO-UAV 2 smoothed reward')

xlabel('训练步数');
ylabel('无人机的奖励');
grid on;


