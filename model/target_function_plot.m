clc;
clear;
%Reward

b=load('AoIs.csv');


reward_size=size(b);%reward的矩阵size
step_count = reward_size(1,1);%step总数

d=zeros(step_count,1);
c=zeros(step_count,1);

evalute_interval = 200;%强化学习训练中评估的间隔
step=1:evalute_interval:step_count*evalute_interval;


plot(step,b(:,1),'Color',[0.69,0.77,0.87],'linewidth',1.5)
hold on;

plot(step,smooth(b(:,1),10,'rlowess'),'Color','#1f77b4','linewidth',2)
set(gca,'linewidth',1);
xlabel('训练步数');%,'FontWeight','bold');
ylabel('目标函数\Psi');
% title('Training reward vs. episodes')
legend('目标函数值','平滑后的目标函数值')
grid on;