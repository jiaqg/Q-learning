clc;
clear;
syms lamda alpha a b c u beta q1nex q2nex r1 r2;
u = 0.1;
beta1 = 95000;%参数设置

round=10;
simu=30;
episode=300;%迭代轮次控制

state_space=10; 
action_space=10;%状态空间和动作空间离散值

R1=zeros(simu,episode);
R2=zeros(simu,episode);%储存不同round的30次simu的300次迭代的reward数据
R1meantotal=zeros(round,episode);
R2meantotal=zeros(round,episode);%储存不同round的300次迭代的平均reward数据
R1t=zeros(1,episode);
R2t=zeros(1,episode);%储存300次迭代的最终平均reward数据

q1re=zeros(simu,episode);
q2re=zeros(simu,episode);%储存不同round的30次simu的300次迭代的实际action数据
q1es=zeros(simu,episode);
q2es=zeros(simu,episode);%储存不同round的30次simu的300次迭代的预测action数据
q1di=zeros(simu,episode);
q2di=zeros(simu,episode);%储存不同round的30次simu的300次迭代的实际-预测action差值数据
q1dimeantotal=zeros(round,episode);
q2dimeantotal=zeros(round,episode);%储存不同round的300次迭代的预测action差值平均数据
q1dit=zeros(1,episode);
q2dit=zeros(1,episode);%储存300次迭代的action差值平均数据

VAR1=zeros(1,episode);
VAR2=zeros(1,episode);%储存300次迭代的action方差数据
ACMRE1=zeros(1,episode);
ACMRE2=zeros(1,episode);%储存300次迭代的累积reward数据

q=zeros(action_space);%储存动作
NE=zeros(round,simu);
NEtotal=zeros(round,1);%储存纳什均衡点

for i=1:state_space
    q(i)=200+(i-1)*200;
end  %离散化动作空间
t1 = xlsread('t1.xlsx');

for r=1:round  
lamda = t1(1,r);
alpha = t1(2,r);
a = t1(3,r);
b= t1(4,r);
c = t1(5,r); %读取参数  
for simulation=1:30
a1 = 1;
a2 = 1;
m = 1;
n = 1;
Q1 = zeros(state_space,action_space);
Q2 = zeros(state_space,action_space);%初始化Q值表
q1es(simulation,1) = q(8);
q2es(simulation,1) = q(9);%首次预测值初始化-随机

for lab=1:episode
beta=beta1/lab;    
box1 = zeros(action_space,1);
box2 = zeros(action_space,1); %存储softpolicy分母   
p1 = zeros(state_space,action_space);
p2 = zeros(state_space,action_space);%储存精确概率

for i=1:state_space
    for j=1:action_space
        box1(i) = box1(i) + exp(Q1(i,j)/beta);
        box2(i) = box2(i) + exp(Q2(i,j)/beta);
    end
end   %分母

for i=1:state_space
    for j = 1:action_space
    p1(i,j) =  exp(Q1(i,j)/beta)/box1(i);
    p2(i,j) =  exp(Q2(i,j)/beta)/box2(i);
    end
end   %计算Q阵不同位置的概率

q1re(simulation,lab) = q(m);
q1es(simulation,lab+1) = q(m);
q2re(simulation,lab) = q(n);
q2es(simulation,lab+1) = q(n);%给出某round的不同simu的不同迭代的预测及实际动作

state21 = q(m);
state12 = q(n);%预测状态，naive agent


next1 = rand();
next2 = rand();%给出random随机数
P1=zeros(action_space);
P2=zeros(action_space);%存储累积概率
for i = 1:state_space
    for j=1:action_space
        for k=1:j
   P1(i,j) = P1(i,j) + p1(i,k);
   P2(i,j) = P2(i,j) + p2(i,k);
    end
end
end%计算累积状态矩阵
 for i=1:action_space
     if (next1<P1(m,i))&&(next1>P1(m,i)-p1(m,i))
        mnew = i;
          end
end%1决定下一步的动作
 for i=1:action_space
     if (next2<P2(n,i))&&(next2>P2(n,i)-p2(n,i))
        nnew = i;
          end
end%2决定下一步的动作
q1nex = q(mnew);
q2nex = q(nnew);%下一步动作值

r1 = (lamda - alpha*(q1nex+q2nex)+rand())*q1nex - a*q1nex*q1nex - b*q1nex - c;
r2 = (lamda - alpha*(q2nex+q1nex)+rand())*q2nex - a*q2nex*q2nex - b*q2nex - c;%计算每一次simu的不同迭代单步reward
Q1(m,mnew) = (1-u)*Q1(m,mnew)+u*r1;
Q2(n,nnew) = (1-u)*Q2(n,nnew)+u*r2;%更新该round的该次simu的Q值表
m=nnew;
n=mnew;%更新current state
R1(simulation,a1)=r1;
R2(simulation,a2)=r2;%存储该round不同simu不同迭代次的reward
a1=a1+1;
a2=a2+1;%脚标+1
end%结束某一simu的300次迭代

for i=1:state_space
    [a1,b1]=max(Q1(i,:));
    [c1,d1]=max(Q2(b1,:));
    if i==d1
        e=q(b1);
        f=q(i);
    end
end
%计算该round的simu的NE，
%disp(e);
%disp(f);
end   %该round结束

for i=1:episode
    for j=1:simu
    R1meantotal(r,i)=R1meantotal(r,i)+R1(j,i);
    R2meantotal(r,i)=R2meantotal(r,i)+R2(j,i);
    end
end  %计算每一round的所有simu在不同迭代次的reward的累计值

for i = 1:simulation
for j = 1:episode
    q1di(i,j) = abs(q1re(i,j)-q1es(i,j));
    q2di(i,j) = abs(q2re(i,j)-q2es(i,j));
end
end  %计算预测与实际的差值 

for i=1:episode
    for j=1:simu
    q1dimeantotal(r,i)=q1dimeantotal(r,i)+q1di(j,i);
    q2dimeantotal(r,i)=q2dimeantotal(r,i)+q2di(j,i);
    end
end  %计算每一round的所有simu在不同迭代次预测与实际差的累计值
end
for i = 1:round
    for j = 1:simu
        NEtotal(i,1) = NEtotal(i,1) + NE(i,j);
    end
end
NEtotal=NEtotal/simu;%计算NE均值

R1meantotal=R1meantotal/simu;
R2meantotal=R2meantotal/simu;
for i = 1:episode
    for j = 1:round
R1t(1,i)=R1t(1,i)+R1meantotal(j,i);
R2t(1,i)=R2t(1,i)+R2meantotal(j,i);
    end
end
R1t=R1t/round;
R2t=R2t/round;
figure(1)
plot(R1t,'r--');%计算reward均值

for i = 1:episode
    for j = 1:round
     VAR1(1,i)=VAR1(1,i)+(R1meantotal(j,i)-R1t(1,i))^2;
     VAR2(1,i)=VAR2(1,i)+(R2meantotal(j,i)-R2t(1,i))^2;
    end
end
VAR1=VAR1/round;
VAR2=VAR2/round;
figure(2)
plot(VAR1,'b--');%计算reward的方差

ACMRE1(1,1) = R1t(1,1);
ACMRE2(1,1) = R2t(1,1);
for i = 2: episode
        ACMRE1(1,i)=ACMRE1(1,i-1)+R1t(1,i);
        ACMRE2(1,i)=ACMRE2(1,i-1)+R2t(1,i);
end
figure(3)
plot(ACMRE1,'b--');%计算accumulate reward

 q1dimeantotal= q1dimeantotal/simu;
 q2dimeantotal= q2dimeantotal/simu;
for i = 1:episode
    for j = 1:round
    q1dit(1,i) = q1dit(1,i)+q1dimeantotal(j,i);
    end
end
q1dit=q1dit/round;
figure(4)
plot(q1dit,'k--');%计算预测-实际的差值

