function [b_delta]=d_bone_delta(s) %计算每个月骨密度变化量 
k=0.000485;
ss=0.1;
sat=7*10^-3;
alph=2;%生成比吸收最大要多
for_rit_th=3*10^-2;
over_sat=-0.05;
over_mid=0.1;
theta1=k*(1-ss);
theta2=k*(1+ss);
C1=sat/theta1;
C2=-4*sat*alph/(theta2-for_rit_th)^2;
C3=abs(over_sat)/(for_rit_th-over_mid)^2;
if s<theta1
    b_delta=C1;
elseif s>theta2&&s<=for_rit_th
    b_delta=2*C2*(s-(theta2+for_rit_th)/2);
elseif s>for_rit_th&&s<=over_mid
    b_delta=2*C3*(s-over_mid);
elseif s>over_mid
    b_delta=0;
else
    b_delta=0;
end
end