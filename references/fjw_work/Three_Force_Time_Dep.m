clear
close all
%% 导入.inp文件
load nod_coo
load ele_nod      
load D_3d               % 弹性张量
load B_3d               % 应变插值矩阵


% load truss_ele.mat      % 椎间融合器内层的的桁架结构(非设计域)
load cor_ele            % 皮质骨(非设计域)
load tra_ele            % 骨翼(非设计域)         

load cage_ele           % 全部椎间融合器单元
load desi_ele           % 外层椎间融合器单元 设计域单元(寻找这部分单元的x)
load obj_ele            % 内层椎间融合器单元(这部分单元的骨量最大为目标)
% load wing_ele           % 骨翼(非设计域)

%% 初始化空间质量分数
% N1=197;N2=176;N3=101;
N1=152;N2=131;N3=134;
ini_str = 0.36 * ones(N1, N2, N3);  %初始腰椎结构

 
% ini_cage = 0.2 * ones(N1, N2, N3);  %初始椎间融合器分布 
ini_cage = 0.3 * ones(N1, N2, N3);  %初始椎间融合器分布 

v = 0.6 ^ 3;
%% 非设计域inp文件
edit_nodesi_inp  

%% 骨演变参数
dt=1;                               %迭代时间,1个月
P=3;                                %骨生长月份，假定1年                                                                                                 
b_max=1.86;                         %骨量真实峰值 
E0_bo = 12000;Emin_bo = 1.2;        %骨弹性模量           
% E0_ce=3000;Emin_ce=0.3;           (%骨水泥弹性模量)
E0_cage = 110000; Emin_cage = 11;   %椎间融合器弹性模量                      %% 不改



%% cage 参数 MMA准备

% 外层
design_num=size(desi_ele, 2);
design_cage=zeros(design_num, 1);

for i=1:design_num
    % 每个单元的第二个节点:-x方向
    design_cage(i)=ini_cage(nod_coo(ele_nod(desi_ele(i),1),1),nod_coo(ele_nod(desi_ele(i),1),2),nod_coo(ele_nod(desi_ele(i),1),3)); %椎间融合器初始结构
end
%需要求椎间融合器外层的质量:以确定约束
M0_cage=sum(design_cage);                 %椎间融合器外层初始质量
xold1=design_cage;xold2=design_cage;
cage_xmin=0.001;cage_xmax=1;
% cage_xmin=0.0005;cage_xmax=0.5;                                                %% 改cage_xmax为0.5
% cage_xmin=0.05;cage_xmax=0.3; 
% 优化设置
penal=3;poisson=0.3;
xmin=cage_xmin*ones(design_num,1);xmax=cage_xmax*ones(design_num,1);
low_old=zeros(design_num,1);up_old=zeros(design_num,1);   % 第三次循环会有值

delta=1;t=0;

% 计算每个方向上的力的时候的变量 保存


while(delta>0.0001)
    
save('tempWorkspace.mat');
%%  三次计算 不同方向上的力 
Calculate_Force_1
Force_1.obj_bo = obj_bo;
Force_1.bo_sum = bo_sum;
% Force_1.U1_ele_nod_dir_P = U1_ele_nod_dir_P;
% Force_1.U1_ele_nod_dir_Fv = U1_ele_nod_dir_Fv;
save(['Force_1_', num2str(t) '.mat'], 'Force_1');

Calculate_Force_2
Force_2.obj_bo = obj_bo;
Force_2.bo_sum = bo_sum;
% Force_2.U1_ele_nod_dir_P = U1_ele_nod_dir_P;
% Force_2.U1_ele_nod_dir_Fv = U1_ele_nod_dir_Fv;
save(['Force_2_', num2str(t) '.mat'], 'Force_2');

Calculate_Force_3
Force_3.obj_bo = obj_bo;
Force_3.bo_sum = bo_sum;
% Force_3.U1_ele_nod_dir_P = U1_ele_nod_dir_P;
% Force_3.U1_ele_nod_dir_Fv = U1_ele_nod_dir_Fv;
save(['Force_3_', num2str(t) '.mat'], 'Force_3');

filename_F1 = ['Force_1_', num2str(t) '.mat'];
load(filename_F1);

filename_F2 = ['Force_2_', num2str(t) '.mat'];
load(filename_F2);

filename_F3 = ['Force_3_', num2str(t) '.mat'];
load(filename_F3);



%%%%%% 优化问题求解
%%%%%%目标函数,第p个月骨量最大
% ob=-bo_sum(P+1);
ob = - Force_1.bo_sum(P + 1) - Force_2.bo_sum(P + 1) - Force_3.bo_sum(P + 1);
%%%%% 约束条件1支架应变能约束
g1=C-C0;
%%%%% 约束条件2支架体积分数约束
g2=sum(design_cage)-M0_cage;         %%
% g=[g1;g2];
g=[g2];

%%%%% 求目标函数偏导值
d_ob=zeros(design_num,1);%%
for j=0:P-1
    p_design=zeros(design_num,1);

    % % p
    filename=['F1_U1_ele_nod_dir_P',num2str(j)];
    load(filename)
    Force_1.U1_ele_nod_dir_P = U1_ele_nod_dir_P;

    filename=['F2_U1_ele_nod_dir_P',num2str(j)];
    load(filename)
    Force_2.U1_ele_nod_dir_P = U1_ele_nod_dir_P;

    filename=['F3_U1_ele_nod_dir_P',num2str(j)];
    load(filename)
    Force_3.U1_ele_nod_dir_P = U1_ele_nod_dir_P;

    % % fv
    filename=['F1_U1_ele_nod_dir_Fv',num2str(j)];
    load(filename)
    Force_1.U1_ele_nod_dir_Fv = U1_ele_nod_dir_Fv;

    filename=['F2_U1_ele_nod_dir_Fv',num2str(j)];
    load(filename)
    Force_2.U1_ele_nod_dir_Fv = U1_ele_nod_dir_Fv;

    filename=['F3_U1_ele_nod_dir_Fv',num2str(j)];
    load(filename)
    Force_3.U1_ele_nod_dir_Fv = U1_ele_nod_dir_Fv;

%     filename=['U1_ele_nod_dir_Fv',num2str(j)];
%     load(filename)
    for k=1:design_num
%             U_temp1=U1_ele_nod_dir_P((desi_ele(k)-1)*24+1:(desi_ele(k)-1)*24+24);
            U_temp1_1=Force_1.U1_ele_nod_dir_P((desi_ele(k)-1)*24+1:(desi_ele(k)-1)*24+24);
            U_temp1_2=Force_2.U1_ele_nod_dir_P((desi_ele(k)-1)*24+1:(desi_ele(k)-1)*24+24);
            U_temp1_3=Force_3.U1_ele_nod_dir_P((desi_ele(k)-1)*24+1:(desi_ele(k)-1)*24+24);
 
%             V_temp1=U1_ele_nod_dir_Fv((desi_ele(k)-1)*24+1:(desi_ele(k)-1)*24+24);
            V_temp1_1=Force_1.U1_ele_nod_dir_Fv((desi_ele(k)-1)*24+1:(desi_ele(k)-1)*24+24);
            V_temp1_2=Force_2.U1_ele_nod_dir_Fv((desi_ele(k)-1)*24+1:(desi_ele(k)-1)*24+24);
            V_temp1_3=Force_3.U1_ele_nod_dir_Fv((desi_ele(k)-1)*24+1:(desi_ele(k)-1)*24+24);
 
%         p_design(k)=(V_temp1_1'*B'*D*B*U_temp1_1)* (2 * 0.9748 * design_cage(k) - 0.0262) * E0_cage + (V_temp1_2'*B'*D*B*U_temp1_2)*(2 * 0.9748 * design_cage(k) - 0.0262)* E0_cage+(V_temp1_3'*B'*D*B*U_temp1_3)*(2 * 0.9748 * design_cage(k) - 0.0262)* E0_cage;
            p_design(k)=(V_temp1_1'*B'*D*B*U_temp1_1)* (3*E0_cage*design_cage(k)^2)+ (V_temp1_2'*B'*D*B*U_temp1_2)*(3*E0_cage*design_cage(k)^2)+(V_temp1_3'*B'*D*B*U_temp1_3)*(3*E0_cage*design_cage(k)^2);
    end
    d_ob=d_ob+p_design;
end

%%%支架应变能约束条件偏导值     
d_g1=d_C;
%%%%%支架体积分数约束偏导值 
d_g2=v*ones(1,design_num);
% d_g=[d_g1;d_g2];
d_g=[d_g2];

% 
% %%%% 约束支架密度值 <=1-bone密度
% up_old=zeros(obj_num,1);
% % for i=1:obj_num
% %     up_old(i)=1-max(obj_bo(:,i)/b_max);
% % end
% up_old(up_old<0.001)=0.001;

low=low_old;
up=up_old;
% c       = [1000 1000]';  
% d       = [1 1]';
% a0      = 1;
% a       = [0 0]';
c       = [1000]';  
d       = [1]';
a0      = 1;
a       = [0]';
[xmma,ymma,~,~,~,~,~,~,~,low,up] = mmasub(1,design_num,t,design_cage,xmin,xmax,xold1,xold2,ob,d_ob,g,d_g,low,up,a0,a,c,d);
delta=sum(abs(xmma-design_cage))/design_num;
xold2=xold1;
xold1=design_cage;
design_cage=xmma;
low_old=low;
up_old=up;
obsave(t)=ob;
gsave1(t)=g1;
gsave2(t)=g2;

fprintf('iterition number %6.0f\n',t);
fprintf('object function %10.6f\n',ob);
% fprintf('mass constraint %10.6f\n',g1);
fprintf('mass constraint %10.6f\n',g2);
fprintf('delta %10.6f\n',delta);
% save(['obj_bo',num2str(t) '.mat'],'obj_bo');
save(['ob',num2str(t) '.mat'],'ob');
end


