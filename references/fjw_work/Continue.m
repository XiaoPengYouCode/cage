 

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