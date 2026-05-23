load('tempWorkspace.mat');

% 外层设计域
design_cage(design_cage<0.001)=0.001;                                          

design_cage(design_cage>1)=1;  


cage_3d=zeros(N1,N2,N3);
for i=1:design_num
    cage_3d(nod_coo(ele_nod(desi_ele(i),1),1),nod_coo(ele_nod(desi_ele(i),1),2),nod_coo(ele_nod(desi_ele(i),1),3))=design_cage(i);
end
% save([num2str(t) '_F2.mat'],'cage_3d');
t=t+1;
% design_cage inp文件
edit_desicage_inp;  


%%% 骨生长模型
% 内层obj的骨量
obj_num = size(obj_ele, 2);
obj_bo=zeros(P+1,obj_num); %存储真实骨密度
% 初始化骨0.36
for i=1:obj_num
    obj_bo(1,i)=ini_str(nod_coo(ele_nod(obj_ele(i),1),1),nod_coo(ele_nod(obj_ele(i),1),2),nod_coo(ele_nod(obj_ele(i),1),3)); %赋初始骨结构
end
bo_sum=zeros(P+1,1);                    %每个月的总骨量(目标函数）
bo_sum(1)=sum(obj_bo(1,:));             % ?

%fprintf('bo_sum(1): %10.6f\n', bo_sum(1));

E_obj=zeros(P+1,obj_num);               % 内层目标域的骨的弹性模量
bone_s=zeros(P,obj_num);
bone_d=zeros(P,obj_num);

E_cage = zeros(P+1, design_num);          % 内层设计域cage的弹性模量
d_Ecage = zeros(P+1, design_num);


ti=0;
% 从第0个月（初始结构）长到第p个月；计算骨量 目标函数 第p个月的骨量最大
while(ti<P)              
% 目标域骨的弹性模量
obj_bo(obj_bo<=0.001)=0.001;
for i=1:obj_num
    %E_mix(ti+1,i)=Emin_bo+E0_bo*(obj_bo(ti+1,i)/b_max)^3+   Emin_ce+E0_ce*design_ce(i)^3;
    E_obj(ti+1,i)=Emin_bo+E0_bo*(obj_bo(ti+1,i)/b_max)^3;
end
E_obj(E_obj>E0_bo)=E0_bo;

% 设计域cage的弹性模量 GUSHENGZHANGXUNHUANWAI
design_cage(design_cage<0.001)=0.001; 
for i = 1 : design_num       
    E_cage(ti + 1, i) = Emin_cage + E0_cage * design_cage(i) ^ 3;      
%     E_cage(ti + 1, i) = (0.9748 * design_cage(i) ^ 2 - 0.0262 * design_cage(i))* E0_cage + Emin_cage;
%     d_Ecage(ti + 1, i) = (2 * 0.9748 * design_cage(i) - 0.0262)* E0_cage;
end
% (0.9748 * 0.3 ^ 2 - 0.0262 * 0.3)* E0_cage + Emin_cage = 8797
% E_cage(E_cage > 8797) = 8797;      
E_cage(E_cage > E0_cage) = E0_cage; 
             
edit_objbo_inp_F2; %骨inp文件 + E_obj文件  %%todo end1.inp file  output:vert.inp
run_abaqus;

if exist('U1.txt', 'file')
    delete('U1.txt');
end

system("abaqus cae noGUI=odbFieldOutput1.py");

if t==1||mod(t,10)==0
    file_name=[num2str(t), '_F2_.odb'];
    copyfile('vert.odb',file_name);
end
delete('vert.odb');

load U1.txt
UU1=sortrows(U1,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% U_ele_nod_dir
U1_ele_nod_dir_P=zeros(size(ele_nod,1)*24,1);
for i=1:size(ele_nod,1)
    ct1=0;
    U_temp1=zeros(24,1);
    for ii=1:8
        U_temp1(ct1+1)=UU1(ele_nod(i,ii),2);
        U_temp1(ct1+2)=UU1(ele_nod(i,ii),3);
        U_temp1(ct1+3)=UU1(ele_nod(i,ii),4);
        ct1=ct1+3;
    end
    U1_ele_nod_dir_P((i-1)*24+1:(i-1)*24+24)=U_temp1;
end
save(['U1_ele_nod_dir_P' num2str(ti)],'U1_ele_nod_dir_P');%%三个方向的位移
% %  更正
copyfile(['U1_ele_nod_dir_P' num2str(ti) '.mat'], ['F2_U1_ele_nod_dir_P' num2str(ti) '.mat']);
% %


if ti==0   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 支架的应变能约束
    C=0;
    d_C=zeros(1,design_num);
    for i=1:design_num
        U_temp1=zeros(24,1);
        for ii=1:24
            U_temp1=U1_ele_nod_dir_P((desi_ele(i)-1)*24+1:(desi_ele(i)-1)*24+24);
        end   
        sen=1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_cage+E0_cage*design_cage(i)^3); 
        d_C(i)=-(1/2)*(U_temp1'*B'*D*B*U_temp1)*(3*E0_cage*design_cage(i)^2);
%         sen=1/2*(U_temp1'*B'*D*B*U_temp1)*(0.9748 * design_cage(i) ^ 2 - 0.0262 * design_cage(i))* E0_cage + Emin_cage; 
%         d_C(i)=-(1/2)*(U_temp1'*B'*D*B*U_temp1)*(2 * 0.9748 * design_cage(i) - 0.0262)* E0_cage;
        C=C+sen;     %
    end
end


filename=['U1_ele_nod_dir_P',num2str(ti)];
load(filename);
for i=1:obj_num
    U_temp1=zeros(24,1);
    for ii=1:24
%         U_temp1=U1_ele_nod_dir_P((desi_ele(i)-1)*24+1:(desi_ele(i)-1)*24+24);
        U_temp1=U1_ele_nod_dir_P((obj_ele(i)-1)*24+1:(obj_ele(i)-1)*24+24);
    end   
    bone_s(ti+1,i)=1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_bo+E0_bo*(obj_bo(ti+1,i)/b_max)^3)/(obj_bo(ti+1,i));   %%%计算BONE stimulus s=应变能/(密度）
    bone_d(ti+1,i)=bone_delta(bone_s(ti+1,i));                  % bone_delta需要改吗？？？
    obj_bo(ti+2,i)=obj_bo(ti+1,i)+bone_d(ti+1,i)*dt; 
end

if(ti==0)&&(t==1)   %%%求支架初始结构的应变能
    bone_sen=0;
    filename=['U1_ele_nod_dir_P',num2str(ti)];
    load(filename);
    for i=1:obj_num
        U_temp1=zeros(24,1);
        for ii=1:24
%             U_temp1=U1_ele_nod_dir_P((desi_ele(i)-1)*24+1:(desi_ele(i)-1)*24+24);
            U_temp1=U1_ele_nod_dir_P((obj_ele(i)-1)*24+1:(obj_ele(i)-1)*24+24);
        end
        bone_sen=bone_sen+1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_bo+E0_bo*(obj_bo(ti+1,i)/b_max)^3);
    end
    C0=1.5*bone_sen; %%%%支架的初始应变能为松质骨结构的2.5倍
end


obj_bo(obj_bo<0.001)=0.001;
obj_bo(obj_bo>b_max)=b_max;
bo_sum(ti+2)=sum(obj_bo(ti+2,:));


%fprintf('bo_sum(%d): %10.6f\n', ti + 2, bo_sum(ti+2));


ti=ti+1;
end
save(['bone_s_F2_',num2str(t) '.mat'],'bone_s');
save(['obj_bo_F2_',num2str(t) '.mat'],'obj_bo');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%求phi函数
% Fai=zeros(design_num*24,P);%%!!!!!!!!!!!!!!!!!!!!obj_num?
% Fai(:,P)=ones(design_num*24,1);
Fai=zeros(obj_num*24,P);%%!!!!!!!!!!!!!!!!!!!!obj_num?
Fai(:,P)=ones(obj_num*24,1);
for tn=P-1:-1:0
    %%%%%%%%% %求K(P-1)
    filename=['U1_ele_nod_dir_P',num2str(tn)];    
    load(filename)
    Fv=zeros(size(nod_coo,1)*3,1);
    for i=1:obj_num
        U_temp1=zeros(24,1);
        for ii=1:24
%             U_temp1=U1_ele_nod_dir_P((desi_ele(i)-1)*24+1:(desi_ele(i)-1)*24+24);
            U_temp1=U1_ele_nod_dir_P((obj_ele(i)-1)*24+1:(obj_ele(i)-1)*24+24);
        end 
        b_s=1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_bo+E0_bo*(obj_bo(tn+1,i)/b_max)^3)/(obj_bo(tn+1,i));     %%%计算BONE stimulus s=应变能/(密度） 
        ai=d_bone_delta(b_s);  %%%%%%ai,n  
        tt_i=ai/(v*obj_bo(tn+1,i));
        ki=B'*D*B*(Emin_bo+E0_bo*(obj_bo(tn+1,i)/b_max)^3);
        ui=U_temp1;
        temp=Fai((i-1)*24+1:(i-1)*24+24,tn+1).*(tt_i*ki*ui);           %%%%%%%%%%%%%%%%%%%%%%%%%%%式38
        for iii=1:8
            % !!!!!!!!!!!!!!!!!!!
%             Fv((ele_nod(desi_ele(i),iii)-1)*3+1:(ele_nod(desi_ele(i),iii)-1)*3+3)=Fv((ele_nod(desi_ele(i),iii)-1)*3+1:(ele_nod(desi_ele(i),iii)-1)*3+3)+temp((iii-1)*3+1:(iii-1)*3+3);
            Fv((ele_nod(obj_ele(i),iii)-1)*3+1:(ele_nod(obj_ele(i),iii)-1)*3+3)=Fv((ele_nod(obj_ele(i),iii)-1)*3+1:(ele_nod(obj_ele(i),iii)-1)*3+3)+temp((iii-1)*3+1:(iii-1)*3+3);
        end
    end
%     Fv_set=find(Fv~=0);%%%%%保留有力的结点
%     Fv_set(1,:)=[];
%     Fv_set=unique(round(Fv_set/3));%%%去重
    
    F_existed  = reshape(Fv,[3,size(Fv,1)/3]);  %reshapr重构数组
    Fv_sets=F_existed(1,:)|F_existed(2,:)|F_existed(3,:);   %判断一个节点x\y\z方向是否全为0，若全为0则删除
    Fv_set=find(Fv_sets==1)';   %find:返回非0元素下标


    editF_desiele_inp;
    run_abaqus;
    
    
    if exist('U1.txt', 'file')
        delete('U1.txt');
    end
    
    system("abaqus cae noGUI=odbFieldOutput1.py");
    load U1.txt
    UU1=sortrows(U1,1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%求Mu(n+1)
    U1_ele_nod_dir_Fv=zeros(size(ele_nod,1)*24,1);
    for i=1:size(ele_nod,1)
        ct1=0;
        U_temp1=zeros(24,1);
        for ii=1:8
            U_temp1(ct1+1)=UU1(ele_nod(i,ii),2);
            U_temp1(ct1+2)=UU1(ele_nod(i,ii),3);
            U_temp1(ct1+3)=UU1(ele_nod(i,ii),4);
            ct1=ct1+3;
        end
        U1_ele_nod_dir_Fv((i-1)*24+1:(i-1)*24+24)=U_temp1;
    end
    save(['U1_ele_nod_dir_Fv' num2str(tn)],'U1_ele_nod_dir_Fv');
    copyfile(['U1_ele_nod_dir_Fv' num2str(tn) '.mat'], ['F2_U1_ele_nod_dir_Fv' num2str(tn) '.mat']);

    if tn==0
        break;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 式39，求phi（n）
    filename=['U1_ele_nod_dir_Fv',num2str(tn)];
    load(filename)
    filename=['U1_ele_nod_dir_P',num2str(tn)];
    load(filename)                          
    for i=1:obj_num
        for ii=1:24
%             U_temp1=U1_ele_nod_dir_P((desi_ele(i)-1)*24+1:(desi_ele(i)-1)*24+24);%%%%%%n+1
            U_temp1=U1_ele_nod_dir_P((obj_ele(i)-1)*24+1:(obj_ele(i)-1)*24+24);%%%%%%n+1
        end  
        for ii=1:24
%             V_temp1=U1_ele_nod_dir_Fv((desi_ele(i)-1)*24+1:(desi_ele(i)-1)*24+24);%%%%%%n+1
            V_temp1=U1_ele_nod_dir_Fv((obj_ele(i)-1)*24+1:(obj_ele(i)-1)*24+24);%%%%%%n+1
        end  
       b_s=1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_bo+E0_bo*(obj_bo(tn+1,i)/b_max)^3)/(obj_bo(tn+1,i));     %%%计算BONE stimulus s=应变能/(密度）    
       ai=d_bone_delta(b_s);  %%%%%%ai,n+1
       seni=1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_bo+E0_bo*(obj_bo(tn+1,i)/b_max)^3);%%%%%si,n+1
       tt_i=(2*ai*seni)/(v*obj_bo(tn+1,i)^2);
       t_boi=(V_temp1'*B'*D*B*U_temp1)*3*obj_bo(tn+1,i)^2*E0_bo/b_max^3;
       Fai((i-1)*24+1:(i-1)*24+24,tn)=(1+tt_i)*Fai((i-1)*24+1:(i-1)*24+24,tn+1)-t_boi;
    end
end
