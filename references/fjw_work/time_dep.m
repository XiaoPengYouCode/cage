%%
% desi_ele： 椎间融合器外层设计域的单元 1*n  找出该部分的结构x
% design_cage：椎间融合器外层设计域每单元的质量分布 n*1 列数表示单元序号
% cage_3D：  椎间融合器外层设计域在三维空间上的分布，将design_cage每单元对应的值赋值给cage_3D [X Y Z] 上对应的值

% obj_ele：  椎间融合器内层目标域的单元 1*n  以该部分的骨量最大为目标
% obj_bo：   椎间融合器内层目标域的每单元的骨量分布  每一行为一个月


%%
close('all');
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


ini_cage = 0.2 * ones(N1, N2, N3);  %初始椎间融合器分布
% ini_cage = 0.3 * ones(N1, N2, N3);  %初始椎间融合器分布
% ini_cage = 0.4 * ones(N1, N2, N3);  %初始椎间融合器分布

% v = 1;
% v = 0.4 ^ 3;
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
    % 每个单元的第1个节点:-x方向
    design_cage(i)=ini_cage(nod_coo(ele_nod(desi_ele(i),1),1),nod_coo(ele_nod(desi_ele(i),1),2),nod_coo(ele_nod(desi_ele(i),1),3)); %椎间融合器初始结构
end
%需要求椎间融合器外层的质量:以确定约束
M0_cage=sum(design_cage);                 %椎间融合器外层初始质量
xold1=design_cage;xold2=design_cage;
% cage_xmin=0.001;cage_xmax=1;
% cage_xmin=0.0005;cage_xmax=0.5;                                                %% 改cage_xmax为0.5
cage_xmin=0.001;cage_xmax=1;
% 优化设置
penal=3;poisson=0.3;
xmin=cage_xmin*ones(design_num,1);xmax=cage_xmax*ones(design_num,1);
low_old=zeros(design_num,1);up_old=zeros(design_num,1);   % 第三次循环会有值

delta=1;t=0;

while(delta>0.0001)

    save('tempWorkspace.mat');

    % 外层设计域
    design_cage(design_cage<0.001)=0.001;                                           %  分成100份最小值还是0.001吗
    % design_cage(design_cage<0.0005)=0.0005;                                       %% 改desi_cage最小值
    % design_cage(design_cage>0.5)=0.5;
    design_cage(design_cage>1)=1;

    cage_3d=zeros(N1,N2,N3);
    for i=1:design_num
        cage_3d(nod_coo(ele_nod(desi_ele(i),1),1),nod_coo(ele_nod(desi_ele(i),1),2),nod_coo(ele_nod(desi_ele(i),1),3))=design_cage(i);
    end
    save([num2str(t) '.mat'],'cage_3d');
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
    bo_sum(1)=sum(obj_bo(1,:));             % 

    %fprintf('bo_sum(1): %10.6f\n', bo_sum(1));

    E_obj=zeros(P+1,obj_num);               % 内层目标域的骨的弹性模量
    bone_s=zeros(P,obj_num);
    bone_d=zeros(P,obj_num);

    E_cage = zeros(P+1, design_num);          % 内层设计域cage的弹性模量



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
            %     E_cage(ti + 1, i) = Emin_cage + E0_cage * design_cage(i) ^ 3;
            E_cage(ti + 1, i) = Emin_cage + E0_cage * design_cage(i) ^ 3;
        end
        % E_cage(E_cage > 0.5^3 * E0_cage) = 0.5^3 * E0_cage;                         %% 现在上限为0.5^3*E0_cage
        E_cage(E_cage > E0_cage) = E0_cage;

        edit_objbo_inp; %骨inp文件 + E_obj文件  %%todo end1.inp file  output:vert.inp
        run_abaqus;

        system("abaqus cae noGUI=odbFieldOutput1.py");
        %
        % if t==1||mod(t,10)==0
        %     file_name=[num2str(t), '.odb'];
        %     copyfile('vert.odb',file_name);
        % end
        % delete('vert.odb');

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

        if ti==0   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 支架的应变能约束
            C=0;
            d_C=zeros(1,design_num);
            for i=1:design_num
                U_temp1=zeros(24,1);
                for ii=1:24
                    U_temp1=U1_ele_nod_dir_P((desi_ele(i)-1)*24+1:(desi_ele(i)-1)*24+24);
                end
                %         sen=1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_ce+E0_ce*design_ce(i)^3);
                sen=1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_cage+E0_cage*design_cage(i)^3);
                d_C(i)=-(1/2)*(U_temp1'*B'*D*B*U_temp1)*(3*E0_cage*design_cage(i)^2);
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
            %     bone_s(ti+1,i)=1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_bo+E0_bo*(obj_bo(ti+1,i)/b_max)^3)/(obj_bo(ti+1,i) * v);   %%%计算BONE stimulus s=应变能/(密度） 替代： 应变能/质量(g)
            bone_s(ti+1,i)=1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_bo+E0_bo*(obj_bo(ti+1,i)/b_max)^3)/(obj_bo(ti+1,i));   %%%计算BONE stimulus s=应变能/(密度） 替代： 应变能/质量(g)
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
    save(['bone_s',num2str(t) '.mat'],'bone_s');


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%求phi函数
    %%% 是设计域还是目标域
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
            %         b_s=1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_bo+E0_bo*(obj_bo(tn+1,i)/b_max)^3)/(obj_bo(tn+1,i)*v);     %%%计算BONE stimulus s=应变能/(体积*密度）
            b_s=1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_bo+E0_bo*(obj_bo(tn+1,i)/b_max)^3)/(obj_bo(tn+1,i));     %%%计算BONE stimulus s=应变能/(体积*密度）
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
            %        b_s=1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_bo+E0_bo*(obj_bo(tn+1,i)/b_max)^3)/(obj_bo(tn+1,i)*v);     %%%计算BONE stimulus s=应变能/(体积*密度）
            b_s=1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_bo+E0_bo*(obj_bo(tn+1,i)/b_max)^3)/(obj_bo(tn+1,i));     %%%计算BONE stimulus s=应变能/(体积*密度）
            ai=d_bone_delta(b_s);  %%%%%%ai,n+1
            seni=1/2*(U_temp1'*B'*D*B*U_temp1)*(Emin_bo+E0_bo*(obj_bo(tn+1,i)/b_max)^3);%%%%%si,n+1
            tt_i=(2*ai*seni)/(v*obj_bo(tn+1,i)^2);
            t_boi=(V_temp1'*B'*D*B*U_temp1)*3*obj_bo(tn+1,i)^2*E0_bo/b_max^3;
            Fai((i-1)*24+1:(i-1)*24+24,tn)=(1+tt_i)*Fai((i-1)*24+1:(i-1)*24+24,tn+1)-t_boi;
        end
    end

    save('bo_sum_iter1','bo_sum');

    %%%%%% 优化问题求解
    %%%%%%目标函数,第p个月骨量最大
    ob=-bo_sum(P+1);
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
        filename=['U1_ele_nod_dir_P',num2str(j)];
        load(filename)
        filename=['U1_ele_nod_dir_Fv',num2str(j)];
        load(filename)
        for k=1:design_num
            U_temp1=U1_ele_nod_dir_P((desi_ele(k)-1)*24+1:(desi_ele(k)-1)*24+24);
            V_temp1=U1_ele_nod_dir_Fv((desi_ele(k)-1)*24+1:(desi_ele(k)-1)*24+24);
            p_design(k)=  (V_temp1'*B'*D*B*U_temp1)*(3*E0_cage*design_cage(k)^2);
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
    fprintf('mass constraint %10.6f\n',g1);
    fprintf('compliance constraint %10.6f\n',g2);
    fprintf('delta %10.6f\n',delta);
    save(['obj_bo',num2str(t) '.mat'],'obj_bo');

end



















