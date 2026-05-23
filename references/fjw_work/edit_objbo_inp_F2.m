bo_0_count=0;bo_0=[];bo_1_count=0;bo_1=[];
bo_2_count=0;bo_2=[];bo_3_count=0;bo_3=[];
bo_4_count=0;bo_4=[];bo_5_count=0;bo_5=[];
bo_6_count=0;bo_6=[];bo_7_count=0;bo_7=[];
bo_8_count=0;bo_8=[];bo_9_count=0;bo_9=[];
bo_10_count=0;bo_10=[];

for i=1:obj_num
    switch round(obj_bo(ti+1,i)/b_max*10)   % A = [1 2 3 4 5] a = 5  则 A/a*20  = [4	8 12 16	20]
        case 0
            bo_0_count=bo_0_count+1;
            bo_0(bo_0_count)=obj_ele(i);       
        case 1
            bo_1_count=bo_1_count+1;
            bo_1(bo_1_count)=obj_ele(i);
        case 2
            bo_2_count=bo_2_count+1;
            bo_2(bo_2_count)=obj_ele(i);            
        case 3
            bo_3_count=bo_3_count+1;
            bo_3(bo_3_count)=obj_ele(i);              
        case 4
            bo_4_count=bo_4_count+1;
            bo_4(bo_4_count)=obj_ele(i);             
        case 5
            bo_5_count=bo_5_count+1;
            bo_5(bo_5_count)=obj_ele(i);              
        case 6
            bo_6_count=bo_6_count+1;
            bo_6(bo_6_count)=obj_ele(i);             
        case 7
            bo_7_count=bo_7_count+1;
            bo_7(bo_7_count)=obj_ele(i);              
        case 8
            bo_8_count=bo_8_count+1;
            bo_8(bo_8_count)=obj_ele(i);                
        case 9
            bo_9_count=bo_9_count+1;
            bo_9(bo_9_count)=obj_ele(i);          
        case 10
            bo_10_count=bo_10_count+1;
            bo_10(bo_10_count)=obj_ele(i);
    end
end


%%
%% 原来是骨水泥和骨的混合Emix等分20份  switch round(E_obj(ti+1,i)/E0_bo*20)
%% 现在单纯将目标域的骨Eobj等分20份？

ele_0_count=0;ele_0=[];ele_1_count=0;ele_1=[];
ele_2_count=0;ele_2=[];ele_3_count=0;ele_3=[];
ele_4_count=0;ele_4=[];ele_5_count=0;ele_5=[];
ele_6_count=0;ele_6=[];ele_7_count=0;ele_7=[];
ele_8_count=0;ele_8=[];ele_9_count=0;ele_9=[];
ele_10_count=0;ele_10=[];

for i=1:obj_num
%     switch round(E_mix(ti+1,i)/E0_bo*20)  
     switch round(E_obj(ti+1,i)/E0_bo*10)
        case 0
            ele_0_count=ele_0_count+1;
            ele_0(ele_0_count)=obj_ele(i);       
        case 1
            ele_1_count=ele_1_count+1;
            ele_1(ele_1_count)=obj_ele(i);
        case 2
            ele_2_count=ele_2_count+1;
            ele_2(ele_2_count)=obj_ele(i);            
        case 3
            ele_3_count=ele_3_count+1;
            ele_3(ele_3_count)=obj_ele(i);              
        case 4
            ele_4_count=ele_4_count+1;
            ele_4(ele_4_count)=obj_ele(i);             
        case 5
            ele_5_count=ele_5_count+1;
            ele_5(ele_5_count)=obj_ele(i);              
        case 6
            ele_6_count=ele_6_count+1;
            ele_6(ele_6_count)=obj_ele(i);             
        case 7
            ele_7_count=ele_7_count+1;
            ele_7(ele_7_count)=obj_ele(i);              
        case 8
            ele_8_count=ele_8_count+1;
            ele_8(ele_8_count)=obj_ele(i);                
        case 9
            ele_9_count=ele_9_count+1;
            ele_9(ele_9_count)=obj_ele(i);          
        case 10
            ele_10_count=ele_10_count+1;
            ele_10(ele_10_count)=obj_ele(i);
    end
end


%% 划分100份desi_cage

cages = cell(101, 1);
cage_counts = zeros(101, 1);

for i = 1:design_num
    index = round(E_cage(ti+1, i) / (E0_cage) * 100);
    cage_counts(index+1) = cage_counts(index+1) + 1;
    cages{index+1}(cage_counts(index+1)) = desi_ele(i);
end

cage_labels = cell(101, 1);
for i = 0:100
    cage_labels{i+1} = sprintf('desi_e_ele%d', i);
end


if exist('ini_noend.inp','file')
delete('ini_noend.inp');
else
end
copyfile('ini_desicage.inp','ini_noend.inp');
fid=fopen('ini_noend.inp','a');


fprintf(fid,'%s\r\n','*Elset,elset=obj_bo_ele0');
for i=1:size(bo_0,2)
    if (rem(i,10)==0)||(i==size(bo_0,2))
        fprintf(fid,'%7.0f',bo_0(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',bo_0(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_1
fprintf(fid,'%s\r\n','*Elset,elset=obj_bo_ele1');
for i=1:size(bo_1,2)
    if (rem(i,10)==0)||(i==size(bo_1,2))
        fprintf(fid,'%7.0f',bo_1(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',bo_1(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_2
fprintf(fid,'%s\r\n','*Elset,elset=obj_bo_ele2');
for i=1:size(bo_2,2)
    if (rem(i,10)==0)||(i==size(bo_2,2))
        fprintf(fid,'%7.0f',bo_2(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',bo_2(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_3
fprintf(fid,'%s\r\n','*Elset,elset=obj_bo_ele3');
for i=1:size(bo_3,2)
    if (rem(i,10)==0)||(i==size(bo_3,2))
        fprintf(fid,'%7.0f',bo_3(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',bo_3(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_4
fprintf(fid,'%s\r\n','*Elset,elset=obj_bo_ele4');
for i=1:size(bo_4,2)
    if (rem(i,10)==0)||(i==size(bo_4,2))
        fprintf(fid,'%7.0f',bo_4(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',bo_4(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_5
fprintf(fid,'%s\r\n','*Elset,elset=obj_bo_ele5');
for i=1:size(bo_5,2)
    if (rem(i,10)==0)||(i==size(bo_5,2))
        fprintf(fid,'%7.0f',bo_5(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',bo_5(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_6
fprintf(fid,'%s\r\n','*Elset,elset=obj_bo_ele6');
for i=1:size(bo_6,2)
    if (rem(i,10)==0)||(i==size(bo_6,2))
        fprintf(fid,'%7.0f',bo_6(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',bo_6(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_7
fprintf(fid,'%s\r\n','*Elset,elset=obj_bo_ele7');
for i=1:size(bo_7,2)
    if (rem(i,10)==0)||(i==size(bo_7,2))
        fprintf(fid,'%7.0f',bo_7(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',bo_7(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_8
fprintf(fid,'%s\r\n','*Elset,elset=obj_bo_ele8');
for i=1:size(bo_8,2)
    if (rem(i,10)==0)||(i==size(bo_8,2))
        fprintf(fid,'%7.0f',bo_8(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',bo_8(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_9
fprintf(fid,'%s\r\n','*Elset,elset=obj_bo_ele9');
for i=1:size(bo_9,2)
    if (rem(i,10)==0)||(i==size(bo_9,2))
        fprintf(fid,'%7.0f',bo_9(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',bo_9(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_10
fprintf(fid,'%s\r\n','*Elset,elset=obj_bo_ele10');
for i=1:size(bo_10,2)
    if (rem(i,10)==0)||(i==size(bo_10,2))
        fprintf(fid,'%7.0f',bo_10(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',bo_10(i));
        fprintf(fid,'%s\t',',');
    end
end


%%
%鍐欏叆鍏冪礌闆唃raft_0
fprintf(fid,'%s\r\n','*Elset,elset=obj_e_ele0');
for i=1:size(ele_0,2)
    if (rem(i,10)==0)||(i==size(ele_0,2))
        fprintf(fid,'%7.0f',ele_0(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',ele_0(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_1
fprintf(fid,'%s\r\n','*Elset,elset=obj_e_ele1');
for i=1:size(ele_1,2)
    if (rem(i,10)==0)||(i==size(ele_1,2))
        fprintf(fid,'%7.0f',ele_1(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',ele_1(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_2
fprintf(fid,'%s\r\n','*Elset,elset=obj_e_ele2');
for i=1:size(ele_2,2)
    if (rem(i,10)==0)||(i==size(ele_2,2))
        fprintf(fid,'%7.0f',ele_2(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',ele_2(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_3
fprintf(fid,'%s\r\n','*Elset,elset=obj_e_ele3');
for i=1:size(ele_3,2)
    if (rem(i,10)==0)||(i==size(ele_3,2))
        fprintf(fid,'%7.0f',ele_3(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',ele_3(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_4
fprintf(fid,'%s\r\n','*Elset,elset=obj_e_ele4');
for i=1:size(ele_4,2)
    if (rem(i,10)==0)||(i==size(ele_4,2))
        fprintf(fid,'%7.0f',ele_4(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',ele_4(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_5
fprintf(fid,'%s\r\n','*Elset,elset=obj_e_ele5');
for i=1:size(ele_5,2)
    if (rem(i,10)==0)||(i==size(ele_5,2))
        fprintf(fid,'%7.0f',ele_5(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',ele_5(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_6
fprintf(fid,'%s\r\n','*Elset,elset=obj_e_ele6');
for i=1:size(ele_6,2)
    if (rem(i,10)==0)||(i==size(ele_6,2))
        fprintf(fid,'%7.0f',ele_6(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',ele_6(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_7
fprintf(fid,'%s\r\n','*Elset,elset=obj_e_ele7');
for i=1:size(ele_7,2)
    if (rem(i,10)==0)||(i==size(ele_7,2))
        fprintf(fid,'%7.0f',ele_7(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',ele_7(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_8
fprintf(fid,'%s\r\n','*Elset,elset=obj_e_ele8');
for i=1:size(ele_8,2)
    if (rem(i,10)==0)||(i==size(ele_8,2))
        fprintf(fid,'%7.0f',ele_8(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',ele_8(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_9
fprintf(fid,'%s\r\n','*Elset,elset=obj_e_ele9');
for i=1:size(ele_9,2)
    if (rem(i,10)==0)||(i==size(ele_9,2))
        fprintf(fid,'%7.0f',ele_9(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',ele_9(i));
        fprintf(fid,'%s\t',',');
    end
end
%鍐欏叆鍏冪礌闆唃raft_10
fprintf(fid,'%s\r\n','*Elset,elset=obj_e_ele10');
for i=1:size(ele_10,2)
    if (rem(i,10)==0)||(i==size(ele_10,2))
        fprintf(fid,'%7.0f',ele_10(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',ele_10(i));
        fprintf(fid,'%s\t',',');
    end
end

%%
for j = 1:101
    fprintf(fid, '*Elset,elset=%s\r\n', cage_labels{j});
    for i = 1:size(cages{j}, 2)
        if (rem(i, 10) == 0) || (i == size(cages{j}, 2))
            fprintf(fid, '%7.0f,\r\n', cages{j}(i));
        else
            fprintf(fid, '%7.0f,\t', cages{j}(i));
        end
    end
end









fclose(fid);
if exist('vert.inp','file')
delete('vert.inp');
else
end
fidA=fopen('ini_noend.inp','r');
fidB=fopen('end2.inp','r');
DataA=fread(fidA);
DataB=fread(fidB);
fidC=fopen('vert.inp','w');
fwrite(fidC,DataA);
fwrite(fidC,DataB);
fclose(fidA);
fclose(fidB);
fclose(fidC);










