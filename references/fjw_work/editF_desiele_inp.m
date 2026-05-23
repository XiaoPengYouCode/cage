ele_0_count=0;ele_0=[];ele_1_count=0;ele_1=[];
ele_2_count=0;ele_2=[];ele_3_count=0;ele_3=[];
ele_4_count=0;ele_4=[];ele_5_count=0;ele_5=[];
ele_6_count=0;ele_6=[];ele_7_count=0;ele_7=[];
ele_8_count=0;ele_8=[];ele_9_count=0;ele_9=[];
ele_10_count=0;ele_10=[];

for i=1:obj_num
     switch round(E_obj(tn+1,i)/E0_bo*10)
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


cage_0_count=0;cage_0=[];cage_1_count=0;cage_1=[];
cage_2_count=0;cage_2=[];cage_3_count=0;cage_3=[];
cage_4_count=0;cage_4=[];cage_5_count=0;cage_5=[];
cage_6_count=0;cage_6=[];cage_7_count=0;cage_7=[];
cage_8_count=0;cage_8=[];cage_9_count=0;cage_9=[];
cage_10_count=0;cage_10=[];

for i=1:design_num
    %switch round(E_cage(tn+1,i)/E0_cage*10)
%     switch round(E_cage(ti+1,i)/(0.5^3*E0_cage)*10)  
    switch round(E_cage(tn+1,i)/(E0_cage)*10)  
        case 0
            cage_0_count=cage_0_count+1;
            cage_0(cage_0_count)=desi_ele(i);       
        case 1
            cage_1_count=cage_1_count+1;
            cage_1(cage_1_count)=desi_ele(i);
        case 2
            cage_2_count=cage_2_count+1;
            cage_2(cage_2_count)=desi_ele(i);            
        case 3
            cage_3_count=cage_3_count+1;
            cage_3(cage_3_count)=desi_ele(i);              
        case 4
            cage_4_count=cage_4_count+1;
            cage_4(cage_4_count)=desi_ele(i);             
        case 5
            cage_5_count=cage_5_count+1;
            cage_5(cage_5_count)=desi_ele(i);              
        case 6
            cage_6_count=cage_6_count+1;
            cage_6(cage_6_count)=desi_ele(i);             
        case 7
            cage_7_count=cage_7_count+1;
            cage_7(cage_7_count)=desi_ele(i);              
        case 8
            cage_8_count=cage_8_count+1;
            cage_8(cage_8_count)=desi_ele(i);                
        case 9
            cage_9_count=cage_9_count+1;
            cage_9(cage_9_count)=desi_ele(i);          
        case 10
            cage_10_count=cage_10_count+1;
            cage_10(cage_10_count)=desi_ele(i);
    end
end

%%

cages = cell(101, 1);
cage_counts = zeros(101, 1);

for i = 1:design_num
    index = round(E_cage(tn+1, i) / (E0_cage) * 100);
    cage_counts(index+1) = cage_counts(index+1) + 1;
    cages{index+1}(cage_counts(index+1)) = desi_ele(i);
end

cage_labels = cell(101, 1);
for i = 0:100
    cage_labels{i+1} = sprintf('desi_e_ele%d', i);
end

%%






if exist('ini_noend.inp','file')
delete('ini_noend.inp');
else
end
copyfile('ini_desicage.inp','ini_noend.inp');
fid=fopen('ini_noend.inp','a');


%%
%写入元素集graft_0
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
%写入元素集graft_1
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
%写入元素集graft_2
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
%写入元素集graft_3
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
%写入元素集graft_4
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
%写入元素集graft_5
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
%写入元素集graft_6
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
%写入元素集graft_7
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
%写入元素集graft_8
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
%写入元素集graft_9
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
%写入元素集graft_10
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

%% 100�� desi_e_ele



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











%%


fclose(fid);
if exist('vert.inp','file')
delete('vert.inp');
else
end
fidA=fopen('ini_noend.inp','r');
fidB=fopen('end_Fv_p1.inp','r'); % !!
fidC=fopen('vert.inp','w');
fidD=fopen('end_Fv_p2.inp','r'); % !!

DataA=fread(fidA);
DataB=fread(fidB);
DataD=fread(fidD);

fwrite(fidC,DataA);
fwrite(fidC,DataB);

for i=1:size(Fv_set,1)
    fprintf(fidC,'%s\r\n',['** Name: CFORCE-' num2str(i)  'Type: Concentrated force']);
    fprintf(fidC,'%s\r\n','*Cload, op=NEW');
    fprintf(fidC,'%s\r\n',['VERT-1.' num2str(Fv_set(i)), ', 1, ' num2str(Fv((Fv_set(i)-1)*3+1))]);
    fprintf(fidC,'%s\r\n',['VERT-1.' num2str(Fv_set(i)), ', 2, ' num2str(Fv((Fv_set(i)-1)*3+2))]);
    fprintf(fidC,'%s\r\n',['VERT-1.' num2str(Fv_set(i)), ', 3, ' num2str(Fv((Fv_set(i)-1)*3+3))]);
end

fwrite(fidC,DataD);

fclose(fidA);
fclose(fidB);
fclose(fidC);
fclose(fidD);

