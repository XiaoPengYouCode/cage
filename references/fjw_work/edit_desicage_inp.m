cage_0_count=0;cage_0=[];cage_1_count=0;cage_1=[];
cage_2_count=0;cage_2=[];cage_3_count=0;cage_3=[];
cage_4_count=0;cage_4=[];cage_5_count=0;cage_5=[];
cage_6_count=0;cage_6=[];cage_7_count=0;cage_7=[];
cage_8_count=0;cage_8=[];cage_9_count=0;cage_9=[];
cage_10_count=0;cage_10=[];

for i=1:design_num
    switch round(design_cage(i)*10)
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

if exist('ini_desicage.inp','file')
delete('ini_desicage.inp');
else
end
copyfile('ini_nodesi.inp','ini_desicage.inp');
fid=fopen('ini_desicage.inp','a');


fprintf(fid,'%s\r\n','*Elset,elset=desi_ele');
for i=1:size(desi_ele,2)
    if (rem(i,10)==0)||(i==size(desi_ele,2))
        fprintf(fid,'%7.0f',desi_ele(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',desi_ele(i));
        fprintf(fid,'%s\t',',');
    end
end

%写入元素集cage_0
fprintf(fid,'%s\r\n','*Elset,elset=desi_ele0');
for i=1:size(cage_0,2)
    if (rem(i,10)==0)||(i==size(cage_0,2))
        fprintf(fid,'%7.0f',cage_0(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',cage_0(i));
        fprintf(fid,'%s\t',',');
    end
end
%写入元素集cage_1
fprintf(fid,'%s\r\n','*Elset,elset=desi_ele1');
for i=1:size(cage_1,2)
    if (rem(i,10)==0)||(i==size(cage_1,2))
        fprintf(fid,'%7.0f',cage_1(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',cage_1(i));
        fprintf(fid,'%s\t',',');
    end
end
%写入元素集cage_2
fprintf(fid,'%s\r\n','*Elset,elset=desi_ele2');
for i=1:size(cage_2,2)
    if (rem(i,10)==0)||(i==size(cage_2,2))
        fprintf(fid,'%7.0f',cage_2(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',cage_2(i));
        fprintf(fid,'%s\t',',');
    end
end
%写入元素集cage_3
fprintf(fid,'%s\r\n','*Elset,elset=desi_ele3');
for i=1:size(cage_3,2)
    if (rem(i,10)==0)||(i==size(cage_3,2))
        fprintf(fid,'%7.0f',cage_3(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',cage_3(i));
        fprintf(fid,'%s\t',',');
    end
end
%写入元素集cage_4
fprintf(fid,'%s\r\n','*Elset,elset=desi_ele4');
for i=1:size(cage_4,2)
    if (rem(i,10)==0)||(i==size(cage_4,2))
        fprintf(fid,'%7.0f',cage_4(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',cage_4(i));
        fprintf(fid,'%s\t',',');
    end
end
%写入元素集cage_5
fprintf(fid,'%s\r\n','*Elset,elset=desi_ele5');
for i=1:size(cage_5,2)
    if (rem(i,10)==0)||(i==size(cage_5,2))
        fprintf(fid,'%7.0f',cage_5(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',cage_5(i));
        fprintf(fid,'%s\t',',');
    end
end
%写入元素集cage_6
fprintf(fid,'%s\r\n','*Elset,elset=desi_ele6');
for i=1:size(cage_6,2)
    if (rem(i,10)==0)||(i==size(cage_6,2))
        fprintf(fid,'%7.0f',cage_6(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',cage_6(i));
        fprintf(fid,'%s\t',',');
    end
end
%写入元素集cage_7
fprintf(fid,'%s\r\n','*Elset,elset=desi_ele7');
for i=1:size(cage_7,2)
    if (rem(i,10)==0)||(i==size(cage_7,2))
        fprintf(fid,'%7.0f',cage_7(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',cage_7(i));
        fprintf(fid,'%s\t',',');
    end
end
%写入元素集cage_8
fprintf(fid,'%s\r\n','*Elset,elset=desi_ele8');
for i=1:size(cage_8,2)
    if (rem(i,10)==0)||(i==size(cage_8,2))
        fprintf(fid,'%7.0f',cage_8(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',cage_8(i));
        fprintf(fid,'%s\t',',');
    end
end
%写入元素集cage_9
fprintf(fid,'%s\r\n','*Elset,elset=desi_ele9');
for i=1:size(cage_9,2)
    if (rem(i,10)==0)||(i==size(cage_9,2))
        fprintf(fid,'%7.0f',cage_9(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',cage_9(i));
        fprintf(fid,'%s\t',',');
    end
end
%写入元素集cage_10
fprintf(fid,'%s\r\n','*Elset,elset=desi_ele10');
for i=1:size(cage_10,2)
    if (rem(i,10)==0)||(i==size(cage_10,2))
        fprintf(fid,'%7.0f',cage_10(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',cage_10(i));
        fprintf(fid,'%s\t',',');
    end
end

fclose(fid);
