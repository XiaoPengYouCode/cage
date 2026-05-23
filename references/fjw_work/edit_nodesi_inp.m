% nod_coo_mm=(nod_coo-1)*1;%%%matlab坐标（1，1，1）对标到abaqus坐标（0，0，0）
% nod_coo_mm=(nod_coo-1)*0.4;%%%matlab坐标（1，1，1）对标到abaqus坐标（0，0，0）还需乘分辨率
nod_coo_mm=(nod_coo-1)*0.6;%%%matlab坐标（1，1，1）对标到abaqus坐标（0，0，0）还需乘分辨率
all_ele=1:size(ele_nod,1);
nodesi_ele_0_count=size(cor_ele,2);
% nodesi_ele_0=nodesi_ele_cor;
nodesi_ele_0=cor_ele;
% nodesi_ele_1_count=size(truss_ele,2);
% nodesi_ele_1=truss_ele;
nodesi_ele_2_count=size(tra_ele,2);
% nodesi_ele_2=nodesi_ele_tra;
nodesi_ele_2=tra_ele;
% nodesi_ele_3_count=size(wing_ele,2);
% nodesi_ele_3=wing_ele;

if exist('ini_nodesi.inp','file')
delete('ini_nodesi.inp');
else
end
copyfile('ini.inp','ini_nodesi.inp');
fid=fopen('ini_nodesi.inp','a');%鍦ㄦ枃浠舵湯灏惧紑濮嬪啓
fprintf(fid,'\r\n');

%鍐欏叆鑺傜偣
fprintf(fid,'%s\r\n','*Node');
for i=1:size(nod_coo,1)
    fprintf(fid,'%s\r\n',[num2str(i) ',' num2str(nod_coo_mm(i,1)) ',' num2str(nod_coo_mm(i,2)) ',' num2str(nod_coo_mm(i,3))]);
end

%鍐欏叆鍏冪礌
fprintf(fid,'%s\r\n','*Element, type=C3D8R');
for i=1:size(ele_nod,1)
    fprintf(fid,'%s\r\n',[num2str(i) ',' num2str(ele_nod(i,1)) ',' num2str(ele_nod(i,2)) ',' num2str(ele_nod(i,3))...
         ',' num2str(ele_nod(i,4)) ',' num2str(ele_nod(i,5)) ',' num2str(ele_nod(i,6)) ',' num2str(ele_nod(i,7)) ',' num2str(ele_nod(i,8))]);
end

%鍐欏叆鑺傜偣闆哸ll
fprintf(fid,'%s\r\n','*Nset, nset=alln, generate');
fprintf(fid,'%s\r\n',['1,' num2str(size(nod_coo,1)) ',1']);

%鍐欏叆鍏冪礌闆哸llele
fprintf(fid,'%s\r\n','*Elset, elset=alle, generate');
fprintf(fid,'%s\r\n',['1,' num2str(size(ele_nod,1)) ',1']);

%鍐欏叆鍏冪礌闆唍odesi_ele_0
fprintf(fid,'%s\r\n','*Elset, elset=NODESI_ELE_COR');
for i=1:size(nodesi_ele_0,2)
    if (rem(i,10)==0)||(i==size(nodesi_ele_0,2))
        fprintf(fid,'%7.0f',nodesi_ele_0(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',nodesi_ele_0(i));
        fprintf(fid,'%s\t',',');
    end
end

%鍐欏叆鍏冪礌闆唍odesi_ele_2
fprintf(fid,'%s\r\n','*Elset, elset=NODESI_ELE_TRA');
for i=1:size(nodesi_ele_2,2)
    if (rem(i,10)==0)||(i==size(nodesi_ele_2,2))
        fprintf(fid,'%7.0f',nodesi_ele_2(i));
        fprintf(fid,'%s\r\n',',');
    else
        fprintf(fid,'%7.0f',nodesi_ele_2(i));
        fprintf(fid,'%s\t',',');
    end
end

%鍐欏叆鍏冪礌闆唍odesi_ele_1
% fprintf(fid,'%s\r\n','*Elset, elset=NODESI_ELE_TRUSS');
% for i=1:size(nodesi_ele_1,2)
%     if (rem(i,10)==0)||(i==size(nodesi_ele_1,2))
%         fprintf(fid,'%7.0f',nodesi_ele_1(i));
%         fprintf(fid,'%s\r\n',',');
%     else
%         fprintf(fid,'%7.0f',nodesi_ele_1(i));
%         fprintf(fid,'%s\t',',');
%     end
% end

% fprintf(fid,'%s\r\n','*Elset, elset=NODESI_ELE_WING');
% for i=1:size(nodesi_ele_3,2)
%     if (rem(i,10)==0)||(i==size(nodesi_ele_3,2))
%         fprintf(fid,'%7.0f',nodesi_ele_3(i));
%         fprintf(fid,'%s\r\n',',');
%     else
%         fprintf(fid,'%7.0f',nodesi_ele_3(i));
%         fprintf(fid,'%s\t',',');
%     end
% end


fclose(fid);