start_num = 589145;
end_num = 593080;
step = 16;

fileID = fopen('numbers.txt', 'w'); % 打开一个文件用于写入

for i = start_num:step:end_num
    numbers = i:i+step-1;
    for j = 1:length(numbers)-1
        fprintf(fileID, '%d, ', numbers(j)); % 输出数字并在后面加上逗号
    end
    fprintf(fileID, '%d\n', numbers(end)); % 单独输出最后一个数字并换行
end

fclose(fileID); % 关闭文件
