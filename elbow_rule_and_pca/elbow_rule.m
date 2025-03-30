% 读取数据
data = readtable('Wholesale customers data.csv'); 
data = table2array(data);

% 标准化数据
data_scaled = zscore(data); 

sse = zeros(1, 10); 

% 计算不同 K 值下的 SSE
for k = 1:10
 
    [idx, ~, sumd] = kmeans(data_scaled, k, 'Replicates', 5); 
    sse(k) = sum(sumd); % sumd 是每个簇的聚类误差平方和
end


figure;
plot(1:10, sse, 'b-o');
xlabel('K值');
ylabel('SSE');
title('肘部法则');
grid on;