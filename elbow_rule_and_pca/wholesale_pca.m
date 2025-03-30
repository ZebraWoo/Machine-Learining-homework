% 读取数据
data_table = readtable('Wholesale customers data.csv'); 
X = table2array(data_table);

% 数据中心化
meanX = mean(X, 1);  
X_centered = X - meanX;  

% 主成分分析 (PCA)
[coeff, score, latent] = pca(X_centered);
X_pca = score(:, 1:2);  % 选择前两个主成分

% 由肘部法则确定聚类数为 2
k = 2;  

% 使用 KMeans 对投影数据进行聚类
[idx, C] = kmeans(X_pca, k);  

% 绘制二维散点图
figure;
gscatter(X_pca(:, 1), X_pca(:, 2), idx, 'rbg', 'osd');  % 使用不同的颜色和标记来表示不同的簇
hold on;

% 绘制聚类的质心
plot(C(:, 1), C(:, 2), 'kx', 'MarkerSize', 10, 'LineWidth', 2);  % 质心以黑色大叉表示

title('K-Means 聚类结果（PCA 降维）');
xlabel('主成分 1');
ylabel('主成分 2');
legend('Cluster 1', 'Cluster 2', 'Centroids', 'Location', 'Best');
hold off;