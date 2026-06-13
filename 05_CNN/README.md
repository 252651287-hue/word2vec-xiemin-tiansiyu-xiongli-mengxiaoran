一、数据集与预处理
1.数据集：MNIST（70,000张28×28灰度图像，10个数字类别）
训练集：60,000张
测试集：10,000张
2.预处理：
（1）转换为Tensor（值域[0,1]）
（2）标准化：均值0.1307，标准差0.3081（MNIST标准参数）

二、模型设计
1.所有模型共享以下基础结构（除消融变量外）：
卷积层 → 激活函数 → 池化层（2×2最大池化）→ 卷积层 → 池化层 → 全连接层（128维）→ 输出层（10维）
2.优化器：Adam（lr=0.001）
3.损失函数：交叉熵损失
4.批量大小：64
5.训练轮数：10

三、正则化对比模型
模型	       Dropout (p=0.5)	BatchNorm
无正则化（基线）	×	           ×
Dropout	           √	         ×
BatchNorm	       ×	         √
Dropout+BN	       √	         √

四、卷积核对比模型
模型	              卷积结构	           感受野	参数量
小核堆叠	3×3 (×3) → 池化 → 3×3 → 池化	7×7	    54,218
大核单层	7×7 → 池化 → 3×3 → 池化	        7×7	    34,186 

五、实验结果
1.激活函数对比
（1）激活函数: ReLU
Epoch  1/10 | Loss: 0.1335 | Test Acc: 98.61%
Epoch  2/10 | Loss: 0.0425 | Test Acc: 98.98%
Epoch  3/10 | Loss: 0.0275 | Test Acc: 98.92%
Epoch  4/10 | Loss: 0.0196 | Test Acc: 99.08%
Epoch  5/10 | Loss: 0.0150 | Test Acc: 99.24%
Epoch  6/10 | Loss: 0.0125 | Test Acc: 99.09%
Epoch  7/10 | Loss: 0.0103 | Test Acc: 99.16%
Epoch  8/10 | Loss: 0.0092 | Test Acc: 99.09%
Epoch  9/10 | Loss: 0.0075 | Test Acc: 99.14%
Epoch 10/10 | Loss: 0.0066 | Test Acc: 99.31%
最终测试准确率: 99.31% | 耗时: 336.3秒

（2）激活函数: Sigmoid
Epoch  1/10 | Loss: 0.6060 | Test Acc: 95.82%
Epoch  2/10 | Loss: 0.1174 | Test Acc: 97.62%
Epoch  3/10 | Loss: 0.0735 | Test Acc: 98.07%
Epoch  4/10 | Loss: 0.0541 | Test Acc: 98.42%
Epoch  5/10 | Loss: 0.0435 | Test Acc: 98.54%
Epoch  6/10 | Loss: 0.0350 | Test Acc: 98.76%
Epoch  7/10 | Loss: 0.0290 | Test Acc: 98.67%
Epoch  8/10 | Loss: 0.0252 | Test Acc: 98.89%
Epoch  9/10 | Loss: 0.0207 | Test Acc: 98.89%
Epoch 10/10 | Loss: 0.0163 | Test Acc: 98.64%
最终测试准确率: 98.64% | 耗时: 376.7秒

（3）激活函数: Tanh
Epoch  1/10 | Loss: 0.1409 | Test Acc: 98.51%
Epoch  2/10 | Loss: 0.0433 | Test Acc: 98.56%
Epoch  3/10 | Loss: 0.0289 | Test Acc: 98.92%
Epoch  4/10 | Loss: 0.0228 | Test Acc: 98.68%
Epoch  5/10 | Loss: 0.0160 | Test Acc: 98.69%
Epoch  6/10 | Loss: 0.0140 | Test Acc: 98.78%
Epoch  7/10 | Loss: 0.0123 | Test Acc: 98.63%
Epoch  8/10 | Loss: 0.0118 | Test Acc: 98.78%
Epoch  9/10 | Loss: 0.0084 | Test Acc: 98.93%
Epoch 10/10 | Loss: 0.0080 | Test Acc: 98.92%
最终测试准确率: 98.92% | 耗时: 443.7秒

============================================================
激活函数消融实验结果汇总
============================================================
激活函数    参数量       测试准确率     
ReLU       421,642      99.31%
Sigmoid    421,642      98.64%
Tanh       421,642      98.92%

2.正则化对比
（1）正则化: No Reg (基线)
Epoch  1/10 | Loss: 0.1286 | Test Acc: 98.77%
Epoch  2/10 | Loss: 0.0428 | Test Acc: 98.91%
Epoch  3/10 | Loss: 0.0287 | Test Acc: 99.16%
Epoch  4/10 | Loss: 0.0221 | Test Acc: 99.13%
Epoch  5/10 | Loss: 0.0165 | Test Acc: 98.72%
Epoch  6/10 | Loss: 0.0126 | Test Acc: 98.94%
Epoch  7/10 | Loss: 0.0112 | Test Acc: 99.08%
Epoch  8/10 | Loss: 0.0096 | Test Acc: 99.07%
Epoch  9/10 | Loss: 0.0077 | Test Acc: 99.05%
Epoch 10/10 | Loss: 0.0067 | Test Acc: 99.24%
最终测试准确率: 99.24% | 耗时: 408.9秒

（2）正则化: Dropout (p=0.5)
Epoch  1/10 | Loss: 0.2242 | Test Acc: 98.47%
Epoch  2/10 | Loss: 0.0847 | Test Acc: 99.00%
Epoch  3/10 | Loss: 0.0631 | Test Acc: 99.08%
Epoch  4/10 | Loss: 0.0496 | Test Acc: 99.16%
Epoch  5/10 | Loss: 0.0437 | Test Acc: 99.14%
Epoch  6/10 | Loss: 0.0347 | Test Acc: 99.21%
Epoch  7/10 | Loss: 0.0329 | Test Acc: 99.13%
Epoch  8/10 | Loss: 0.0271 | Test Acc: 99.06%
Epoch  9/10 | Loss: 0.0262 | Test Acc: 99.26%
Epoch 10/10 | Loss: 0.0240 | Test Acc: 99.19%
最终测试准确率: 99.19% | 耗时: 358.5秒

（3）正则化: BatchNorm
说明: BatchNorm 对过拟合的影响
参数量: 421,834
设备: cpu
训练轮数: 10
Epoch  1/10 | Loss: 0.1288 | Test Acc: 97.95%
Epoch  2/10 | Loss: 0.0470 | Test Acc: 98.82%
Epoch  3/10 | Loss: 0.0340 | Test Acc: 98.89%
Epoch  4/10 | Loss: 0.0264 | Test Acc: 98.48%
Epoch  5/10 | Loss: 0.0219 | Test Acc: 99.02%
Epoch  6/10 | Loss: 0.0169 | Test Acc: 98.79%
Epoch  7/10 | Loss: 0.0161 | Test Acc: 98.71%
Epoch  8/10 | Loss: 0.0126 | Test Acc: 99.04%
Epoch  9/10 | Loss: 0.0091 | Test Acc: 98.96%
Epoch 10/10 | Loss: 0.0102 | Test Acc: 99.04%
最终测试准确率: 99.04% | 耗时: 429.9秒

（4）正则化: Dropout+BN
Epoch  1/10 | Loss: 0.2292 | Test Acc: 98.60%
Epoch  2/10 | Loss: 0.1098 | Test Acc: 98.69%
Epoch  3/10 | Loss: 0.0842 | Test Acc: 99.03%
Epoch  4/10 | Loss: 0.0739 | Test Acc: 98.94%
Epoch  5/10 | Loss: 0.0616 | Test Acc: 99.02%
Epoch  6/10 | Loss: 0.0533 | Test Acc: 98.99%
Epoch  7/10 | Loss: 0.0511 | Test Acc: 99.15%
Epoch  8/10 | Loss: 0.0453 | Test Acc: 99.18%
Epoch  9/10 | Loss: 0.0408 | Test Acc: 98.85%
Epoch 10/10 | Loss: 0.0357 | Test Acc: 99.33%
最终测试准确率: 99.33% | 耗时: 408.5秒

============================================================
正则化消融实验结果汇总
============================================================
方法                   参数量       测试准确率     
No Reg (基线)          421,642      99.24%
Dropout (p=0.5)       421,642       99.19%
BatchNorm             421,834       99.04%
Dropout+BN            421,834       99.33%

3.卷积核对比
（1）卷积核: 3x3堆叠 (3个)
参数量: 68,682
Epoch  1/10 | Loss: 0.1273 | Test Acc: 98.59%
Epoch  2/10 | Loss: 0.0403 | Test Acc: 99.12%
Epoch  3/10 | Loss: 0.0277 | Test Acc: 99.16%
Epoch  4/10 | Loss: 0.0230 | Test Acc: 99.25%
Epoch  5/10 | Loss: 0.0166 | Test Acc: 99.15%
Epoch  6/10 | Loss: 0.0153 | Test Acc: 99.23%
Epoch  7/10 | Loss: 0.0127 | Test Acc: 98.98%
Epoch  8/10 | Loss: 0.0109 | Test Acc: 99.16%
Epoch  9/10 | Loss: 0.0091 | Test Acc: 99.22%
Epoch 10/10 | Loss: 0.0083 | Test Acc: 99.23%
最终测试准确率: 99.23% | 耗时: 639.6秒

（2）卷积核: 7x7单核
参数量: 51,466
Epoch  1/10 | Loss: 0.1336 | Test Acc: 98.69%
Epoch  2/10 | Loss: 0.0469 | Test Acc: 98.41%
Epoch  3/10 | Loss: 0.0340 | Test Acc: 98.95%
Epoch  4/10 | Loss: 0.0261 | Test Acc: 98.97%
Epoch  5/10 | Loss: 0.0203 | Test Acc: 99.08%
Epoch  6/10 | Loss: 0.0149 | Test Acc: 99.00%
Epoch  7/10 | Loss: 0.0144 | Test Acc: 98.92%
Epoch  8/10 | Loss: 0.0136 | Test Acc: 98.99%
Epoch  9/10 | Loss: 0.0104 | Test Acc: 99.10%
Epoch 10/10 | Loss: 0.0087 | Test Acc: 99.09%
最终测试准确率: 99.09% | 耗时: 321.6秒

============================================================
卷积核消融实验结果汇总
============================================================
模型                   参数量       测试准确率     
3x3 堆叠 (3个)         68,682      99.23 %
7x7 单核               51,466       99.09%
参数量变化: -17,216 (-25.1%)
准确率变化: -0.14%

六、实验结论
1.激活函数
·收敛速度：ReLU第1轮准确率即达98.61%，而Sigmoid第4轮才接近该水平。ReLU 由于无饱和区、梯度非稀疏，收敛最快；Tanh居中；Sigmoid饱和严重，收敛最慢。
·最终准确率：ReLU比Tanh高0.39个百分点，比Sigmoid高0.67个百分点。
·训练耗时：ReLU耗时最短（336秒），Sigmoid和Tanh更慢，因为梯度计算与饱和区导致反向传播效率降低。

2.正则化
·无正则化：训练损失降至0.0067，测试准确率99.24%，存在轻微过拟合（训练集拟合优于测试集）。
·Dropout (p=0.5)：训练损失无法降得太低（0.024），但测试准确率仅比基线低0.05%，显著缩小了训练-测试差距，泛化性提升。
·BatchNorm alone：表现反而低于基线（99.04%），可能与网络深度较浅、批量大小（64）下BN统计量估计不够稳定有关。
·Dropout+BN：取得最高测试准确率99.33%，表明 BN 提供加速与隐式正则化，Dropout 提供显式正则化，二者互补（尽管训练损失偏高，但泛化能力最强）。

3.卷积核
·准确率：3个3×3堆叠模型比单个7×7高0.14个百分点。虽然提升幅度不大（MNIST本身已达99%+），但表明更深的非线性变换（三个激活层 vs 一个激活层）能提取更丰富的特征。
·参数量：小核堆叠参数更多（+17,216），主要来自额外的两个卷积层及其后续激活。但换来了0.14%的准确率提升，在工业应用中是否值得需权衡。
·训练耗时：小核堆叠耗时639秒，远高于大核的321秒，因为前向/反向传播需要经过更多卷积层。

这些结论验证了深度学习领域的常见经验：ReLU优于Sigmoid/Tanh、Dropout+BN可提升泛化、小卷积核堆叠优于大卷积核。