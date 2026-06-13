# 基于MNIST数据集的卷积网络消融实验报告
## 一、数据集与预处理
### 1. 数据集
采用**MNIST手写数字数据集**，总计70000张28×28灰度图像，包含10个数字类别：
- 训练集：60000张
- 测试集：10000张

### 2. 预处理
1. 图像转换为Tensor格式，像素值归一至值域 $[0,1]$
2. 标准化处理，采用MNIST标准参数：均值=0.1307，标准差=0.3081

## 二、模型设计
### 1. 基础网络结构
所有对照模型（消融变量除外）统一使用如下结构：
`卷积层 → 激活函数 → 2×2最大池化层 → 卷积层 → 池化层 → 128维全连接层 → 10维输出层`

### 2. 通用超参数
- 优化器：Adam，学习率 $\text{lr}=0.001$
- 损失函数：交叉熵损失
- 批次大小：64
- 训练轮数：10

## 三、正则化对比实验设置
| 模型 | Dropout (p=0.5) | BatchNorm |
| ---- | --------------- | --------- |
| 无正则化（基线） | × | × |
| Dropout | √ | × |
| BatchNorm | × | √ |
| Dropout+BN | √ | √ |

## 四、卷积核对比实验设置
| 模型 | 卷积结构 | 感受野 | 参数量 |
| ---- | -------- | ------ | ------ |
| 小核堆叠 | 3×3(×3) → 池化 → 3×3 → 池化 | 7×7 | 54218 |
| 大核单层 | 7×7 → 池化 → 3×3 → 池化 | 7×7 | 34186 |

## 五、实验结果
### 1. 激活函数对比实验
#### （1）激活函数：ReLU
```
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
```
最终测试准确率: 99.31% | 耗时: 336.3秒

#### （2）激活函数：Sigmoid
```
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
```
最终测试准确率: 98.64% | 耗时: 376.7秒

#### （3）激活函数：Tanh
```
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
```
最终测试准确率: 98.92% | 耗时: 443.7秒

#### （4）激活函数消融结果汇总
| 激活函数 | 参数量 | 测试准确率 |
| -------- | ------ | ---------- |
| ReLU     | 421642 | 99.31%     |
| Sigmoid  | 421642 | 98.64%     |
| Tanh     | 421642 | 98.92%     |

---

### 2. 正则化对比实验
#### （1）无正则化（基线 No Reg）
```
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
```
最终测试准确率: 99.24% | 耗时: 408.9秒

#### （2）Dropout（p=0.5）
```
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
```
最终测试准确率: 99.19% | 耗时: 358.5秒

#### （3）BatchNorm
> 说明：BatchNorm 对过拟合的影响；参数量: 421834；运行设备: CPU；训练轮数: 10
```
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
```
最终测试准确率: 99.04% | 耗时: 429.9秒

#### （4）Dropout+BN
```
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
```
最终测试准确率: 99.33% | 耗时: 408.5秒

#### （5）正则化消融结果汇总
| 方法 | 参数量 | 测试准确率 |
| ---- | ------ | ---------- |
| No Reg (基线) | 421642 | 99.24% |
| Dropout (p=0.5) | 421642 | 99.19% |
| BatchNorm | 421834 | 99.04% |
| Dropout+BN | 421834 | 99.33% |

---

### 3. 卷积核对比实验
#### （1）3×3卷积核堆叠（3个）
参数量: 68682
```
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
```
最终测试准确率: 99.23% | 耗时: 639.6秒

#### （2）7×7大卷积核（单层）
参数量: 51466
```
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
```
最终测试准确率: 99.09% | 耗时: 321.6秒

#### （3）卷积核消融结果汇总
| 模型 | 参数量 | 测试准确率 |
| ---- | ------ | ---------- |
| 3×3 堆叠 (3个) | 68682 | 99.23% |
| 7×7 单核 | 51466 | 99.09% |

参数量变化: -17216（-25.1%）
准确率变化: -0.14%

## 六、实验结论
### 1. 激活函数
- **收敛速度**：ReLU首轮测试准确率即达到98.61%，Sigmoid至第4轮才接近该水平。ReLU无梯度饱和、梯度传播更有效，收敛速度最快；Tanh收敛速度居中；Sigmoid梯度饱和问题严重，收敛最慢。
- **最终准确率**：ReLU准确率较Tanh高出0.39个百分点，较Sigmoid高出0.67个百分点。
- **训练耗时**：ReLU训练耗时最短（336秒），Sigmoid、Tanh耗时更长，梯度饱和会降低反向传播效率。

### 2. 正则化策略
- 无正则化（基线）：训练损失降至0.0067，测试准确率99.24%，模型存在轻微过拟合，训练集拟合效果优于测试集。
- Dropout（p=0.5）：训练损失无法持续降低至低位（最终0.024），测试准确率仅较基线低0.05%，有效缩小训练集与测试集差距，泛化能力得到提升。
- 单独使用BatchNorm：效果弱于基线（99.04%），推测原因是网络层数较浅，且批次大小为64时，批量归一化的统计量估计稳定性不足。
- Dropout+BN组合：取得全场最高测试准确率99.33%。BatchNorm可加速训练并提供隐式正则化，Dropout提供显式正则化，二者形成互补；该组合虽训练损失偏高，但模型泛化能力最优。

### 3. 卷积核设计
- **准确率**：3个3×3小核堆叠模型比单个7×7大核模型准确率高0.14个百分点。由于堆叠结构拥有更多非线性变换层，能够提取更加丰富的图像特征。
- **参数量**：小核堆叠结构参数量更多，多出17216个参数，来源于额外卷积层与激活层，小幅准确率提升需结合业务场景权衡参数量成本。
- **训练耗时**：3×3堆叠模型训练耗时639秒，远高于7×7单核模型的321.6秒，多层卷积会增加前向传播与反向传播的计算量。

本次实验验证了深度学习经典结论：ReLU综合表现优于Sigmoid、Tanh；**Dropout与BatchNorm组合使用可进一步提升模型泛化能力**；**多层小卷积核堆叠的特征提取效果优于单层大卷积核**。