"""
MNIST 手写体识别 CNN 消融实验（三个完整实验）
1. 激活函数对比：ReLU vs Sigmoid / Tanh
2. 正则化对比：无 Dropout/BN vs Dropout vs BatchNorm
3. 卷积核对比：3个3x3堆叠 vs 单个7x7

运行方式：
    python mnist_cnn_ablation.py           # 运行所有实验，结果保存到 results_all.txt
或单独运行某个实验（修改 main 函数中的实验编号）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import sys

# ----------------------------- 超参数配置 ---------------------------------
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------- 数据加载 -----------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------- 辅助函数 -----------------------------------
def count_parameters(model):
    """统计模型参数量（可训练参数）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

def run_experiment(model, model_name, description=""):
    """训练并评估模型，返回（参数量，最终测试准确率）"""
    model = model.to(DEVICE)
    param_count = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n{'='*60}")
    print(f"实验: {model_name}")
    if description:
        print(f"说明: {description}")
    print(f"参数量: {param_count:,}")
    print(f"设备: {DEVICE}")
    print(f"训练轮数: {EPOCHS}")
    
    start_time = time.time()
    for epoch in range(1, EPOCHS+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        test_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    final_acc = evaluate(model, test_loader)
    elapsed = time.time() - start_time
    print(f"最终测试准确率: {final_acc:.2f}% | 耗时: {elapsed:.1f}秒")
    return param_count, final_acc

# ============================= 模型定义 ====================================
# ---------- 1. 激活函数对比实验的模型 ----------
class BaseCNN_Activation(nn.Module):
    """基础CNN结构，激活函数可配置"""
    def __init__(self, activation=nn.ReLU()):
        super(BaseCNN_Activation, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.activation = activation
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------- 2. Dropout / BN 对比实验的模型 ----------
class CNN_Regularization(nn.Module):
    """正则化对比模型，可选择添加 Dropout 和 BatchNorm"""
    def __init__(self, use_dropout=False, use_batchnorm=False, dropout_rate=0.5):
        super(CNN_Regularization, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---------- 3. 卷积核对比实验的模型（直接复制之前的两个模型） ----------
class SmallKernelStack(nn.Module):
    """小卷积核堆叠：3个3x3卷积，感受野7x7"""
    def __init__(self):
        super(SmallKernelStack, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 7 * 7, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool1(x)
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LargeKernelSingle(nn.Module):
    """大卷积核：单个7x7卷积"""
    def __init__(self):
        super(LargeKernelSingle, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 7 * 7, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ============================= 三个消融实验主函数 ============================
def experiment_activation():
    """实验1：激活函数对比 (ReLU vs Sigmoid vs Tanh)"""
    print("\n\n" + "#"*60)
    print("# 消融实验 1：激活函数对比 (ReLU vs Sigmoid vs Tanh)")
    print("#"*60)
    
    results = []
    activations = {
        "ReLU": nn.ReLU(),
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh()
    }
    
    for name, act in activations.items():
        model = BaseCNN_Activation(activation=act)
        params, acc = run_experiment(model, f"激活函数: {name}",
                                     description=f"使用 {name} 作为非线性激活函数")
        results.append((name, params, acc))
    
    # 打印对比结果
    print("\n" + "="*60)
    print("激活函数消融实验结果汇总")
    print("="*60)
    print(f"{'激活函数':<10} {'参数量':<12} {'测试准确率':<10}")
    for name, params, acc in results:
        print(f"{name:<10} {params:<12,} {acc:<10.2f}%")
    
    # 保存到文件
    with open("results_activation.txt", "w") as f:
        f.write("激活函数消融实验 (MNIST CNN)\n")
        f.write(f"{'激活函数':<10} {'参数量':<12} {'测试准确率':<10}\n")
        for name, params, acc in results:
            f.write(f"{name:<10} {params:<12,} {acc:<10.2f}%\n")
    print("\n结果已保存到 results_activation.txt")

def experiment_regularization():
    """实验2：正则化对比 (无Dropout/BN vs Dropout vs BN)"""
    print("\n\n" + "#"*60)
    print("# 消融实验 2：正则化对比 (无正则化 vs Dropout vs BatchNorm)")
    print("#"*60)
    
    models = {
        "No Reg (基线)": CNN_Regularization(use_dropout=False, use_batchnorm=False),
        "Dropout (p=0.5)": CNN_Regularization(use_dropout=True, use_batchnorm=False),
        "BatchNorm": CNN_Regularization(use_dropout=False, use_batchnorm=True),
        "Dropout+BN": CNN_Regularization(use_dropout=True, use_batchnorm=True)
    }
    
    results = []
    for name, model in models.items():
        params, acc = run_experiment(model, f"正则化: {name}",
                                     description=f"{name} 对过拟合的影响")
        results.append((name, params, acc))
    
    print("\n" + "="*60)
    print("正则化消融实验结果汇总")
    print("="*60)
    print(f"{'方法':<20} {'参数量':<12} {'测试准确率':<10}")
    for name, params, acc in results:
        print(f"{name:<20} {params:<12,} {acc:<10.2f}%")
    
    with open("results_regularization.txt", "w") as f:
        f.write("正则化消融实验 (MNIST CNN)\n")
        f.write(f"{'方法':<20} {'参数量':<12} {'测试准确率':<10}\n")
        for name, params, acc in results:
            f.write(f"{name:<20} {params:<12,} {acc:<10.2f}%\n")
    print("\n结果已保存到 results_regularization.txt")

def experiment_kernel():
    """实验3：卷积核对比 (3x3堆叠 vs 7x7单核)"""
    print("\n\n" + "#"*60)
    print("# 消融实验 3：卷积核对比 (3x3堆叠 vs 7x7单核)")
    print("#"*60)
    
    models = {
        "3x3 堆叠 (3个)": SmallKernelStack(),
        "7x7 单核": LargeKernelSingle()
    }
    
    results = []
    for name, model in models.items():
        params, acc = run_experiment(model, f"卷积核: {name}",
                                     description=f"{name} 感受野均为7x7，对比参数量和准确率")
        results.append((name, params, acc))
    
    print("\n" + "="*60)
    print("卷积核消融实验结果汇总")
    print("="*60)
    print(f"{'模型':<20} {'参数量':<12} {'测试准确率':<10}")
    for name, params, acc in results:
        print(f"{name:<20} {params:<12,} {acc:<10.2f}%")
    
    # 计算变化
    baseline_name, baseline_params, baseline_acc = results[0]
    ablation_name, ablation_params, ablation_acc = results[1]
    param_change = ablation_params - baseline_params
    acc_change = ablation_acc - baseline_acc
    print(f"\n对比: {ablation_name} 相比 {baseline_name}")
    print(f"参数量变化: {param_change:+,d} ({param_change/baseline_params*100:+.1f}%)")
    print(f"准确率变化: {acc_change:+.2f}%")
    
    with open("results_kernel.txt", "w") as f:
        f.write("卷积核消融实验 (MNIST CNN)\n")
        f.write(f"{'模型':<20} {'参数量':<12} {'测试准确率':<10}\n")
        for name, params, acc in results:
            f.write(f"{name:<20} {params:<12,} {acc:<10.2f}%\n")
        f.write(f"\n结论：{ablation_name} 参数量减少 {abs(param_change):,} ({-param_change/baseline_params*100:.1f}%)，")
        f.write(f"准确率 {acc_change:+.2f}%")
    print("\n结果已保存到 results_kernel.txt")

# ============================= 主程序 ======================================
def main():
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"运行设备: {DEVICE}")
    print("\n将执行三个完整的消融实验，每个实验会单独输出结果文件。")
    print("注意：每个实验需要训练多个模型，总耗时约 10-20 分钟（取决于硬件）。")
    
    # 询问是否全部运行
    choice = input("\n请选择执行方式：\n1 - 运行全部三个实验\n2 - 只运行实验1（激活函数）\n3 - 只运行实验2（正则化）\n4 - 只运行实验3（卷积核）\n请输入数字 [1-4]: ").strip()
    
    if choice == '2':
        experiment_activation()
    elif choice == '3':
        experiment_regularization()
    elif choice == '4':
        experiment_kernel()
    else:
        print("\n开始运行全部三个实验...")
        experiment_activation()
        experiment_regularization()
        experiment_kernel()
        print("\n所有实验完成！结果已保存至：")
        print(" - results_activation.txt")
        print(" - results_regularization.txt")
        print(" - results_kernel.txt")
        print("\n同时，每个实验的控制台输出中也包含了详细的训练过程和最终对比。")

if __name__ == "__main__":
    main()