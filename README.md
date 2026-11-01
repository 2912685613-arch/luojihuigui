# NumPy 逻辑回归：MNIST 数字分类

使用纯 NumPy 实现逻辑回归，对 MNIST 数据集中的数字 3 和 6 进行二分类。

## 项目结构

- `src/luojihuigui.py` - 主程序，实现逻辑回归训练和预测
- `src/load_data.py` - 数据加载模块，读取和预处理 MNIST 数据
- `data/mnist_data/mnist.npz` - MNIST 数据集
- `outputs/` - 训练结果和可视化输出

## 运行

```bash
pip install -r requirements.txt
python src/luojihuigui.py
```

## 主要特性

- 纯 NumPy 实现，无深度学习框架依赖
- 数值稳定的 sigmoid 和损失函数计算
- Mini-batch 梯度下降
- 训练过程可视化
