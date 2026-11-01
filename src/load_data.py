from pathlib import Path

import numpy as np


def load_binary_mnist(digit_0, digit_1):
    """提取两个数字：digit_0 的标签设为 0，digit_1 的标签设为 1。"""
    if digit_0 == digit_1:
        raise ValueError("请选择两个不同的数字")

    if digit_0 not in range(10) or digit_1 not in range(10):
        raise ValueError("数字必须在 0～9 之间")

    
    data_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "mnist_data"
        / "mnist.npz"
    )

    with np.load(data_path) as data:
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

    # 1. 筛选指定两个数字的样本
    train_mask = (y_train == digit_0) | (y_train == digit_1)
    test_mask = (y_test == digit_0) | (y_test == digit_1)

    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    # 2. 将原数字标签转换为二分类标签
    # digit_0 → 0，digit_1 → 1
    y_train = (y_train == digit_1).astype(np.float32)
    y_test = (y_test == digit_1).astype(np.float32)

    # 3. 每张图片展开成 784 个特征，并将像素值从 0～255 缩放到 0～1
    x_train = x_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_binary_mnist(3, 7)

    print("训练集特征形状：", x_train.shape)
    print("训练集标签形状：", y_train.shape)
    print("测试集特征形状：", x_test.shape)
    print("测试集标签形状：", y_test.shape)
    print("前 10 个训练标签：", y_train[:10])