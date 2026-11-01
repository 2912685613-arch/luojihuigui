"""用 NumPy 从零实现 MNIST 数字 3/6 的二分类逻辑回归。"""

from pathlib import Path

import numpy as np

from load_data import load_binary_mnist

# 学习率和固定训练轮数仅由训练设置决定，不根据测试集调整。
LEARNING_RATE = 0.1
BATCH_SIZE = 128
EPOCHS = 20
RANDOM_SEED = 42
DIGIT_NEGATIVE = 3  # 标签 -1 对应的原始数字
DIGIT_POSITIVE = 6  # 标签 +1 对应的原始数字

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"



def sigmoid(z):
    """同一个 sigmoid 公式按正负分开计算，避免 exp 的指数溢出。"""
    z = np.asarray(z, dtype=np.float64)
    probability = np.empty_like(z)
    positive = z >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    probability[~positive] = exp_z / (1.0 + exp_z)
    return probability


def train(X, y):
    """PPT：每个 epoch 打乱样本，然后逐个小批量计算梯度、更新参数。"""
    rng = np.random.default_rng(RANDOM_SEED)
    n, feature_count = X.shape                           # X: (n, 785)，y: (n,)
    omega = rng.normal(0.0, 0.01, size=feature_count)      # omega: (785,)
    initial_loss = np.mean(np.logaddexp(0.0, -y * (X @ omega)))
    history = []
    batches_per_epoch = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n训练设置：learning_rate={LEARNING_RATE}，batch_size={BATCH_SIZE}，"
          f"epochs={EPOCHS}，seed={RANDOM_SEED}")
    print(f"omega.shape={omega.shape}；每轮 {batches_per_epoch} 次参数更新；"
          f"最后一批 {n - (batches_per_epoch - 1) * BATCH_SIZE} 个样本")
    print(f"初始化时整个训练集的平均损失：{initial_loss:.8f}")

    for epoch in range(1, EPOCHS + 1):
        # 对 X 和 y 使用同一排列，保持图片和标签一一对应。
        order = rng.permutation(n)
        X_shuffled = X[order]
        y_shuffled = y[order]
        for start in range(0, n, BATCH_SIZE):
            X_batch = X_shuffled[start:start + BATCH_SIZE]  # (m, 785)
            y_batch = y_shuffled[start:start + BATCH_SIZE]  # (m,)
            m = len(y_batch)  # 最后一批也参与训练，并除以实际样本数 m。

            z = X_batch @ omega                         # (m,)，z_i = omega^T x_i
            # 正确梯度：-(1/m) sum[y_i x_i sigmoid(-y_i z_i)]。
            
            coefficient = -y_batch * sigmoid(-y_batch * z)  # (m,)
            gradient = X_batch.T @ coefficient / m       # (785,)
            omega = omega - LEARNING_RATE * gradient     # (785,)，沿负梯度更新

        # 使用本轮结束时的同一组参数，对整个训练集重新计算指标。
        # logaddexp(0, -yz) = log(exp(0) + exp(-yz)) = log(1 + exp(-yz))。
        z = X @ omega                                   # (n,)
        loss = np.mean(np.logaddexp(0.0, -y * z))
        prediction = np.where(sigmoid(z) >= 0.5, +1, -1)
        accuracy = np.mean(prediction == y)
        history.append((epoch, loss, accuracy))
        print(f"Epoch {epoch:02d}/{EPOCHS} | 训练平均损失={loss:.8f} | "
              f"训练准确率={accuracy:.4%}")

    print(f"训练总体损失：{initial_loss:.8f} -> {history[-1][1]:.8f}")
    return omega, np.asarray(history)


def save_figures(history, X_test, y_test, indices, predicted_digits):
    """Matplotlib 只负责画图；保存 PNG，在没有图形界面的环境也能运行。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history[:, 0], history[:, 1], marker="o", markersize=3)
    ax.set(xlabel="Epoch", ylabel="Mean training loss", title=f"MNIST {DIGIT_NEGATIVE}/{DIGIT_POSITIVE} logistic regression")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(indices), figsize=(12, 3))
    for ax, index, predicted_digit in zip(axes, indices, predicted_digits):
        # 最后一个特征是偏置用的 1，画图时只取前 784 个像素。
        ax.imshow(X_test[index, :784].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        true_digit = DIGIT_POSITIVE if y_test[index] == +1 else DIGIT_NEGATIVE
        ax.set_title(f"True: {true_digit} | Pred: {predicted_digit}\nTest index: {index}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_predictions.png", dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 复用已有函数：筛选指定数字、展开成 784 维、除以 255。
    X_train, y_train, X_test, y_test = load_binary_mnist(DIGIT_NEGATIVE, DIGIT_POSITIVE)

    # 加载函数返回 0/1 标签；转换为损失和梯度使用的 -1/+1 标签。
    y_train = np.where(y_train == 0, -1, +1)             # (n_train,)
    y_test = np.where(y_test == 0, -1, +1)               # (n_test,)
    # x = [x1, ..., x784, 1]，最后一个参数就是偏置 b。
    X_train = np.column_stack((X_train, np.ones(len(X_train))))  # (n_train, 785)
    X_test = np.column_stack((X_test, np.ones(len(X_test))))     # (n_test, 785)

    for name, X, y in (("训练集", X_train, y_train), ("测试集", X_test, y_test)):
        print(f"{name}：{len(y)} 个样本；数字 {DIGIT_NEGATIVE} / y=-1：{np.sum(y == -1)}；"
              f"数字 {DIGIT_POSITIVE} / y=+1：{np.sum(y == 1)}")
        print(f"  X.shape={X.shape}，y.shape={y.shape}")
    
    omega, history = train(X_train, y_train)
    np.savetxt(OUTPUT_DIR / "training_history.csv", history, delimiter=",",
               header="epoch,train_loss,train_accuracy", comments="", fmt=["%d", "%.12f", "%.12f"])

    train_prediction = np.where(sigmoid(X_train @ omega) >= 0.5, +1, -1)
    train_correct = np.sum(train_prediction == y_train)
    print(f"\n最终训练集准确率：{train_correct / len(y_train):.4%}（{train_correct}/{len(y_train)}）")

    # 训练结束后才使用测试集；不对 omega 更新，也不根据测试结果调整超参数。
    p_positive = sigmoid(X_test @ omega)                 # (n_test,)
    prediction = np.where(p_positive >= 0.5, +1, -1)      # (n_test,)
    correct = np.sum(prediction == y_test)
    print(f"\n最终测试集准确率：{correct / len(y_test):.4%}（{correct}/{len(y_test)}）")
    
    # 从已完成的预测中取前 5 个结果，用于展示测试图片。
    indices = np.arange(5)
    predicted_digits = np.where(prediction[indices] == +1, DIGIT_POSITIVE, DIGIT_NEGATIVE)
    save_figures(history, X_test, y_test, indices, predicted_digits)
    print(f"\n训练曲线、测试图片和 CSV 已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
