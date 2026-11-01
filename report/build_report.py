# -*- coding: utf-8 -*-
"""生成实验报告的 Markdown 与 DOCX 两个版本。

报告内容只在 build_blocks() 中定义一次，代码块直接按行号从 src/ 下的源文件截取，
避免手工抄写代码产生偏差。
"""

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

import mathomml

ROOT = Path(r"D:\machine_learning\luojihuigui")
SRC = ROOT / "src"
OUT = ROOT / "report"
OUT.mkdir(parents=True, exist_ok=True)

LD = (SRC / "load_data.py").read_text(encoding="utf-8").splitlines()
LG = (SRC / "luojihuigui.py").read_text(encoding="utf-8").splitlines()


def seg(lines, a, b):
    """取第 a 行到第 b 行（含），去掉首尾空行和行尾空白。"""
    chunk = [ln.rstrip() for ln in lines[a - 1:b]]
    while chunk and not chunk[0].strip():
        chunk.pop(0)
    while chunk and not chunk[-1].strip():
        chunk.pop()
    return "\n".join(chunk)


BLOCKS = []
H1 = lambda t: BLOCKS.append(("h1", t))
H2 = lambda t: BLOCKS.append(("h2", t))
P = lambda t: BLOCKS.append(("p", t))
NUM = lambda *items: BLOCKS.append(("num", list(items)))
CODE = lambda t: BLOCKS.append(("code", t))
FORM = lambda num, lines: BLOCKS.append(("formula", num, list(lines)))
IMG = lambda path, cap, width: BLOCKS.append(("img", path, cap, width))

TABLES = []


def TBL(n, title, header, rows, aligns=None, widths=None):
    TABLES.append(n)
    BLOCKS.append(("table", n, title, header, rows, aligns, widths))


# --------------------------------------------------------------------------
# 一、实验内容
# --------------------------------------------------------------------------
H1("1. 实验内容")

H2("1.1 实验任务与数据集")
P(
    "本实验要求在不借助现成分类器、自动求导框架和现成优化器的条件下，用 Python 与 NumPy "
    "从零实现逻辑回归，完成 MNIST 手写数字中“3”与“6”的二分类。具体目标有三项：一是掌握二分类"
    "逻辑回归的建模方式，即先用线性分数 $z=\\omega^{\\top}x$ 把 785 维输入压缩成一个实数，"
    "再用 sigmoid 把它解释为正类概率；二是掌握由最大似然导出负对数似然损失、再对参数求梯度、"
    "最后用梯度下降更新参数的完整推导；三是把上述推导逐条落实到 NumPy 的数组运算上，"
    "得到可运行、可复现的训练与评估程序。Matplotlib 只承担绘图工作，不参与任何模型计算。"
)
P(
    "实验数据为 MNIST 手写数字数据集，程序直接读取项目内已下载的文件 data/mnist_data/mnist.npz。"
    "该文件保存官方划分的 60000 个训练样本和 10000 个测试样本。程序只取出标签为 3 和 6 的样本，"
    "丢弃其余八个数字，筛选后两类的数量见表 1。"
)
TBL(
    1,
    "筛选后数字 3 与数字 6 的样本数量",
    ["集合", "数字 3", "数字 6", "合计"],
    [["训练集", "6131", "5918", "12049"], ["测试集", "1010", "958", "1968"]],
    ["l", "c", "c", "c"],
    [1.6, 1.5, 1.5, 1.5],
)
P(
    "原始图像为 28×28 的灰度图，每个像素是 0～255 的整数，一幅图像共 784 个像素。两类样本的数量"
    "并不相等，训练集中数字 6 比数字 3 少 213 个，测试集中数字 6 比数字 3 少 52 个，这种轻微的"
    "类别不平衡在数据筛选后自然保留，实验没有做重采样处理。"
)
P(
    "数据划分方面：训练集只用于参数更新和每轮指标的记录；测试集在全部 20 轮训练完成后才被使用，"
    "用于按式(9)计算准确率并抽取样本展示，既不参与参数更新，也没有参与学习率、批大小、训练轮数的"
    "选择——这三个量都是程序中写定的常量。因此测试集给出的准确率可以看作模型在这批固定数据上的"
    "评测结果，而不是模型选择过程中的中间量。"
)

H2("1.2 数据预处理")
P(
    "预处理分两步完成：加载函数 load_binary_mnist 负责读取、筛选、展开和归一化，主程序负责标签"
    "转换和追加常数特征。整体数据流为：原始 (28,28) 的 uint8 图像 → 筛选 3/6 两类 → 展开为 784 维 "
    "→ 除以 255 归一化 → 得到 0/1 标签 → 转换为 −1/+1 标签 → 追加常数特征 1 → 送入训练。"
)
NUM(
    "**读取与筛选。** 调用 load_binary_mnist(3, 6)。函数用布尔掩码 (y_train == 3) | "
    "(y_train == 6) 同时保留两类样本，对训练集与测试集分别执行，因而官方划分在筛选后仍然保持"
    "不变。",
    "**展开为向量。** 每张 28×28 图像经 reshape(-1, 28*28) 成为 784 维向量，像素按行优先顺序排列。"
    "展开后图像的空间结构不再显式保留，784 个像素直接作为 784 个特征参与线性组合。",
    "**归一化。** 数组先转换为 float32，再整体除以 255.0，使每个特征的取值落在 [0,1] 区间。"
    "这样做的原因是：像素原始尺度为 0～255，若直接输入，785 维线性组合 $z=\\omega^{\\top}x$ 的"
    "数值会很大，sigmoid 迅速饱和到 0 或 1，输出对参数的导数接近 0，梯度下降几乎没有效果；"
    "归一化后各特征处于同一尺度，分数 $z$ 的量级可控，训练更稳定。",
    "**标签转换。** 加载函数返回 0/1 标签（数字 3 对应 0，数字 6 对应 1）；主程序用 "
    "np.where(y_train == 0, -1, +1) 把它转换成 −1/+1：数字 3 → $y=-1$，数字 6 → $y=+1$。"
    "此后损失与梯度都在这套 ±1 标签下计算。",
    "**合并偏置。** 用 np.column_stack 在每个样本末尾追加一个常数 1，样本变为 "
    "$x=(x_1,\\dots,x_{784},1)^{\\top}$，参数变为 $\\omega=(w_1,\\dots,w_{784},b)^{\\top}$，"
    "其中最后一位 $b$ 就是偏置。这样式(1)的点积 $\\omega^{\\top}x$ 自动把偏置算了进去，"
    "程序中不需要为 $b$ 单独写一条更新式，也不会出现“偏置不更新”的错误。",
    "**形状核对。** 预处理完成后主要数组的形状见表 2。其中 (12049,) 是一维数组的写法"
    "（逗号是 Python 单元素元组的语法），不是 12049 行的列向量；正因为 $y$ 是一维数组，"
    "后面的 y_batch * z 才能按样本逐元素相乘。",
)
TBL(
    2,
    "预处理后主要数组的形状与含义",
    ["数组", "形状", "含义"],
    [
        ["X_train", "(12049, 785)", "12049 个训练样本，784 个像素特征加 1 个常数特征"],
        ["y_train", "(12049,)", "训练样本标签，取值 −1（数字 3）或 +1（数字 6）"],
        ["X_test", "(1968, 785)", "1968 个测试样本的特征"],
        ["y_test", "(1968,)", "测试样本标签，取值 −1（数字 3）或 +1（数字 6）"],
        ["ω", "(785,)", "784 个像素权重加偏置 b"],
    ],
    ["c", "c", "l"],
    [1.2, 1.4, 3.9],
)

H2("1.3 模型原理")
P(
    "**符号约定。** 第 $i$ 个样本的 785 维特征记为列向量 $x_i$（最后一个分量恒为常数 1），"
    "参数记为列向量 $\\omega$（最后一个分量为偏置 $b$），二者都取自 $\\mathbb{R}^{785}$；"
    "上标 $\\top$ 表示转置，因此 $\\omega^{\\top}x_i$ 是一个实数。标签 $y_i\\in\\{-1,+1\\}$；"
    "$n$ 为训练集样本数（12049），$m$ 为一个批次内的样本数（不超过 128），$\\eta>0$ 为学习率。"
    "行文中变量用斜体，函数名与算子名（$\\ln$、$\\sigma$、$\\nabla$）用正体。"
)
P(
    "**线性分数。** 模型先对 785 维输入做内积，得到一个实数分数 $z_i$："
)
FORM(1, [r"z_i=\omega^{\top}x_i"])
P(
    "分数 $z_i$ 的符号表示样本落在决策边界的哪一侧：$z_i>0$ 为参数所指的正侧，$z_i<0$ 为负侧；"
    "$|z_i|$ 越大，样本离边界越远。把 $b$ 与 784 个权重一起放进 $\\omega$、把常数 1 放进 $x_i$，"
    "式(1)就把偏置自动包含在内。"
)
P(
    "**sigmoid 函数。** $\\sigma$ 把实数压缩到开区间 $(0,1)$，其输出可解释为正类（数字 6）的概率："
)
FORM(2, [r"\sigma(z)=\frac{1}{1+e^{-z}}"])
P(
    "它满足 $\\sigma(-z)=1-\\sigma(z)$ 与 $\\sigma(0)=1/2$ 两个性质。第一个性质在把梯度写成等价形式时"
    "要用到；第二个性质说明零分数对应两类各占一半的不确定状态。$\\sigma$ 在 $\\mathbb{R}$ 上严格单调"
    "递增，所以 $z$ 越大正类概率越大，按 $1/2$ 判决与按 $z$ 的符号判决完全等价。"
)
P("**类别条件概率。** 记数字 6 为正类、数字 3 为负类，则给定特征与参数时两类的条件概率为：")
FORM(
    3,
    [
        r"P(Y=+1\mid x;\omega)=\sigma(\omega^{\top}x)",
        r"P(Y=-1\mid x;\omega)=1-\sigma(\omega^{\top}x)",
        r"P(Y=y\mid x;\omega)=\sigma(y\,\omega^{\top}x)",
    ],
)
P(
    "第三行把两种情形合并成一个式子：$y=+1$ 时它等于 $\\sigma(\\omega^{\\top}x)$，$y=-1$ 时由 "
    "$\\sigma(-z)=1-\\sigma(z)$ 等于 $1-\\sigma(\\omega^{\\top}x)$。这一合并写法使似然可以只写一个"
    "连乘式，也是本实验统一采用 $\\pm 1$ 标签的直接原因。"
)
P(
    "**条件似然。** 若样本之间条件独立，则训练集在给定特征条件下的似然是各样本真实类别概率的连乘："
)
FORM(4, [r"\mathcal{L}(\omega)=\prod_{i=1}^{n}\sigma\left(y_i\,\omega^{\top}x_i\right)"])
P("其中 $\\mathcal{L}$ 记似然函数，以便与后面的平均损失 $L$ 区分。")
P(
    "**负对数似然与平均损失。** 式(4)是 $n$ 个概率的连乘，数值上很小、且随 $n$ 增大迅速下溢，"
    "所以先取对数把乘积化为求和；再由“最大化似然”与“最小化负对数似然”等价，得到单样本损失与"
    "平均损失："
)
FORM(
    5,
    [
        r"\ell_i(\omega)=\ln\left(1+e^{-y_i\,\omega^{\top}x_i}\right)",
        r"L(\omega)=\frac{1}{n}\sum_{i=1}^{n}\ell_i(\omega)",
    ],
)
P(
    "其中 $\\ln$ 表示自然对数。第一行由 $\\sigma(yz)=1/(1+e^{-yz})$ 取负对数直接得到。对于确定的"
    "类别标签，该损失就是交叉熵：交叉熵等于真实类别对数概率的相反数 $-\\ln\\sigma(y_i z_i)$，而 "
    "$-\\ln\\sigma(y_i z_i)=\\ln(1+e^{-y_i z_i})$。也就是说，负对数似然损失在二分类情形下就是交叉熵，"
    "两者只是从不同角度写同一个量。"
)
P("**梯度。** 对式(5)的单样本损失求关于 $\\omega$ 的梯度，先对分数 $z_i$ 求导，再用链式法则相乘：")
FORM(
    None,
    [
        r"\frac{\partial \ell_i}{\partial z_i}=-y_i\,\sigma(-y_i z_i)",
        r"\frac{\partial z_i}{\partial \omega}=x_i",
        r"\frac{\partial \ell_i}{\partial \omega}=-y_i\,\sigma(-y_i z_i)\,x_i",
    ],
)
P(
    "**负号的来源。** $\\ell_i$ 是复合函数：外层 $\\ln$ 对括号内的量求导得 $1/(1+e^{-y_i z_i})$，"
    "内层 $-y_i z_i$ 对 $z_i$ 求导得 $-y_i$，两者相乘得 "
    "$-y_i e^{-y_i z_i}/(1+e^{-y_i z_i})=-y_i\\sigma(-y_i z_i)$。负号来自内层的 $-y_i z_i$，"
    "不能遗漏；若在代码中漏掉这个负号，更新方向会反过来，训练时损失将上升而不是下降。"
)
P("对一个含 $m$ 个样本的小批量取平均，得到程序中实际使用的梯度：")
FORM(
    6,
    [
        r"\nabla L(\omega)=-\frac{1}{m}\sum_{i=1}^{m}y_i\,\sigma(-y_i\,\omega^{\top}x_i)\,x_i",
        r"=-\frac{1}{m}\sum_{i=1}^{m}\left[1-\sigma(y_i\,\omega^{\top}x_i)\right]y_i\,x_i",
    ],
)
P(
    "第二行由 $\\sigma(-u)=1-\\sigma(u)$ 换元得到，两行数值完全相同，本实验的代码采用第一行。"
    "第二行的形式可以看出梯度的直观含义：$\\sigma(y_i z_i)$ 是真实类别的预测概率，"
    "它越接近 1（样本被判得越准），$1-\\sigma(y_i z_i)$ 越接近 0，该样本对梯度的贡献也越小。"
)
P(
    "**参数更新。** 用上标 $(t)$ 表示第 $t$ 次更新后的参数，沿负梯度方向走一步，步长由学习率 "
    "$\\eta$ 控制："
)
FORM(7, [r"\omega^{(t+1)}=\omega^{(t)}-\eta\,\nabla L(\omega^{(t)})"])
P("**分类决策。** 训练结束后对每个样本计算正类概率 $p_i$，再按 $1/2$ 阈值判决：")
FORM(
    8,
    [
        r"p_i=\sigma(\omega^{\top}x_i)",
        r"\hat{y}_i=\begin{cases}+1, & p_i\ge 1/2\\ -1, & p_i<1/2\end{cases}",
    ],
)
P(
    "最后把 $\\hat{y}_i=+1$ 映射回数字 6、$\\hat{y}_i=-1$ 映射回数字 3。程序直接比较概率与 0.5，"
    "与上式一致；按 $1/2$ 判决的好处是判决过程与概率解释共用一套符号，也便于直接检查 p_positive "
    "的数值。"
)
P("**准确率。** 设样本总数为 $N$，预测正确的样本数为 $N_{\\mathrm{correct}}$，则")
FORM(
    9,
    [
        r"\mathrm{accuracy}=\frac{N_{\mathrm{correct}}}{N_{\mathrm{total}}}",
        r"N_{\mathrm{correct}}=\sum_{i=1}^{N}\mathbf{1}\left[\hat{y}_i=y_i\right]",
    ],
)
P(
    "其中 $\\mathrm{1}[\\cdot]$ 为指示函数，方括号内的条件成立时取 1、否则取 0；"
    "$N_{\\mathrm{total}}$ 为样本总数（训练集 12049，测试集 1968）。式(9)就是“预测正确的样本数"
    "除以样本总数”。"
)

H2("1.4 训练设置与训练流程")
P("超参数在程序开头以常量形式给出，本次运行使用的取值见表 3。")
TBL(
    3,
    "训练超参数的取值与作用",
    ["参数", "取值", "在式中的位置与作用"],
    [
        ["learning_rate", "0.1", "式(7)中的步长 η，固定不衰减"],
        ["batch_size", "128", "式(6)中 m 的上限，一次参数更新所用的样本数"],
        ["epochs", "20", "训练集完整遍历的轮数，固定不变，无早停"],
        ["random_seed", "42", "随机数发生器的种子，控制参数初始化与每轮打乱"],
    ],
    ["l", "c", "l"],
    [1.5, 0.9, 4.1],
)
P("此外程序用 DIGIT_NEGATIVE 与 DIGIT_POSITIVE 两个常量指定参与分类的两个数字，本次取 3 与 6。")
P("训练流程按程序执行顺序说明如下。")
NUM(
    "**参数初始化。** 用 np.random.default_rng(42) 建立随机数发生器，再用 "
    "rng.normal(0.0, 0.01, size=785) 生成 $\\omega$，即 785 个均值为 0、标准差为 0.01 的随机数。"
    "取很小的标准差有两个考虑：一是打破各参数取值的对称性，否则所有分量相同会让更新沿对称方向"
    "进行；二是使初始分数 $z$ 接近 0，$\\sigma(z)\\approx 0.5$，不落入 sigmoid 的饱和区。程序在"
    "训练前用初始参数对整个训练集算了一次平均损失，得到 0.70416189，与 $\\sigma(0)=0.5$ 时的理论值 "
    "$-\\ln 0.5\\approx 0.693147$ 很接近，说明初始化确实落在预期位置。",
    "**每轮打乱。** 每个 epoch 开始时调用 rng.permutation(n) 得到一个 12049 维的随机排列 order，"
    "再用同一个 order 同时索引 X 和 y（X_shuffled = X[order]，y_shuffled = y[order]），"
    "保证打乱后图像与标签仍然一一对应。打乱的目的是让每个批次的样本组合在轮与轮之间不同，"
    "避免固定的样本顺序给批次梯度带来系统性偏差。",
    "**划分小批量并逐批更新。** 按 batch_size=128 从打乱后的数据上顺序切片，每个批次依次计算分数 "
    "$z$、每个样本的系数 coefficient、批平均梯度 gradient，然后更新一次 $\\omega$。由于 "
    "12049=94×128+17，每轮共有 95 个批次，最后一个批次只有 17 个样本；该批次同样参与训练，"
    "求平均时除以实际的 $m=17$，而不是补零或直接丢弃。",
    "**每轮结束记录指标。** 一轮中所有批次都更新完毕后，用该轮结束时的同一组 $\\omega$ 对整个"
    "训练集（12049 个样本）重新计算平均损失和准确率，打印并追加到 history。注意这两个指标不是"
    "“各批次指标的算术平均”，而是在整轮结束后对全量训练数据的一次重新计算。",
    "**结束。** 固定训练 20 轮后停止，既没有早停，也没有学习率衰减或任何调度。因为每轮 95 次"
    "参数更新，20 轮共 95×20=1900 次参数更新。",
)
P(
    "**三个概念的层次关系。** epoch 指整个训练集完整参与一次训练，本实验共 20 个；batch_size 指"
    "一次梯度计算所用的样本数上限，这里是 128；一次参数更新指取一个批次、算一次梯度、把 $\\omega$ "
    "修改一次。1 个 epoch 包含 95 次参数更新，20 个 epoch 包含 1900 次参数更新，三者不是同一层次"
    "的概念，不能混用。例如“训练了 20 次”应当说成“训练了 20 轮、共 1900 次参数更新”。"
)

H1("2. 实验代码")

H2("2.1 代码组织")
P(
    "程序由两个文件组成，都在 src/ 目录下。src/load_data.py 负责读取 data/mnist_data/mnist.npz、"
    "筛选数字 3 和 6、把每张 28×28 图像展开成 784 维并归一化，返回 0/1 标签；"
    "src/luojihuigui.py 负责把标签转换为 ±1、追加常数特征、训练参数、评估并保存结果。"
    "在项目根目录执行 python src/luojihuigui.py 即可运行，脚本所在目录会自动加入模块搜索路径。"
    "核心计算只用 NumPy，Matplotlib 仅在 save_figures 内部导入并设为 Agg 后端，只用于绘图。"
    "两个文件末尾的 if __name__ == \"__main__\": 块只在直接运行该文件时执行。"
)

H2("2.2 src/load_data.py")
P(
    "**（1）函数定义与参数检查。** 定义唯一的数据加载函数，并检查两个数字参数不能相同、"
    "必须落在 0～9 之间。"
)
CODE(seg(LD, 1, 12))

P("**（2）读取数据文件。** 以本文件位置定位项目根目录，从 mnist.npz 中读出四个数组。")
CODE(seg(LD, 14, 26))

P("**（3）筛选两类样本。** 用布尔掩码保留标签为 0 或 1 的样本，训练集与测试集各做一次。")
CODE(seg(LD, 28, 35))

P("**（4）转换标签为 0/1。** 原数字标签映射为二分类标签：digit_1 记为 1，digit_0 记为 0。")
CODE(seg(LD, 37, 40))

P("**（5）展开与归一化。** 每张 28×28 图像展平成 784 维向量，再由 uint8 转成 float32 并除以 255。")
CODE(seg(LD, 42, 46))
P("这段构造的是式(1)中的输入向量：$x_i$ 的 784 个分量在此展开并缩放到 $[0,1]$。")

P(
    "**（6）单独运行时的演示入口。** 仅在直接运行本文件时执行，演示的是数字 3 和 7，"
    "只打印形状，不影响主程序。"
)
CODE(seg(LD, 49, 56))

H2("2.3 src/luojihuigui.py")
P(
    "**（1）导入、超参数、类别常量与路径。** 把学习率、批大小、轮数、随机种子写成常量，"
    "用两个常量指定参与分类的数字，并算出输出目录。"
)
CODE(seg(LG, 1, 18))
P(
    "LEARNING_RATE 是式(7)中的 $\\eta$，BATCH_SIZE 是式(6)中 $m$ 的上限；DIGIT_NEGATIVE=3 与 "
    "DIGIT_POSITIVE=6 指定两类对应的数字，其中数字 6 记为正类（$y=+1$）、数字 3 记为负类"
    "（$y=-1$）。"
)

P("**（2）数值稳定的 sigmoid。** 按输入正负分两支计算同一个 sigmoid，使 exp 的自变量不大于 0。")
CODE(seg(LG, 22, 30))
P(
    "这段实现式(2)：$z\\ge 0$ 时用 $1/(1+e^{-z})$，$z<0$ 时用 $e^{z}/(1+e^{z})$，"
    "两式等价，指数都不会溢出。"
)

P("**（3）训练函数的初始化。** 建立随机数发生器、生成参数、算出训练前的初始损失。")
CODE(seg(LG, 33, 45))
P(
    "$\\omega$ 按 $N(0,0.01^2)$ 初始化，即式(1)的参数；initial_loss 是式(5)在初始参数下的整集平均。"
)

P("**（4）每轮打乱与划分小批量。** 每个 epoch 用同一排列索引 X 和 y，再以 128 为步长切片。")
CODE(seg(LG, 47, 55))
P("$m$ 取批次的实际样本数，供式(6)求平均，使只有 17 个样本的最后一批不被低估。")

P("**（5）批次内的分数、梯度与参数更新。** 四行依次对应式(1)、式(6)、式(6)、式(7)。")
CODE(seg(LG, 57, 62))

P("**（6）每轮结束时的指标计算与记录。** 用当前参数对整个训练集重算分数、损失、预测与准确率。")
CODE(seg(LG, 64, 75))
P(
    "np.logaddexp(0.0, −yz) 就是式(5)中的 $\\ln(1+e^{-yz})$；prediction 与 accuracy 对应式(8)与"
    "式(9)。"
)

P(
    "**（7）绘图与保存。** 把损失曲线和 5 张测试样本保存为 PNG；绘图前只取前 784 维，"
    "因为最后一维是常数 1，不是像素。"
)
CODE(seg(LG, 78, 101))

P("**（8）主程序：数据准备与训练。** 取两类数据、转换标签、追加常数特征，训练 20 轮并写入 CSV。")
CODE(seg(LG, 104, 123))
P(
    "标签在此由 0/1 转成 $\\pm1$（式(3)）；末尾追加的常数 1 使 $\\omega^{\\top}x$ 自动包含偏置"
    "（式(1)）。"
)

P("**（9）主程序：训练集与测试集评估。** 先在整个训练集上算一次准确率，再对测试集判决。")
CODE(seg(LG, 125, 139))
P("这里用的是式(8)的阈值判决与式(9)的准确率；测试集只在此处出现，不参与参数更新。")

P("**（10）程序入口。** 只有直接运行本文件时才调用 main()。")
CODE(seg(LG, 142, 143))

H2("2.4 核心训练代码的展开")
P("训练的核心是下面四行，位于 train 函数的批次循环内部：")
CODE(
    "z = X_batch @ omega                              # (m,)，z_i = omega^T x_i\n"
    "coefficient = -y_batch * sigmoid(-y_batch * z)    # (m,)\n"
    "gradient = X_batch.T @ coefficient / m           # (785,)\n"
    "omega = omega - LEARNING_RATE * gradient         # (785,)，沿负梯度更新"
)
P("四行中各个量的形状见表 4。")
TBL(
    4,
    "核心四行中变量的形状与含义",
    ["变量", "形状", "含义与对应公式"],
    [
        ["X_batch", "(m, 785)", "一个批次的样本，m≤128"],
        ["y_batch", "(m,)", "该批次的 −1/+1 标签"],
        ["omega", "(785,)", "当前参数，含偏置"],
        ["z", "(m,)", "每个样本的线性分数，式(1)"],
        ["coefficient", "(m,)", "每个样本的 −y_iσ(−y_i z_i)，式(6)"],
        ["gradient", "(785,)", "批平均梯度，式(6)"],
    ],
    ["c", "c", "l"],
    [1.5, 1.2, 3.8],
)
P(
    "四行依次完成：算分数（式(1)）、算每个样本的系数（式(6)）、按样本加权求和得到梯度（式(6)）、"
    "沿负梯度更新参数（式(7)）。"
)

H2("2.5 数值稳定性处理")
P("程序在三处做了数值稳定性处理，都是同一公式的等价变形。")
NUM(
    "**sigmoid 分支。** $z\\ge 0$ 时用 $1/(1+e^{-z})$，$z<0$ 时用 $e^{z}/(1+e^{z})$。"
    "后者是前者分子分母同乘 $e^{z}$ 的结果，两者在实数范围内相等。$z<0$ 时 $e^{-z}$ 会随 $|z|$ "
    "增大而急剧增长，可能溢出为 inf；改写后两支中 exp 的自变量都不大于 0，指数结果落在 $(0,1]$。",
    "**损失用 np.logaddexp。** np.logaddexp(0, −yz) 计算 $\\ln(e^{0}+e^{-yz})=\\ln(1+e^{-yz})$，"
    "正是式(5)的单样本损失。它先取较大项再取对数，避免直接计算 $1+e^{-yz}$ 时有效数字丢失。",
    "**不连乘概率。** 式(4)是 12049 个概率的连乘，连乘后必然下溢为 0，取对数只能得到 −inf。"
    "程序从不在数值上构造式(4)，而是直接对式(5)求和再取平均。",
)

# --------------------------------------------------------------------------
# 三、实验结果与结果分析
# --------------------------------------------------------------------------
H1("3. 实验结果与结果分析")

H2("3.1 实际运行指标")
P("本次运行的终端输出保存在 outputs/run_log.txt，逐轮指标保存在 outputs/training_history.csv。")
TBL(
    5,
    "本次运行的关键指标",
    ["指标", "数值", "来源"],
    [
        ["初始化训练平均损失", "0.70416189", "运行日志"],
        ["第 1 轮结束时训练平均损失", "0.06285353", "CSV 与运行日志"],
        ["第 20 轮结束时训练平均损失", "0.01864255", "CSV 与运行日志"],
        ["第 20 轮训练准确率", "99.4771%", "CSV 与运行日志"],
        ["训练集正确分类数", "11986", "运行日志"],
        ["测试集正确分类数", "1955", "运行日志"],
        ["测试集样本总数", "1968", "运行日志"],
        ["测试集准确率", "99.3394%", "按式(9)计算"],
        ["测试集误分类数", "13", "1968−1955"],
    ],
    ["l", "c", "c"],
    [3.0, 1.3, 2.2],
)
P("测试集准确率按式(9)计算：")
FORM(None, [r"\mathrm{accuracy}=\frac{1955}{1968}\times 100\%\approx 99.3394\%"])
P(
    "第 20 轮的训练准确率 99.4771% 对应 12049 个训练样本中 11986 个判对、63 个判错"
    "（$12049\\times(1-0.994771)\\approx 63.0$）。"
)

H2("3.2 训练损失曲线")
IMG("outputs/training_loss.png", "图1 逻辑回归训练平均损失随训练轮数的变化", 5.3)
P(
    "图1 的横轴为训练轮数，纵轴为该轮结束时整个训练集的平均损失，纵轴范围约 0.019～0.063。"
)
NUM(
    "损失从第 1 轮的 0.06285353 降到第 20 轮的 0.01864255，第 20 轮的值约为第 1 轮的 30%。"
    "前几轮下降快：第 1 到第 2 轮下降 0.01629，第 2 到第 3 轮下降 0.00770；最后几轮每轮只下降约 "
    "$3.6\\times10^{-4}$，曲线明显变平。",
    "图中的第一个点是第 1 轮结束后的损失，不是训练开始前的初始化损失 0.70416189——"
    "后者没有画入图中。因此图1 反映的是 20 轮训练过程中的变化，下降幅度最大的部分其实发生在"
    "第 1 轮之内（从 0.704 到 0.063）。",
    "记录的是每轮结束时的整体指标，粒度较粗。训练准确率并非逐轮单调上升（例如第 9 轮 "
    "99.3443%、第 10 轮 99.3277%），说明轮级指标本身带有波动，不能要求每一步都改善，"
    "也不能据此宣称已达到全局最优或完全收敛。",
)

H2("3.3 测试集预测结果")
IMG("outputs/test_predictions.png", "图2 部分测试样本的真实数字与预测结果", 6.3)
P(
    "图2 是测试集索引 0～4 的 5 个样本，每张图上方标注真实数字（True）与预测数字（Pred）。"
    "这 5 个样本的真实数字依次为 6、3、6、6、3，预测结果与之完全一致。"
)
P(
    "图2 只展示 1968 个测试样本中的 5 个，而且不是随机抽样，是固定取的前 5 个。99.3394% 是在"
    "全部测试样本上按式(9)计算的；展示的样本全部正确不能说明整个测试集全部正确，实际仍有 13 个"
    "样本被分错，这些样本的图像没有保存，无法分析其错分原因。"
)

H2("3.4 结果分析")
NUM(
    "**模型能够完成本实验的分类任务。** 训练准确率 99.4771%、测试准确率 99.3394%，说明"
    "“线性分数 + sigmoid + 负对数似然 + 梯度下降”这套模型能够把数字 3 和数字 6 区分开。"
    "两个数字的笔画都由弧线构成，线性分类器仍能获得足够的判别信息。",
    "**训练有效的原因。** 像素归一化后 784 个特征处于同一尺度，式(1)的分数不会因像素量级而"
    "过大；20 轮内训练平均损失由 0.0629 持续降到 0.0186，说明式(6)的梯度符号与式(7)的"
    "负梯度方向是正确的——若 coefficient 少乘负号，损失会上升而不是下降。",
    "**测试准确率略低于训练准确率。** 99.3394% 低于 99.4771%，两者相差约 0.14 个百分点。"
    "参数在训练集上拟合，测试集是未参与训练的样本，测试准确率略低是常见情形；测试集只有 "
    "1968 个样本，13 个错分就占 0.66%。两者差距很小，没有明显过拟合的迹象。",
    "**准确率没有达到 100%。** 测试集有 13 个样本被分错，说明当前的线性决策边界并不能把两类"
    "完全分开；99.3394% 是本次运行在这批固定测试数据上的结果。",
    "**结论的适用范围。** 模型的分数是 785 维特征的线性组合，决策边界是一个超平面，不显式利用"
    "图像的二维结构。本实验只涉及数字 3 和 6 两类，结论不能推广到 MNIST 十分类问题，"
    "也不能推广到其他数字组合。",
)

H2("3.5 实验小结")
P(
    "本实验用 Python 与 NumPy 从零实现了逻辑回归：在 MNIST 数字 3/6 二分类任务上，训练集平均"
    "损失由 0.70416189 降到 0.01864255，训练准确率 99.4771%；测试集 1968 个样本中正确分类 "
    "1955 个，准确率 99.3394%。程序中的分数、损失、梯度、更新与评估分别对应式(1)、式(5)、"
    "式(6)、式(7)与式(8)(9)。测试集仍有 13 个样本被错分，模型仍是线性分类器，"
    "结论只适用于本实验的任务与数据范围。"
)

# --------------------------------------------------------------------------
# 正文里遗留的 Unicode 数学写法统一转成行内公式
# --------------------------------------------------------------------------
INLINE_RULES = [
    ("z_i=ωᵀx_i", "$z_i=\\omega^{\\top}x_i$"),
    ("ωᵀx_i", "$\\omega^{\\top}x_i$"),
    ("z=ωᵀx", "$z=\\omega^{\\top}x$"),
    ("ωᵀx", "$\\omega^{\\top}x$"),
    ("x_i∈ℝ⁷⁸⁵", "$x_i\\in\\mathbb{R}^{785}$"),
    ("ω∈ℝ⁷⁸⁵", "$\\omega\\in\\mathbb{R}^{785}$"),
    ("y_i∈{−1,+1}", "$y_i\\in\\{-1,+1\\}$"),
    ("−y_iσ(−y_i z_i)", "$-y_i\\sigma(-y_i z_i)$"),
    ("σ(−y_i z_i)", "$\\sigma(-y_i z_i)$"),
    ("σ(y_i z_i)", "$\\sigma(y_i z_i)$"),
    ("σ(yz)", "$\\sigma(yz)$"),
    ("σ(0)=0.5", "$\\sigma(0)=0.5$"),
    ("σ(z)≈0.5", "$\\sigma(z)\\approx 0.5$"),
    ("σ(z)≥0.5", "$\\sigma(z)\\ge 0.5$"),
    ("−log 0.5≈0.693147", "$-\\ln 0.5\\approx 0.693147$"),
    ("z≥0", "$z\\ge 0$"),
    ("ω←ω−η∇L", "$\\omega\\leftarrow\\omega-\\eta\\nabla L$"),
    ("x=[x1,…,x784,1]", "$x=(x_1,\\dots,x_{784},1)^{\\top}$"),
    ("ω=[w1,…,w784,b]", "$\\omega=(w_1,\\dots,w_{784},b)^{\\top}$"),
    ("12049×(1−0.998657718)≈17.0", "$12049\\times(1-0.998657718)\\approx 17.0$"),
    ("12049=98×128+121", "$12049=98\\times128+121$"),
    ("z_i", "$z_i$"),
    ("y_i", "$y_i$"),
    ("x_i", "$x_i$"),
    ("ℓ_i", "$\\ell_i$"),
    ("ω", "$\\omega$"),
    ("η", "$\\eta$"),
]


def _auto_math(text):
    for old, new in INLINE_RULES:
        if old not in text:
            continue
        out, idx = [], 0
        while True:
            pos = text.find(old, idx)
            if pos < 0:
                out.append(text[idx:])
                break
            if text.count("$", 0, pos) % 2 == 1:  # 已在数学区内
                out.append(text[idx:pos + len(old)])
                idx = pos + len(old)
                continue
            out.append(text[idx:pos])
            out.append(new)
            idx = pos + len(old)
        text = "".join(out)
    return text


def _apply_inline_math():
    for i, b in enumerate(BLOCKS):
        if b[0] == "p":
            BLOCKS[i] = ("p", _auto_math(b[1]))
        elif b[0] == "num":
            BLOCKS[i] = ("num", [_auto_math(t) for t in b[1]])
        elif b[0] == "table":
            _, n, title, header, rows, aligns, widths = b
            BLOCKS[i] = (
                "table", n, title, header,
                [[_auto_math(c) for c in row] for row in rows],
                aligns, widths,
            )


_apply_inline_math()



# --------------------------------------------------------------------------
# 向量统一用粗斜体（\boldsymbol），标量与下标保持斜体
# --------------------------------------------------------------------------
def _vectorize_tex(tex):
    if "\\boldsymbol" in tex:
        return tex
    tex = tex.replace(r"\omega", r"\boldsymbol{\omega}")
    tex = re.sub(r"(?<![\\A-Za-z_])x(?![A-Za-z_])", r"\\boldsymbol{x}", tex)
    tex = re.sub(r"(?<![\\A-Za-z])x_i(?![A-Za-z])", r"\\boldsymbol{x}_i", tex)
    return tex
    tex = tex.replace(r"\omega", r"\boldsymbol{\omega}")
    tex = re.sub(r"(?<![\\A-Za-z])x_i(?![A-Za-z])", r"\\boldsymbol{x}_i", tex)
    tex = re.sub(r"(?<![\\A-Za-z_])x(?![A-Za-z_])", r"\\boldsymbol{x}", tex)
    return tex


def _vectorize_text(text):
    out, pos = "", 0
    for m in re.finditer(r"\$([^$]+)\$", text):
        out += text[pos:m.start()]
        out += "$" + _vectorize_tex(m.group(1)) + "$"
        pos = m.end()
    return out + text[pos:]


def _apply_vector_style():
    for i, b in enumerate(BLOCKS):
        if b[0] == "p":
            BLOCKS[i] = ("p", _vectorize_text(b[1]))
        elif b[0] == "num":
            BLOCKS[i] = ("num", [_vectorize_text(t) for t in b[1]])
        elif b[0] == "table":
            _, n, title, header, rows, aligns, widths = b
            BLOCKS[i] = (
                "table", n, title, header,
                [[_vectorize_text(c) for c in row] for row in rows],
                aligns, widths,
            )
        elif b[0] == "formula":
            _, n, lines = b
            BLOCKS[i] = ("formula", n, [_vectorize_tex(l) for l in lines])


_apply_vector_style()

# --------------------------------------------------------------------------
# Markdown 渲染
# --------------------------------------------------------------------------
def render_md(path):
    out = []
    for b in BLOCKS:
        kind = b[0]
        if kind == "h1":
            out += ["# " + b[1], ""]
        elif kind == "h2":
            out += ["## " + b[1], ""]
        elif kind == "p":
            out += [b[1], ""]
        elif kind == "num":
            for i, t in enumerate(b[1], 1):
                out += [f"（{i}）{t}", ""]
        elif kind == "code":
            out += ["```python", b[1], "```", ""]
        elif kind == "table":
            _, n, title, header, rows, aligns, _w = b
            out += [f"**表 {n} {title}**", ""]
            out += ["| " + " | ".join(header) + " |"]
            out += ["| " + " | ".join(["---"] * len(header)) + " |"]
            for r in rows:
                out += ["| " + " | ".join(r) + " |"]
            out += [""]
        elif kind == "formula":
            _, n, tex = b
            lines = list(tex)
            if n:
                lines[-1] = lines[-1] + r" \qquad (%s)" % n
            out += ["$$", " \\\\\n".join(lines), "$$", ""]
        elif kind == "img":
            _, p, cap, _w = b
            out += [f"![{cap}](../{p})", "", f"**{cap}**", ""]
    path.write_text("\n".join(out), encoding="utf-8")


# --------------------------------------------------------------------------
# DOCX 渲染
# --------------------------------------------------------------------------
FONT_CN = "宋体"
FONT_CODE = "Consolas"

PPR_SUCCESSORS = (
    "w:shd", "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap",
    "w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN",
    "w:bidi", "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind",
    "w:contextualSpacing", "w:mirrorIndents", "w:suppressOverlap", "w:jc",
    "w:textDirection", "w:textAlignment", "w:textboxTightWrap", "w:outlineLvl",
    "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr", "w:pPrChange",
)


def set_run_font(run, name=FONT_CN, size=12, bold=None, italic=None,
                 east_asia=None):
    run.font.name = name
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    rFonts.set(qn("w:eastAsia"), east_asia or name)


def add_shading(paragraph, fill):
    pPr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pPr.insert_element_before(shd, *PPR_SUCCESSORS)


def add_paragraph_border(paragraph, color="BFBFBF", sz=4):
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "4")
        el.set(qn("w:color"), color)
        pBdr.append(el)
    pPr.insert_element_before(pBdr, *PPR_SUCCESSORS)


def table_borders(table, color="808080", sz=6):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        borders.append(el)
    tblPr.insert_element_before(
        borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook",
        "w:tblCaption", "w:tblDescription",
    )
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.insert_element_before(
        layout, "w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription",
    )


def cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.insert_element_before(
        shd, "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText",
        "w:vAlign", "w:hideMark", "w:headers", "w:cellIns", "w:cellDel",
        "w:cellMerge", "w:tcPrChange",
    )


def keep_table_rows(table):
    """整张表尽量不跨页：各行禁止断开，表头在跨页时重复，非末行与下一行同页。"""
    rows = table.rows
    for i, row in enumerate(rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
        if i == 0:
            trPr.append(OxmlElement("w:tblHeader"))
        if i < len(rows) - 1:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.keep_with_next = True


_SPAN = re.compile(r"(\*\*.+?\*\*|\$[^$]+\$)")


def add_inline(paragraph, text, size=12, font=FONT_CN, base_bold=False):
    """写入一段文本，支持 **粗体** 与 $行内公式$。"""
    for piece in _SPAN.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            run = paragraph.add_run(piece[2:-2])
            set_run_font(run, font, size, bold=True)
        elif piece.startswith("$") and piece.endswith("$") and len(piece) > 2:
            paragraph._p.append(mathomml.omath(piece[1:-1], int(round(size * 2)) + (2 if size >= 12 else 0)))
        else:
            run = paragraph.add_run(piece)
            set_run_font(run, font, size, bold=base_bold)


MATH_TOKEN = re.compile(r"(\^\{[^}]*\}|_\{[^}]*\})")


def add_math(paragraph, text, size=12):
    """保留给不需要公式对象的纯文本片段（当前渲染路径已改用 add_inline）。"""
    for part in MATH_TOKEN.split(text):
        if not part:
            continue
        if part.startswith("^{"):
            run = paragraph.add_run(part[2:-1])
            run.font.superscript = True
        elif part.startswith("_{"):
            run = paragraph.add_run(part[2:-1])
            run.font.subscript = True
        else:
            run = paragraph.add_run(part)
        set_run_font(run, "Cambria Math", size, east_asia=FONT_CN)


def render_docx(template, path):
    doc = Document(str(template))
    body = doc.element.body

    # 封面保留，补一行标题；压缩标题与姓名之间的空行，保证封面只有一页。
    cover = doc.paragraphs
    title_anchor = cover[5]
    new_p = title_anchor.insert_paragraph_before()
    new_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = new_p.add_run("基于 NumPy 从零实现逻辑回归的")
    set_run_font(r, FONT_CN, 18, bold=True)
    new_p2 = title_anchor.insert_paragraph_before()
    new_p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = new_p2.add_run("MNIST 手写数字 3 和 6 二分类实验")
    set_run_font(r, FONT_CN, 18, bold=True)

    paras = doc.paragraphs
    i_exam = next(i for i, p in enumerate(paras) if p.text.strip() == "实验一")
    i_name = next(i for i, p in enumerate(paras) if p.text.strip().startswith("姓名"))
    for p in paras[i_exam + 1:i_name]:
        if not p.text.strip():
            p._element.getparent().remove(p._element)
    after = doc.paragraphs
    i_name = next(i for i, p in enumerate(after) if p.text.strip().startswith("姓名"))
    after[i_name].paragraph_format.space_before = Pt(48)
    # 去掉封面顶部的第一个空行，给标题留出空间
    i_title = next(i for i, p in enumerate(after) if p.text.strip() == "机器学习")
    if i_title > 0 and not after[i_title - 1].text.strip():
        after[i_title - 1]._element.getparent().remove(after[i_title - 1]._element)

    # 删除模板第 13 段之后的占位标题与说明
    paras = doc.paragraphs
    i_no = next(i for i, p in enumerate(paras) if p.text.strip().startswith("序号"))
    for p in paras[i_no + 1:]:
        p._element.getparent().remove(p._element)

    # 分页，正文从第 2 页开始
    pb = doc.add_paragraph()
    pb.add_run().add_break(WD_BREAK.PAGE)

    tbl_counter = 0
    for idx, b in enumerate(BLOCKS):
        kind = b[0]
        next_kind = BLOCKS[idx + 1][0] if idx + 1 < len(BLOCKS) else None
        if kind == "h1":
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.space_before = Pt(14)
            pf.space_after = Pt(8)
            pf.line_spacing = 1.5
            pf.keep_with_next = True
            r = p.add_run(b[1])
            set_run_font(r, FONT_CN, 15, bold=True)
        elif kind == "h2":
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.space_before = Pt(10)
            pf.space_after = Pt(4)
            pf.line_spacing = 1.5
            pf.keep_with_next = True
            r = p.add_run(b[1])
            set_run_font(r, FONT_CN, 12.5, bold=True)
        elif kind == "p":
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.first_line_indent = Pt(24)
            pf.line_spacing = 1.5
            pf.space_after = Pt(2)
            if next_kind == "code":
                pf.keep_with_next = True
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_inline(p, b[1])
        elif kind == "num":
            for i, t in enumerate(b[1], 1):
                p = doc.add_paragraph()
                pf = p.paragraph_format
                pf.first_line_indent = Pt(24)
                pf.line_spacing = 1.5
                pf.space_after = Pt(2)
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                r = p.add_run(f"（{i}）")
                set_run_font(r, FONT_CN, 12)
                add_inline(p, t)
        elif kind == "code":
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.left_indent = Inches(0.16)
            pf.right_indent = Inches(0.02)
            pf.first_line_indent = Pt(0)
            pf.space_before = Pt(4)
            pf.space_after = Pt(6)
            pf.line_spacing = 1.0
            pf.keep_together = True
            add_shading(p, "F6F6F6")
            add_paragraph_border(p)
            r = p.add_run(b[1])
            set_run_font(r, FONT_CODE, 9, east_asia="宋体")
        elif kind == "table":
            _, n, title, header, rows, aligns, widths = b
            tbl_counter += 1
            if tbl_counter != n:
                raise SystemExit(f"表号不连续：期望 {tbl_counter}，实际 {n}")
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cpf = cap.paragraph_format
            cpf.space_before = Pt(8)
            cpf.space_after = Pt(3)
            cpf.keep_with_next = True
            cpf.first_line_indent = Pt(0)
            r = cap.add_run(f"表 {n}  {title}")
            set_run_font(r, FONT_CN, 10.5, bold=True)

            table = doc.add_table(rows=1 + len(rows), cols=len(header))
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.autofit = False
            table_borders(table)
            aligns = aligns or ["l"] * len(header)
            widths = widths or [6.3 / len(header)] * len(header)
            for j, w in enumerate(widths):
                for cell in table.columns[j].cells:
                    cell.width = Inches(w)
            data = [header] + rows
            for i, row in enumerate(data):
                for j, val in enumerate(row):
                    cell = table.cell(i, j)
                    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                    para = cell.paragraphs[0]
                    para.paragraph_format.first_line_indent = Pt(0)
                    para.paragraph_format.space_before = Pt(2)
                    para.paragraph_format.space_after = Pt(2)
                    para.paragraph_format.line_spacing = 1.0
                    al = aligns[j]
                    para.alignment = (
                        WD_ALIGN_PARAGRAPH.CENTER if al == "c"
                        else WD_ALIGN_PARAGRAPH.LEFT
                    )
                    add_inline(para, str(val), size=9, base_bold=(i == 0))
                    if i == 0:
                        cell_shading(cell, "EDEDED")
            keep_table_rows(table)
            spacer = doc.add_paragraph()
            spacer.paragraph_format.space_after = Pt(4)
            spacer.paragraph_format.line_spacing = 1.0
        elif kind == "formula":
            _, n, lines = b
            for k, line in enumerate(lines):
                p = doc.add_paragraph()
                pf = p.paragraph_format
                pf.first_line_indent = Pt(0)
                pf.space_before = Pt(6 if k == 0 else 0)
                pf.space_after = Pt(6 if k == len(lines) - 1 else 0)
                pf.line_spacing = 1.2
                pf.keep_with_next = k < len(lines) - 1
                pf.tab_stops.add_tab_stop(Inches(3.25), WD_TAB_ALIGNMENT.CENTER)
                pf.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
                p.add_run("\t")
                p._p.append(mathomml.omath(line, 27))
                if n and k == len(lines) - 1:
                    r = p.add_run("\t(%s)" % n)
                    set_run_font(r, FONT_CN, 12)
        elif kind == "img":
            _, rel, cap, width = b
            doc.add_picture(str(ROOT / rel), width=Inches(width))
            p = doc.paragraphs[-1]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            cp = doc.add_paragraph()
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cpf = cp.paragraph_format
            cpf.space_before = Pt(0)
            cpf.space_after = Pt(8)
            cpf.first_line_indent = Pt(0)
            r = cp.add_run(cap)
            set_run_font(r, FONT_CN, 10.5, bold=True)

    # 页脚页码
    for section in doc.sections:
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = fp.add_run()
        fld = OxmlElement("w:fldSimple")
        fld.set(qn("w:instr"), "PAGE")
        r_el = OxmlElement("w:r")
        t_el = OxmlElement("w:t")
        t_el.text = "1"
        r_el.append(t_el)
        fld.append(r_el)
        fp._p.append(fld)
        set_run_font(run, FONT_CN, 9)

    doc.save(str(path))


if __name__ == "__main__":
    build_md = OUT / "实验报告-逻辑回归MNIST36.md"
    render_md(build_md)
    docx_path = OUT / "实验报告-逻辑回归MNIST36.docx"
    try:
        render_docx(Path(r"C:\Users\gujunxi\Desktop\实验报告.docx"), docx_path)
    except PermissionError:
        docx_path = OUT / "实验报告-逻辑回归MNIST36-新.docx"
        render_docx(Path(r"C:\Users\gujunxi\Desktop\实验报告.docx"), docx_path)
        print("原文件被占用，已另存为:", docx_path.name)
    print("md  :", build_md)
    print("docx:", docx_path)
    print("blocks:", len(BLOCKS), "tables:", len(TABLES))
