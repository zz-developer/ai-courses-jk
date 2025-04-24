---
# You can also start simply with 'default'
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://cover.sli.dev
# some information about your slides (markdown enabled)
title: "Course 2: Derivative and MLP"
info: |
  ## Course 2: Derivative and MLP

  Lecture in Jike Club.
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
# take snapshot for each slide in the overview
overviewSnapshots: true
---

# Course 02: 导数与多层感知机

Derivative and MLP

$$
f^\prime(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

---
layout: section
---

# Section 1: 神经网络与多层感知机

---
layout: two-cols
---

# 神经元

<v-clicks>

我们已经学习过线性回归模型，而多个线性回归模型的组合可以看作是一个神经网络．

每个神经元都可以看作是一个线性回归模型，输入为 $x$，输出为 $y$．

$$
\boldsymbol{t}=f\left(\boldsymbol{W}^\textsf{T}\boldsymbol{A}+\boldsymbol{b}\right)
$$

其中，$\boldsymbol{t}$ 为输出，$\boldsymbol{W}$ 为权重，$\boldsymbol{A}$ 为输入，$\boldsymbol{b}$ 为偏置．

</v-clicks>

::right::

<v-clicks>
<img src="./figures/Ncell.png" class="w-[80%] mx-auto" />
</v-clicks>

---
layout: two-cols
---

# 神经网络

<v-clicks>

将不同的神经元组合在一起，就可以形成一个神经网络．

神经网络可以分为多个层，每一层的神经元都可以看作是一个线性回归模型，而每一层的输出都可以看作是下一层的输入．

神经网络的输入层为输入数据，输出层为预测结果，中间的层为隐藏层．

</v-clicks>

::right::

<v-clicks>
<img src="./figures/neural-network.png" class="w-[80%] mx-auto" />
</v-clicks>

---

# 神经网络的优化

<v-clicks>

神经网络的优化可以看作是一个最优化问题，目标是最小化损失函数．

损失函数可以看作是一个多元函数，输入为神经网络的参数，输出为损失值．

神经网络的优化可以使用梯度下降法，梯度下降法的目标是找到一个最小值，使得损失函数最小．

</v-clicks>

---

# 多层感知机

<v-clicks>

多层感知机（MLP）是最基础的神经网络模型，由输入层、隐藏层和输出层组成．

其包含多个层：

1. 输入层：输入数据的特征向量．
2. 隐藏层：多个神经元组成的层，每个神经元都可以看作是一个线性回归模型．
3. 输出层：输出预测结果的层．

每个神经元由以下部分组成：

- 输入：输入数据的特征向量．
- 权重：每个输入数据的权重，表示该输入数据对输出结果的影响程度．
- 偏置：每个神经元的偏置，表示该神经元的输出结果．
- 激活函数：每个神经元的激活函数，表示该神经元的输出结果．
- 输出：每个神经元的输出结果，表示该神经元的输出结果．

</v-clicks>

---

# 激活函数

<v-clicks>

激活函数用于将神经元的输出结果转换为非线性函数．线性函数的组合仍然是线性函数，而我们需要非线性函数来拟合复杂的函数关系．

常见的激活函数有 $\sigma$、$\tanh$ 和 $\mathrm{ReLU}$ 等．

$$
\sigma(x) = \frac{1}{1 + e^{-x}} \quad \text{Sigmoid}
$$

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \quad \text{Tanh}
$$

$$
\mathrm{ReLU}(x) = \max(0, x) \quad \text{ReLU}
$$

</v-clicks>

---

# Dropout

<v-clicks>

Dropout 是一种正则化方法，用于防止神经网络的过拟合．

Dropout 的原理是随机丢弃一部分神经元，使得神经网络在训练时不会依赖于某些特定的神经元，从而提高模型的泛化能力．

</v-clicks>

---

# 反向传播

<v-clicks>

反向传播是神经网络的训练算法，用于计算神经网络的梯度．

反向传播的原理是链式法则，将损失函数对每个参数的偏导数计算出来，从而更新神经网络的参数．

我们将在后面详细讲解反向传播的原理和实现．

</v-clicks>

---
layout: section
---

# Section 2: 导数与梯度下降

---

# 导数、偏导与梯度

<v-clicks>

导数是函数在某一点处的切线斜率．

$$
f^\prime(x) = \lim_{h \to 0} \dfrac{f(x+h) - f(x)}{h}
$$

对于多元函数，偏导数对于某个变量的导数．如：

$$
\dfrac{\partial f}{\partial x_i} = \lim_{h \to 0} \dfrac{f(x_1, x_2, \ldots, x_i+h, \ldots, x_n) - f(x_1, x_2, \ldots, x_i, \ldots, x_n)}{h}
$$

而梯度是函数在某一点处的切线斜率的向量表示．

$$
\nabla f(x) = \begin{pmatrix}
\dfrac{\partial f}{\partial x_1} \\
\dfrac{\partial f}{\partial x_2} \\
\vdots \\
\dfrac{\partial f}{\partial x_n}
\end{pmatrix}
$$

</v-clicks>


---
layout: two-cols
---

# 梯度下降

<v-clicks>

对于函数 $f(x)$，我们希望找到一个最小值 $x^*$，使得 $f(x^*)$ 最小．

此时，随机选取一点 $x=x_0$，计算 $f(x_0)$ 的导数 $f^\prime(x_0)$，如果 $f^\prime(x_0) > 0$，则说明 $x_0$ 在最小值的右侧，此时我们需要向左移动一点；如果 $f^\prime(x_0) < 0$，则说明 $x_0$ 在最小值的左侧，此时我们需要向右移动一点．

</v-clicks>

::right::

<v-clicks>

<img src="./figures/grad-desc.png" class="w-[80%] mx-auto" />

</v-clicks>

---
layout: section
---

# Section 2: 神经网络

---

# 回归函数

<v-clicks>

一元线性回归的回归函数为：

$${\color{red}\hat{y}} = \theta_0 + \theta_1 x$$

- $\hat{y}$：预测值，其中的 `^` 表示预测值，以区别于真实值 $y$．
- $\theta_0$：截距，表示当 $x=0$ 时的 $y$ 值．
- $\theta_1$：斜率，表示 $x$ 增加一个单位时 $y$ 的变化量．

可以看出，回归函数是一个关于 $x$ 的线性函数．

</v-clicks>

---
layout: section
---

# 如何知道拟合的好坏？

---

# 残差与损失函数

<v-clicks>

- 残差：$e_i = y_i - \hat{y}_i$，表示**单个**样本的预测值与真实值的差距．

</v-clicks>

<div class="grid grid-cols-2 gap-4">
  <div>
    <img v-click src="./images/shanghai_house_residuals.png" />
  </div>
  <div v-click class="pt-4">
    残差的绝对值越小，表示在该样本上的预测越准确．一般而言，残差应当在以 0 为中心的上下均匀分布．可以看出，该拟合的残差分布整体较为均匀，拟合效果还行，但应该有更好的方法．
  </div>
</div>

---

# 残差与损失函数

<p class="text-[1.5rem]" v-click>如果把所有样本的残差加起来呢？</p>

<div v-click v-click.hide>

$$J(\theta_0, \theta_1) = \sum_{i=1}^{n} e_i = \sum_{i=1}^{n} (y_i - \hat{y}_i) = \sum_{i=1}^{n} (y_i - \theta_0 - \theta_1 x_i)$$

</div>

<v-clicks>

湿拜．

</v-clicks>

<div v-click v-click.hide>

$$J(\theta_0, \theta_1) = \sum_{i=1}^{n} \lvert e_i \rvert = \sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert = \sum_{i=1}^{n} \lvert y_i - \theta_0 - \theta_1 x_i \rvert?$$

</div>

<v-clicks>

$$J(\theta_0, \theta_1) = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - \theta_0 - \theta_1 x_i)^2$$

你说得对，但是<span v-mark.circle>乘法</span>和绝对值哪个简单？

~~肯定是绝对值啊！~~

但是平方更加符合正态分布的假设．

</v-clicks>

---

# 最小二乘法

<v-clicks>

1. 展开公式：
   $$
   Q(\theta_0, \theta_1) = \sum_{i=1}^n [(y_i - \theta_1 x_i - \bar{y} + \theta_1 \bar{x}) + (\bar{y} - \theta_1 \bar{x}) - \theta_0]^2
   $$

2. 简化条件：
   $$
   Q(\theta_0, \theta_1) = \sum_{i=1}^n [(y_i - \bar{y}) - \theta_1(x_i - \bar{x}) + (\bar{y} - \theta_1 \bar{x} - \theta_0)]^2
   $$

3. 展开平方式：
   $$
   Q(\theta_0, \theta_1) = \sum_{i=1}^n [(y_i - \bar{y}) - \theta_1(x_i - \bar{x})]^2 + 2 \sum_{i=1}^n [(y_i - \bar{y}) - \theta_1(x_i - \bar{x})] \cdot \\ (\bar{y} - \theta_1 \bar{x} - \theta_0) + n (\bar{y} - \theta_1 \bar{x} - \theta_0)^2
   $$

</v-clicks>

---

4. <span v-mark.circle v-mark.red>注意到</span>中间表达式：
   $$
   \sum_{i=1}^n [(y_i - \bar{y}) - \theta_1(x_i - \bar{x})] (\bar{y} - \theta_1\bar{x} - \theta_0) = (\bar{y} - \theta_1 \bar{x} - \theta_0) \sum_{i=1}^n [(y_i - \bar{y}) - \theta_1(x_i - \bar{x})]
   $$

   > 我可不是<span v-mark.underline>注意力大师</span>

   可以简化为：
   $$
   (\bar{y} - \theta_0 \bar{x} - \theta_1) [n\bar{y} - n\bar{y} - \theta_1(n\bar{x} - n\bar{x})] = 0
   $$

<v-clicks>

5. 因此：
   $$
   Q(a, b) = \sum_{i=1}^n [(y_i - \bar{y}) - \theta_1(x_i - \bar{x})]^2 + n (\bar{y} - \theta_1 \bar{x} - \theta_0)^2
   $$


可根据该公式求解 $\theta_0$ 和 $\theta_1$：$\begin{cases}\theta_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} \\\theta_0 = \bar{y} - \theta_1 \bar{x}\end{cases}$

<span class="color-gray">过程来自《普通高中教科书 数学 选择性必修 第三册》</span>

</v-clicks>

---

# 仁济视角：计算机如何优化

<v-clicks>

<div class="text-[48px] text-center mt-8">梯度下降！</div>

> 相关算法需要<span v-mark.red>偏导数</span>和<span v-mark.red>向量</span>的知识，将在下节课详细讲解．

</v-clicks>

---
layout: section
---

# 如何手搓一个回归模型？

---

# 方法一：`Excel` 即可

<v-clicks>

当然，后面就不行了．

<video controls class="w-[60%]">
<source src="./images/excel-regression.mov" type="video/mp4">
</video>

</v-clicks>

---

# 方法二：使用 `scikit-learn`

<v-clicks>

除了一元线性回归，`scikit-learn` 还支持多元线性回归、逻辑回归等．

```python {6-7}{lines:true}
from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.read_csv('data.csv')

model = LinearRegression()
model.fit(data[['x']], data['y'])

print(model.coef_, model.intercept_)
```

环境搭建：

```bash
pip install scikit-learn pandas numpy
```

</v-clicks>

---
layout: section
---

# Do It Yourself

---
layout: end
---

# Thank you!
