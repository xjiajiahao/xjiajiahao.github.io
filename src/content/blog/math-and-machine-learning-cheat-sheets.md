---
title: "数学与机器学习 Cheat Sheets"
description: "汇总优化不等式、矩阵分析、概率论、Metropolis–Hastings 与最优传输中的常用定义、公式和推导。"
publishedAt: 2026-10-04
tags: ["数学", "机器学习", "概率论", "优化"]
readingMinutes: 25
accent: green
draft: false
---
## 目录

- [1. 优化中的常用不等式](#1-优化中的常用不等式)
- [2. 常见不等式与分析公式](#2-常见不等式与分析公式)
- [3. 矩阵分析](#3-矩阵分析)
- [4. 概率论](#4-概率论)
- [5. Metropolis-Hastings 算法](#5-metropolis-hastings-算法)
- [6. 最优传输](#6-最优传输)

---

## 1. 优化中的常用不等式

### 基本定义与记号

以下允许函数取扩展实数值：$f:\mathcal X\to(-\infty,+\infty]$。其有效定义域为

$$
\operatorname{dom}f=\{x\in\mathcal X:f(x)<+\infty\}.
$$

- **凸集**：若对任意 $x,y\in\mathcal X$ 和 $\gamma\in[0,1]$，都有 $\gamma x+(1-\gamma)y\in\mathcal X$，则 $\mathcal X$ 是凸集。
- **凸函数**：若 $\operatorname{dom}f$ 是凸集，且

  $$
  f(\gamma x+(1-\gamma)y)
  \le \gamma f(x)+(1-\gamma)f(y),
  $$

  则 $f$ 是凸函数。
- **正常函数（proper function）**：$f$ 不恒等于 $+\infty$，并且从不取 $-\infty$。
- **闭凸函数（closed convex function）**：$f$ 是凸函数，且其 epigraph

  $$
  \operatorname{epi}f=\{(x,t):f(x)\le t\}
  $$

  是闭集。在有限维空间中，这等价于 $f$ 是下半连续凸函数。
- **次梯度与次微分**：对凸函数 $f$，若

  $$
  f(y)\ge f(x)+\langle g,y-x\rangle,
  \quad \forall y,
  $$

  则 $g$ 是 $f$ 在 $x$ 处的次梯度；所有次梯度组成

  $$
  \partial f(x)=\{g:f(y)\ge f(x)+\langle g,y-x\rangle,\ \forall y\}.
  $$

  若 $f$ 在 $x$ 处可微，则 $\partial f(x)=\{\nabla f(x)\}$。
- **Fenchel 共轭**：

  $$
  f^*(\theta)=\sup_{\omega}
  \{\langle\theta,\omega\rangle-f(\omega)\}.
  $$
- **近端算子**：对 $\alpha>0$，

  $$
  \operatorname{prox}_{\alpha f}(x)
  =\arg\min_z\left\{f(z)+\frac1{2\alpha}\|z-x\|^2\right\}.
  $$

  若 $f$ 是正常、闭、凸函数，则上式目标函数是强凸的，因而近端点存在且唯一。

### 1.1 $L$-光滑函数

若 $f$ 可微且其梯度是 $L$-Lipschitz 连续的，即

$$
\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|.
$$

等价地，

$$
\left|f(x)-f(y)-\langle \nabla f(y),x-y\rangle\right|
\le \frac L2\|x-y\|^2.
$$

也就是

$$
f(y)+\langle\nabla f(y),x-y\rangle-\frac L2\|x-y\|^2
\le f(x)
\le f(y)+\langle\nabla f(y),x-y\rangle+\frac L2\|x-y\|^2.
$$

若 $f$ 二阶可微，则可写为

$$
\|\nabla^2 f(x)\|_2\le L,
\qquad\text{等价于}\qquad
-LI\preceq\nabla^2 f(x)\preceq LI.
$$

若再假设 $f$ 凸，则 $\nabla^2f(x)\succeq0$，从而简化为 $0\preceq\nabla^2f(x)\preceq LI$。

### 1.2 凸函数

若 $f$ 可微且为凸函数，则

$$
f(y)\ge f(x)+\langle\nabla f(x),y-x\rangle.
$$

等价地，

$$
f(x)\le f(y)+\langle\nabla f(x),x-y\rangle.
$$

梯度具有单调性：

$$
\langle\nabla f(x)-\nabla f(y),x-y\rangle\ge 0.
$$

Jensen 不等式：

$$
f(\mathbb E X)\le \mathbb E[f(X)].
$$

若 $\lambda_i\ge 0$ 且 $\sum_i\lambda_i=1$，则

$$
f\!\left(\sum_i\lambda_i x_i\right)
\le \sum_i\lambda_i f(x_i).
$$

若 $f$ 在凸集 $\mathcal X$ 上可微且凸，则

$$
x^*\in\arg\min_{x\in\mathcal X}f(x)
\iff
\langle\nabla f(x^*),y-x^*\rangle\ge 0,
\quad \forall y\in\mathcal X.
$$


#### 近端算子

设 $f$ 是 Hilbert 空间上的正常、闭、凸函数，$f^*$ 是它的 Fenchel 共轭，且 $\alpha>0$。此时 Moreau 分解为

$$
x=\operatorname{prox}_{\alpha f}(x)
+\alpha\operatorname{prox}_{f^*/\alpha}(x/\alpha).
$$

### 1.4 $L$-光滑且凸

梯度的余强单调性（cocoercivity）：

$$
\frac1L\|\nabla f(x)-\nabla f(y)\|^2
\le
\langle\nabla f(x)-\nabla f(y),x-y\rangle.
$$

此外，

$$
0\le f(x)-f(y)-\langle\nabla f(y),x-y\rangle
\le \frac L2\|x-y\|^2,
$$

以及

$$
f(x)\le f(y)+\langle\nabla f(x),x-y\rangle
-\frac1{2L}\|\nabla f(x)-\nabla f(y)\|^2.
$$

### 1.5 $\mu$-强凸函数

若 $f$ 是 $\mu$-strongly convex，则

$$
f(y)\ge f(x)+\langle\nabla f(x),y-x\rangle
+\frac\mu2\|y-x\|^2.
$$

$\mu=0$ 时即退化为普通凸性。

等价的弦定义是：对任意 $x,y\in\operatorname{dom}f$ 和 $\gamma\in[0,1]$，

$$
f(\gamma x+(1-\gamma)y)
\le
\gamma f(x)+(1-\gamma)f(y)
-\frac\mu2\gamma(1-\gamma)\|x-y\|^2.
$$

梯度具有强单调性：

$$
\langle\nabla f(x)-\nabla f(y),x-y\rangle
\ge \mu\|x-y\|^2.
$$

若 $x^*$ 是凸集 $\mathcal X$ 上的最小点，则对任意 $x\in\mathcal X$，一阶最优性条件给出

$$
\langle\nabla f(x^*),x-x^*\rangle\ge0.
$$

代入强凸性不等式可得二次增长条件

$$
\frac\mu2\|x-x^*\|^2\le f(x)-f(x^*).
$$

因此，二次增长条件对无约束问题和闭凸约束域上的可微强凸优化问题都成立。

### 1.6 $L$-光滑且 $\mu$-强凸

有

$$
\frac{\mu L}{\mu+L}\|x-y\|^2
+\frac1{\mu+L}\|\nabla f(x)-\nabla f(y)\|^2
\le
\langle\nabla f(x)-\nabla f(y),x-y\rangle.
$$

若 $f$ 二阶可微，则

$$
\mu I\preceq\nabla^2f(x)\preceq LI.
$$

梯度下降映射的收缩界：

$$
\|x-y-\varepsilon(\nabla f(x)-\nabla f(y))\|
\le
\max\{|1-\varepsilon\mu|,|1-\varepsilon L|\}\,\|x-y\|.
$$

相应地，在二阶可微情形下，

$$
(1-\varepsilon L)I
\preceq I-\varepsilon\nabla^2f(x)
\preceq(1-\varepsilon\mu)I.
$$

---

## 2. 常见不等式与分析公式

### 2.1 Leibniz 积分法则

设积分上下限依赖于 $x$，则

$$
\frac{d}{dx}\int_{a(x)}^{b(x)}f(x,t)\,dt
=f(x,b(x))b'(x)-f(x,a(x))a'(x)
+\int_{a(x)}^{b(x)}\frac{\partial f(x,t)}{\partial x}\,dt.
$$

当上下限为常数时，

$$
\frac{d}{dx}\int_a^b f(x,t)\,dt
=\int_a^b\frac{\partial f(x,t)}{\partial x}\,dt.
$$

### 2.2 求和与积分的比较

令 $f:\mathbb R_+\to\mathbb R_+$ 单调递减，

$$
S=\sum_{i=1}^n f(i),
\qquad
I=\int_1^n f(x)\,dx,
$$

则

$$
I+f(n)\le S\le I+f(1).
$$

单调递增时，不等号方向相应调整。

### 2.3 Chebyshev 求和不等式

若 $a_1\ge\cdots\ge a_n$ 且 $b_1\ge\cdots\ge b_n$，则

$$
\frac1n\sum_{k=1}^n a_kb_k
\ge
\left(\frac1n\sum_{k=1}^n a_k\right)
\left(\frac1n\sum_{k=1}^n b_k\right).
$$

若两个序列排序方向相反，则不等号反向。

### 2.4 平均数不等式

对正数 $a_1,\dots,a_n$，均方根、算术平均、几何平均与调和平均满足

$$
\sqrt{\frac{\sum_{i=1}^n a_i^2}{n}}
\ge
\frac1n\sum_{i=1}^n a_i
\ge
\sqrt[n]{\prod_{i=1}^n a_i}
\ge
\frac{n}{\sum_{i=1}^n 1/a_i}.
$$

### 2.5 Cauchy-Schwarz 不等式

$$
|\langle x,y\rangle|^2
\le \langle x,x\rangle\langle y,y\rangle
=\|x\|^2\|y\|^2.
$$

有限维形式：

$$
\left(\sum_{i=1}^n a_i b_i\right)^2
\le
\left(\sum_{i=1}^n a_i^2\right)
\left(\sum_{i=1}^n b_i^2\right).
$$

#### 随机变量形式及成立条件

设 $X,Y$ 定义在同一个概率空间 $(\Omega,\mathcal F,\mathbb P)$ 上。称 $X$ **平方可积**，是指

$$
\mathbb E[|X|^2]<\infty.
$$

所有平方可积随机变量组成的空间记作 $L^2(\Omega,\mathcal F,\mathbb P)$；严格地说，几乎处处相等的随机变量视为同一个元素。将随机变量视为 $L^2$ 空间中的向量，是因为随机变量可以像有限维向量一样相加、数乘，并且可以定义内积和由内积诱导的长度：

$$
\|X\|_2=(\mathbb E|X|^2)^{1/2}.
$$

Cauchy-Schwarz 不等式不要求 $X,Y$ 服从正态分布、同分布或相互独立；只要求它们定义在同一概率空间且二阶矩有限，即 $X,Y\in L^2$。

对实值随机变量，$L^2$ 内积定义为

$$
\langle X,Y\rangle_{L^2}=\mathbb E[XY],
$$

所以

$$
|\mathbb E[XY]|^2
\le \mathbb E[X^2]\,\mathbb E[Y^2].
$$

对复值随机变量，内积中必须带复共轭。采用 $\langle X,Y\rangle=\mathbb E[X\overline Y]$ 的约定时，

$$
|\mathbb E[X\overline Y]|^2
\le \mathbb E[|X|^2]\cdot\mathbb E[|Y|^2].
$$

### 2.6 Young 不等式

若 $a,b\ge 0$，$p,q>1$ 且 $1/p+1/q=1$，则

$$
ab\le \frac{a^p}{p}+\frac{b^q}{q}.
$$

特别地，

$$
ab\le \frac{a^2+b^2}{2},
$$

以及对任意 $\varepsilon>0$，

$$
ab\le \frac{a^2}{2\varepsilon}+\frac{\varepsilon b^2}{2}.
$$

### 2.7 Hölder 不等式

在测度空间 $(S,\Sigma,\mu)$ 上，若 $1/p+1/q=1$，则

$$
\|fg\|_1\le \|f\|_p\|g\|_q.
$$

离散形式：

$$
\sum_{k=1}^n|x_ky_k|
\le
\left(\sum_{k=1}^n|x_k|^p\right)^{1/p}
\left(\sum_{k=1}^n|y_k|^q\right)^{1/q}.
$$

随机变量形式：设 $X,Y$ 定义在同一个概率空间上，$p,q>1$ 且 $1/p+1/q=1$。若

$$
\mathbb E|X|^p<\infty,
\qquad
\mathbb E|Y|^q<\infty,
$$

即 $X\in L^p$、$Y\in L^q$，则

$$
\mathbb E[|XY|]
\le
(\mathbb E|X|^p)^{1/p}
(\mathbb E|Y|^q)^{1/q}.
$$

该不等式不要求 $X,Y$ 服从特定分布，也不要求二者独立；有限的 $p$ 阶矩和 $q$ 阶矩就是所需的可积性条件。若某一侧为 $+\infty$，不等式仍可按扩展实数理解，但无法给出有限上界。端点情形 $p=1,q=\infty$ 也成立，此时右端为 $\|X\|_1\|Y\|_\infty$。

**推导。** 若 $\|X\|_p=0$ 或 $\|Y\|_q=0$，结论显然成立。否则令

$$
U=\frac{|X|}{\|X\|_p},
\qquad
V=\frac{|Y|}{\|Y\|_q}.
$$

由 Young 不等式 $uv\le u^p/p+v^q/q$，逐点有

$$
UV\le\frac{U^p}{p}+\frac{V^q}{q}.
$$

两边取期望，并使用 $\mathbb E[U^p]=\mathbb E[V^q]=1$，得到

$$
\mathbb E[UV]\le\frac1p+\frac1q=1.
$$

再乘以 $\|X\|_p\|Y\|_q$ 即得结论。随机变量形式正是测度空间 Hölder 不等式在概率测度下的应用。

### 2.8 Taylor 展开

$$
f(x)=f(a)+\frac{f'(a)}{1!}(x-a)
+\frac{f''(a)}{2!}(x-a)^2
+\cdots+
\frac{f^{(n)}(a)}{n!}(x-a)^n
+o((x-a)^n).
$$

### 2.9 指数与对数相关不等式

若 $1+ax>0$、$x\ne0$ 且 $b/x\ge0$，则

$$
(1+ax)^{b/x}\le e^{ab}.
$$

证明只需使用 $\ln(1+t)\le t$。令 $t=ax$，再乘以非负数 $b/x$：

$$
\frac bx\ln(1+ax)\le\frac bx(ax)=ab.
$$

两边取指数即得上述结论。

若 $x\ge0$，则

$$
1-x+\frac{x^2}{2}\ge e^{-x}.
$$

Bernoulli 不等式：当 $x>-1$ 时，

$$
(1+x)^r
\begin{cases}
\ge 1+rx, & r\ge1\text{ 或 }r\le0,\\
\le 1+rx, & 0\le r\le1.
\end{cases}
$$

另外，

$$
1+x\le e^x,
\qquad
1-\frac1x\le \ln x\le x-1.
$$

### 2.10 Fenchel 共轭与对偶关系

若 $g(u)\le h(u)$ 对所有 $u$ 成立，则共轭函数满足反序关系

$$
g^*(v)\ge h^*(v).
$$

Fenchel-Young 不等式为

$$
f(x)+f^*(y)\ge\langle x,y\rangle.
$$

若 $f$ 是正常闭凸函数，则 Fenchel-Young 等号成立当且仅当变量互为次梯度：

$$
\theta\in\partial f(\omega)
\iff
\omega\in\partial f^*(\theta)
\iff
f(\omega)+f^*(\theta)=\langle\theta,\omega\rangle.
$$

上述等价关系可由次梯度定义直接推出。若 $\theta\in\partial f(\omega)$，则对任意 $z$ 都有

$$
f(z)\ge f(\omega)+\langle\theta,z-\omega\rangle,
$$

移项得到

$$
\langle\theta,z\rangle-f(z)
\le
\langle\theta,\omega\rangle-f(\omega),
\quad\forall z.
$$

这表示函数 $z\mapsto\langle\theta,z\rangle-f(z)$ 在任何 $z$ 处的值，都不超过它在 $z=\omega$ 处的值；而当 $z=\omega$ 时等号确实成立。因此 $z=\omega$ 是一个最大点，其最大值也就是上确界：

$$
\sup_z\{\langle\theta,z\rangle-f(z)\}
=\langle\theta,\omega\rangle-f(\omega).
$$

根据 Fenchel 共轭的定义，左边正是 $f^*(\theta)$，所以

$$
f^*(\theta)=\langle\theta,\omega\rangle-f(\omega).
$$

移项便得到

$$
f(\omega)+f^*(\theta)=\langle\theta,\omega\rangle.
$$

反之，若该等式成立，则 Fenchel 共轭的定义保证对任意 $z$，

$$
\langle\theta,z\rangle-f(z)
\le f^*(\theta)
=\langle\theta,\omega\rangle-f(\omega),
$$

重新移项即可得到次梯度不等式，因此 $\theta\in\partial f(\omega)$。

若 $f$ 是闭凸函数，则 $f^{**}=f$。此外，强凸性与共轭函数的光滑性互为对偶：

$$
f\text{ 关于某范数是 }\mu\text{-强凸}
\iff
f^*\text{ 关于对偶范数是 }\frac1\mu\text{-光滑}.
$$

### 2.11 凸约束问题的一阶最优性条件

设 $\mathcal X$ 是闭凸集，$f$ 是在 $\mathcal X$ 上可微的凸函数。则

$$
x^*\in\arg\min_{x\in\mathcal X}f(x)
\iff
\langle\nabla f(x^*),x-x^*\rangle\ge0,
\quad\forall x\in\mathcal X.
$$

---

## 3. 矩阵分析

### 基本对象与适用范围

设 $A\in\mathbb F^{m\times n}$，其中 $\mathbb F$ 表示标量域，通常取实数域 $\mathbb R$ 或复数域 $\mathbb C$。

| 概念 | 定义 | 适用对象 |
|---|---|---|
| 行列式 | $\det(A)$；可理解为线性变换对有向体积的缩放因子 | 仅方阵 |
| 特征值 | 若存在 $v\ne0$ 使 $Av=\lambda v$，则 $\lambda$ 是特征值 | 仅方阵 |
| 奇异值 | $\sigma_i(A)=\sqrt{\lambda_i(A^*A)}$ | 任意矩阵 |
| Trace（迹） | $\operatorname{tr}(A)=\sum_iA_{ii}$ | 仅方阵 |
| 秩 | 线性无关列（或行）的最大个数，即列空间维数 | 任意矩阵 |
| 谱范数 | $\\|A\\|_2=\sigma_{\max}(A)$ | 任意矩阵 |
| Frobenius 范数 | $\\|A\\|_F=(\sum_{i,j} \| A_{ij} \| ^2)^{1/2}$ | 任意矩阵 |

$A^*$ 表示共轭转置；当矩阵为实矩阵时，$A^*=A^T$。奇异值总是非负实数，而一般矩阵的特征值可能是复数。

### 3.1 分块矩阵的行列式

对分块矩阵

$$
M=\begin{bmatrix}A&B\\C&D\end{bmatrix},
$$

若 $A$ 可逆，则

$$
\det(M)=\det(A)\det(D-CA^{-1}B).
$$

若 $D$ 可逆，则

$$
\det(M)=\det(D)\det(A-BD^{-1}C).
$$

对分块三角矩阵，

$$
\det\begin{bmatrix}A&0\\C&D\end{bmatrix}
=
\det\begin{bmatrix}A&B\\0&D\end{bmatrix}
=\det(A)\det(D).
$$

### 3.2 分块矩阵的逆

令

$$
P=\begin{bmatrix}I_m&A\\0&I_n\end{bmatrix},
$$

则

$$
P^{-1}=\begin{bmatrix}I_m&-A\\0&I_n\end{bmatrix},
\qquad
P^{-1}P=I_{m+n}.
$$

### 3.3 相似矩阵

若存在可逆矩阵 $P$，使

$$
B=P^{-1}AP,
$$

则 $A$ 与 $B$ 相似。相似矩阵有相同的特征值和行列式。

### 3.4 特征值

若 $A\in\mathbb F^{m\times n}$、$B\in\mathbb F^{n\times m}$，则 $AB$ 是 $m\times m$ 方阵，$BA$ 是 $n\times n$ 方阵。假设 $m\le n$，定义特征多项式

$$
C_M(\lambda)=\det(\lambda I-M),
$$

则

$$
C_{BA}(\lambda)=\lambda^{n-m}C_{AB}(\lambda).
$$

因此 $BA$ 比 $AB$ 多出的 $n-m$ 个维度只会贡献零特征值；二者的非零特征值相同，并且每个非零特征值作为特征多项式根的重数也相同。

若 $C_M(\lambda)$ 含有因子 $(\lambda-\lambda_0)^k$，则 $\lambda_0$ 的代数重数为 $k$。几何重数则是 $\dim\ker(M-\lambda_0I)$。

特征值满足

$$
Ax_i=\lambda_i x_i,
\qquad
\det(A-\lambda I)=0.
$$

并且

$$
\operatorname{tr}(A)=\sum_i\lambda_i,
\qquad
\det(A)=\prod_i\lambda_i.
$$

### 3.5 Trace 的常用性质

对方阵 $A=(A_{ij})\in\mathbb F^{n\times n}$，迹定义为主对角线元素之和：

$$
\operatorname{tr}(A)=\sum_{i=1}^nA_{ii}.
$$

迹本身只对方阵定义。如果 $A$ 是复矩阵，其对角元可以是复数，因此 $\operatorname{tr}(A)$ 也可以是复数。例如 $A=[i]$ 时，$\operatorname{tr}(A)=i$。若 $A$ 是 Hermitian（厄米）矩阵，即 $A^*=A$，则其特征值和迹都是实数；若 $A$ 还是半正定矩阵，则 $\operatorname{tr}(A)\ge0$。

当 $A\in\mathbb F^{m\times n}$、$B\in\mathbb F^{n\times m}$ 时，$AB$ 和 $BA$ 都是方阵，因此二者的迹都有意义。

$$
\operatorname{tr}(A+B)=\operatorname{tr}(A)+\operatorname{tr}(B),
$$

$$
\operatorname{tr}(cA)=c\operatorname{tr}(A),
\qquad
\operatorname{tr}(A)=\operatorname{tr}(A^T),
$$

$$
\operatorname{tr}(AB)=\operatorname{tr}(BA),
$$

$$
\operatorname{tr}(ABCD)
=\operatorname{tr}(BCDA)
=\operatorname{tr}(CDAB)
=\operatorname{tr}(DABC),
$$

这称为迹的 **cyclic（循环）不变性**：矩阵乘积可循环移位，但通常不能任意交换相邻因子。

$$
\operatorname{tr}(A\otimes B)
=\operatorname{tr}(A)\operatorname{tr}(B),
$$

其中 $\otimes$ 是 **Kronecker 积**。若 $A=(a_{ij})$，则

$$
A\otimes B=
\begin{bmatrix}
a_{11}B&\cdots&a_{1n}B\\
\vdots&\ddots&\vdots\\
a_{m1}B&\cdots&a_{mn}B
\end{bmatrix}.
$$

$$
\operatorname{tr}(P^{-1}AP)=\operatorname{tr}(A),
$$

这称为 **similarity invariance（相似不变性）**。

$$
\operatorname{tr}(A^k)=\sum_i\lambda_i^k.
$$

其中 $\lambda_i$ 表示 $A$ 的第 $i$ 个特征值，并按代数重数重复计算。

Frobenius 内积与向量化：

$$
\operatorname{tr}(A^TB)=\sum_{i,j}A_{ij}B_{ij}
=\operatorname{vec}(A)^T\operatorname{vec}(B).
$$

对同型实矩阵 $A,B$，还有

$$
\operatorname{tr}(A^TB)=\operatorname{tr}(AB^T).
$$

正半定矩阵 $A,B$ 还满足

$$
0\le\operatorname{tr}(AB)\le\operatorname{tr}(A)\operatorname{tr}(B).
$$

若 $A,B$ 为同阶实对称半正定矩阵，则 Lieb-Thirring 不等式的一个特例为

$$
\operatorname{tr}(ABAB)\le\operatorname{tr}(A^2B^2).
$$

对实对称 $A,B$，还可直接看到

$$
\operatorname{tr}(A^2B^2)-\operatorname{tr}(ABAB)
=\frac12\|AB-BA\|_F^2\ge0.
$$

### 3.6 范数、像空间、零空间与秩

矩阵 $A\in\mathbb F^{m\times n}$ 的像空间（列空间）与零空间分别定义为

$$
\operatorname{im}(A)=\{Ax:x\in\mathbb F^n\}\subseteq\mathbb F^m,
$$

$$
\operatorname{Null}(A)=\ker(A)
=\{x\in\mathbb F^n:Ax=0\}.
$$

$$
\|AB\|_F\le\|A\|_F\|B\|_2,
\qquad
\|AB\|_F\le\|A\|_2\|B\|_F.
$$

$$
\operatorname{im}(A)=\operatorname{im}(AA^T),
$$

$$
\operatorname{Null}(A)=\operatorname{Null}(A^TA).
$$

$$
A^TA\text{ 可逆}\iff A\text{ 满列秩}.
$$

由此可得一个常用推论：若 $A\in\mathbb R^{m\times r}$ 和 $B\in\mathbb R^{n\times r}$ 都满列秩，则

$$
\operatorname{rank}(A)=\operatorname{rank}(B)
=\operatorname{rank}(AB^T)=r.
$$

例如，$A^TA$ 与 $B^TB$ 可逆；结合秩不等式或左逆、右逆即可证明 $AB^T$ 的秩为 $r$。

$$
\operatorname{rank}(AB)
\le\min\{\operatorname{rank}(A),\operatorname{rank}(B)\}.
$$

矩阵乘法不会增加矩阵的秩：$AB$ 的列空间包含于 $A$ 的列空间，其行空间维数也不超过 $B$ 的秩。

### 3.7 谱范数与其他公式

谱范数为

$$
\|A\|_2=\sqrt{\lambda_{\max}(A^*A)}=\sigma_{\max}(A).
$$

在 $A$ 对称且 $\|v\|=1$ 时，

$$
(v^TAv)^2\le v^TA^2v.
$$

方阵乘积的行列式：

$$
\det(AB)=\det(A)\det(B).
$$

### 3.8 投影矩阵

投影矩阵既可以是实矩阵，也可以是复矩阵。若 $P\in\mathbb F^{n\times n}$（$\mathbb F=\mathbb R$ 或 $\mathbb C$）满足

$$
P^2=P,
$$

则 $P$ 是投影矩阵。它把任意向量投影到 $\operatorname{im}(P)$，并满足

$$
\operatorname{Null}(P)=\operatorname{im}(I-P).
$$

投影矩阵的特征值只能是 $0$ 或 $1$，因此

$$
\operatorname{rank}(P)=\operatorname{tr}(P).
$$

一般的 $P^2=P$ 称为投影，但不一定是正交投影。若进一步有

$$
P^*=P,
$$

则 $P$ 是正交投影矩阵。实数情形下 $P^*=P^T$；复数情形下 $P^*=\overline{P}^{T}$。此时

$$
\operatorname{im}(P)\perp\operatorname{Null}(P),
\qquad
P\succeq0,
\qquad
I-P\succeq0,
$$

并且

$$
\|Px\|_2\le\|x\|_2,
\qquad
\|x\|_2^2=\|Px\|_2^2+\|(I-P)x\|_2^2.
$$

若 $A\in\mathbb F^{m\times n}$ 满列秩，则到 $\operatorname{im}(A)$ 的正交投影为

$$
P_A=A(A^*A)^{-1}A^*.
$$

在实数情形下，这就是 $P_A=A(A^TA)^{-1}A^T$。

一般情形可用 Moore-Penrose 伪逆写成

$$
P_A=AA^\dagger.
$$

---

## 4. 概率论

### Bayes 公式、KL divergence 与 log trick

#### Bayes 公式

若 $\mathbb P(B)>0$，则

$$
\mathbb P(A\mid B)
=\frac{\mathbb P(B\mid A)\mathbb P(A)}{\mathbb P(B)}.
$$

若 $\{A_i\}$ 构成样本空间的一个划分，则全概率公式和离散 Bayes 公式为

$$
\mathbb P(B)=\sum_i\mathbb P(B\mid A_i)\mathbb P(A_i),
$$

$$
\mathbb P(A_i\mid B)
=
\frac{\mathbb P(B\mid A_i)\mathbb P(A_i)}
{\sum_j\mathbb P(B\mid A_j)\mathbb P(A_j)}.
$$

密度形式通常写成

$$
p(\theta\mid x)
=\frac{p(x\mid\theta)p(\theta)}{p(x)}
\propto p(x\mid\theta)p(\theta),
$$

即“后验 $\propto$ 似然 $\times$ 先验”。

#### KL divergence

若概率测度 $P$ 关于 $Q$ 绝对连续，则

$$
\mathrm{KL}(P\|Q)
=\int\log\!\left(\frac{dP}{dQ}\right)dP.
$$

离散形式为

$$
\mathrm{KL}(P\|Q)=\sum_xp(x)\log\frac{p(x)}{q(x)},
$$

连续密度形式为

$$
\mathrm{KL}(P\|Q)=\int p(x)\log\frac{p(x)}{q(x)}\,dx.
$$

KL divergence 满足 $\mathrm{KL}(P\|Q)\ge0$，等号成立当且仅当 $P=Q$（几乎处处），但它通常不对称，也不满足三角不等式，因此不是距离。

#### Log trick（score-function trick）

对可微的正密度 $p_\theta(x)$，

$$
\nabla_\theta p_\theta(x)
=p_\theta(x)\nabla_\theta\log p_\theta(x).
$$

在可交换求导与积分的条件下，若 $h$ 不显含 $\theta$，则

$$
\nabla_\theta\mathbb E_{X\sim p_\theta}[h(X)]
=
\mathbb E_{X\sim p_\theta}
[h(X)\nabla_\theta\log p_\theta(X)].
$$

同时 score 的期望为零：

$$
\mathbb E_{p_\theta}[\nabla_\theta\log p_\theta(X)]
=\int\nabla_\theta p_\theta(x)\,dx
=\nabla_\theta1=0.
$$

#### 用途

Log trick 将概率分布对参数的导数改写为关于该分布的期望。只要能够从 $p_\theta$ 采样，就可以用 Monte Carlo 方法构造梯度估计：

$$
\nabla_\theta\mathbb E_{p_\theta}[h(X)]
\approx
\frac1N\sum_{i=1}^N
h(X_i)\nabla_\theta\log p_\theta(X_i),
\qquad X_i\sim p_\theta.
$$

它特别适合样本不可直接对参数求导的场景，例如离散随机变量、REINFORCE/策略梯度和含离散潜变量的变分推断。代价是这种估计的方差往往较大。由于 score 的期望为零，可以减去不依赖于当前样本动作的 baseline $b$ 而不改变期望：

$$
\mathbb E[(h(X)-b)\nabla_\theta\log p_\theta(X)]
=
\mathbb E[h(X)\nabla_\theta\log p_\theta(X)],
$$

从而选择合适的 $b$ 来降低方差。上述恒等式通常要求 $p_\theta$ 的支撑不随 $\theta$ 改变，并满足足以交换求导与积分的正则性条件。

### 4.1 全期望公式

$$
\mathbb E[X]=\mathbb E[\mathbb E[X\mid Y]].
$$

### 4.2 全方差公式

$$
\operatorname{Var}(Y)
=\mathbb E[\operatorname{Var}(Y\mid X)]
+\operatorname{Var}(\mathbb E[Y\mid X]).
$$

### 4.3 Markov 不等式

若 $X$ 为非负随机变量且 $a>0$，则

$$
\mathbb P(X\ge a)\le\frac{\mathbb E[X]}a.
$$

等价地，对任意 $\tilde a>0$，

$$
\mathbb P(X\ge\tilde a\,\mathbb E[X])\le\frac1{\tilde a}.
$$

### 4.4 Chebyshev 不等式与大数定律

对任意 $a>0$，

$$
\mathbb P(|Z-\mathbb E Z|\ge a)
\le\frac{\operatorname{Var}(Z)}{a^2}.
$$

若 $X_1,\dots,X_n$ 独立同分布，$\bar X=n^{-1}\sum_{i=1}^nX_i$，则

$$
\mathbb P(|\bar X-\mathbb E X_1|\ge\varepsilon)
\le\frac{\operatorname{Var}(X_1)}{n\varepsilon^2}.
$$

**弱大数定律（WLLN）。** 若 $X_1,X_2,\dots$ 独立同分布，且 $\mathbb E[X_1]=\mu$、$\operatorname{Var}(X_1)<\infty$，则

$$
\bar X_n=\frac1n\sum_{i=1}^nX_i
\xrightarrow{\mathbb P}\mu,
$$

即对任意 $\varepsilon>0$，

$$
\mathbb P(|\bar X_n-\mu|>\varepsilon)\to0.
$$

Chebyshev 界可直接推出这一结论。

**强大数定律（SLLN）。** 若 $X_1,X_2,\dots$ 独立同分布且 $\mathbb E|X_1|<\infty$，则

$$
\bar X_n\xrightarrow{\mathrm{a.s.}}\mu,
$$

即

$$
\mathbb P\!\left(\lim_{n\to\infty}\bar X_n=\mu\right)=1.
$$

强大数定律的“几乎必然收敛”强于弱大数定律的“依概率收敛”。

### 4.5 Hoeffding 不等式

令 $Z_1,\dots,Z_m$ 为独立同分布随机变量，$\mathbb E[Z_i]=\mu$，并且 $\mathbb P(a\le Z_i\le b)=1$。则对任意 $\varepsilon>0$，

$$
\mathbb P\!\left(
\left|\frac1m\sum_{i=1}^m Z_i-\mu\right|>\varepsilon
\right)
\le
2\exp\!\left(-\frac{2m\varepsilon^2}{(b-a)^2}\right).
$$

### 4.6 Jensen 不等式

设随机变量 $X$ 的取值落在凸函数 $\varphi$ 的定义域内。若 $X$ 可积，即 $\mathbb E|X|<\infty$，并且 $\mathbb E[|\varphi(X)|]<\infty$，则

$$
\mathbb E[\varphi(X)]\ge\varphi(\mathbb E[X]).
$$

更一般地，也可在扩展实数意义下允许 $\mathbb E[\varphi(X)]=+\infty$。Jensen 不等式不要求 $X$ 服从某种特定分布；关键是 $\mathbb E[X]$ 有定义、$X$ 几乎处处位于 $\varphi$ 的凸定义域中，并且相关期望有意义。若 $\varphi$ 是凹函数，不等号反向。

### 4.7 随机变量的 Hölder 不等式

设 $X,Y$ 定义在同一个概率空间上。若 $p,q>1$、$1/p+1/q=1$，并且

$$
\mathbb E|X|^p<\infty,
\qquad
\mathbb E|Y|^q<\infty,
$$

则

$$
\mathbb E[|XY|]\le\|X\|_p\|Y\|_q,
$$

其中

$$
\|X\|_r=(\mathbb E|X|^r)^{1/r},
\qquad
\|X\|_\infty=\inf\{m:\mathbb P(|X|>m)=0\}.
$$

该不等式不要求 $X,Y$ 独立、同分布，也不要求它们属于特定分布族；只需相应的 $p$ 阶矩和 $q$ 阶矩有限。端点情形 $p=1,q=\infty$ 同样成立。

### 4.8 Chernoff 界

设 $X_1,\dots,X_n$ 相互独立，$0\le X_i\le1$，令

$$
X=\sum_{i=1}^nX_i,
\qquad
\mu=\mathbb E[X].
$$

则对 $\varepsilon\ge0$，

$$
\mathbb P(X>(1+\varepsilon)\mu)
\le
\exp\!\left(-\frac{\varepsilon^2}{2+\varepsilon}\mu\right),
$$

$$
\mathbb P(X<(1-\varepsilon)\mu)
\le
\exp\!\left(-\frac{\varepsilon^2}{2}\mu\right).
$$

### 4.9 Azuma-Hoeffding 不等式

令 $(\mathcal F_k)$ 为信息随时间增长的 filtration（滤过），即 $\mathcal F_{k-1}\subseteq\mathcal F_k$。

- 若 $X_k$ 对 $\mathcal F_k$ 可测、$\mathbb E|X_k|<\infty$，且

  $$
  \mathbb E[X_k\mid\mathcal F_{k-1}]=X_{k-1},
  $$

  则 $(X_k)$ 是 **martingale（鞅）**。直观上，给定过去信息后，下一步的条件期望等于当前值，是“公平游戏”。
- 若

  $$
  \mathbb E[X_k\mid\mathcal F_{k-1}]\le X_{k-1},
  $$

  则 $(X_k)$ 是 **supermartingale（上鞅）**。直观上，它的条件期望不会上升。
- 若不等号反向，则称为 submartingale（下鞅）。

令 $\{X_k\}_{k=0}^N$ 为 martingale 或 supermartingale，且几乎处处满足

$$
|X_k-X_{k-1}|\le c_k.
$$

则

$$
\mathbb P(X_N-X_0\ge\varepsilon)
\le
\exp\!\left(-\frac{\varepsilon^2}{2\sum_{k=1}^Nc_k^2}\right).
$$

若 $\{X_k\}$ 为 martingale，则双侧界为

$$
\mathbb P(|X_N-X_0|\ge\varepsilon)
\le
2\exp\!\left(-\frac{\varepsilon^2}{2\sum_{k=1}^Nc_k^2}\right).
$$

### 4.10 向量值 Azuma-Hoeffding 不等式

令 $\varepsilon_k\in\mathbb R^d$ 为关于 $\mathcal F_k$ 的向量值 martingale difference sequence，满足

$$
\mathbb E[\varepsilon_k\mid\mathcal F_{k-1}]=0,
\qquad
\|\varepsilon_k\|^2\le B_k^2.
$$

则对任意 $\lambda>0$，

$$
\mathbb P\!\left(\left\|\sum_{k=1}^K\varepsilon_k\right\|^2>\lambda^2\right)
\le
4\exp\!\left(-\frac{\lambda^2}{4\sum_{k=1}^KB_k^2}\right).
$$

---

## 5. Metropolis-Hastings 算法

目标：从只知道未归一化密度 $p(x)\propto f(x)$ 的分布中采样。由于接受率只使用密度比，未知的归一化常数会自动消掉。

### 5.1 算法步骤

1. 初始化 $x_0$，并选择提议分布 $Q(x'\mid x)$。
2. 对 $t=0,\dots,T-1$：
   1. 从 $Q(x'\mid x_t)$ 采样候选状态 $x'$；
   2. 计算接受概率

      $$
      \alpha(x_t,x')=
      \min\left\{
      1,
      \frac{f(x')Q(x_t\mid x')}{f(x_t)Q(x'\mid x_t)}
      \right\};
      $$

   3. 以概率 $\alpha$ 令 $x_{t+1}=x'$，否则令 $x_{t+1}=x_t$。

当 $Q(x'\mid x)=Q(x\mid x')$ 时，接受概率可简化为

$$
\alpha(x_t,x')=\min\left\{1,\frac{f(x')}{f(x_t)}\right\}.
$$

一个典型特例是 **Gaussian random-walk proposal**：

$$
x'\mid x_t\sim\mathcal N(x_t,\sigma^2I).
$$

它是以当前状态 $x_t$ 为中心的高斯分布。因为

$$
Q(x'\mid x_t)=Q(x_t\mid x'),
$$

所以接受率只剩目标密度比。$\sigma$ 控制步长：过小会导致移动缓慢，过大则可能导致接受率过低。

### 5.2 推导要点

设 Markov 链的转移核为 $P(x'\mid x)$。若分布 $\pi$ 满足

$$
\pi(x')=\int\pi(x)P(x'\mid x)\,dx,
$$

则 $\pi$ 是 stationary distribution（平稳分布）。构造目标是让给定的 $\pi$ 成为该链的平稳分布。

一个充分条件是 detailed balance（详细平衡）：

$$
\pi(x)P(x'\mid x)=\pi(x')P(x\mid x').
$$

它表示平稳状态下从 $x$ 流向 $x'$ 的概率流与反向概率流相同。对上式关于 $x$ 积分即可验证 $\pi$ 的平稳性。

Metropolis-Hastings 先用 $Q$ 提议 $x'$，再以 $A(x,x')$ 接受，因此对 $x'\ne x$，

$$
P(x'\mid x)=Q(x'\mid x)A(x,x'),
$$

而拒绝提议时留在原状态的概率补足转移核。详细平衡要求

$$
\pi(x)Q(x'\mid x)A(x,x')
=
\pi(x')Q(x\mid x')A(x',x).
$$

因此只需选择 $A$ 使

$$
\frac{A(x,x')}{A(x',x)}
=
\frac{\pi(x')Q(x\mid x')}{\pi(x)Q(x'\mid x)}.
$$

记

$$
r(x,x')=
\frac{\pi(x')Q(x\mid x')}{\pi(x)Q(x'\mid x)}.
$$

取

$$
A(x,x')=
\min\{1,r(x,x')\}
$$

即可满足详细平衡，因为 $r(x',x)=1/r(x,x')$：

- 若 $r\le1$，则 $A(x,x')=r$、$A(x',x)=1$；
- 若 $r>1$，则 $A(x,x')=1$、$A(x',x)=1/r$。

两种情形都有 $A(x,x')/A(x',x)=r$。又因为 $\pi(x)=f(x)/Z$，所以

$$
\frac{\pi(x')}{\pi(x)}=\frac{f(x')}{f(x)},
$$

未知归一化常数 $Z$ 被约掉。

详细平衡保证 $\pi$ 是平稳分布；要使链从一般初始状态收敛到唯一的 $\pi$，通常还需要不可约、非周期，并满足适当的正返性条件。

---

## 6. 最优传输

### 6.1 Kantorovich 问题

给定测度 $\alpha,\beta$ 和代价函数 $c(x,y)$，Kantorovich 最优传输问题为

$$
\operatorname{OT}_c(\alpha,\beta)
\equiv\mathcal L_c(\alpha,\beta)
=
\min_{\pi\in U(\alpha,\beta)}
\int_{\mathcal X\times\mathcal Y}c(x,y)\,d\pi(x,y),
$$

其中耦合集合

$$
U(\alpha,\beta)
=
\left\{
\pi\in\mathcal M_+^1(\mathcal X\times\mathcal Y):
P_{\mathcal X\#}\pi=\alpha,
\;P_{\mathcal Y\#}\pi=\beta
\right\}.
$$

$P_{\mathcal X}$、$P_{\mathcal Y}$ 分别表示到两个分量的投影，$P_{\mathcal X\#}\pi$、$P_{\mathcal Y\#}\pi$ 是对应的 push-forward。

### 6.2 Push-forward 测度

给定可测映射 $T:\mathcal X\to\mathcal Y$，若 $\beta=T_\#\alpha$，则对任意可测函数 $h$，

$$
\int_{\mathcal Y}h(y)\,d\beta(y)
=
\int_{\mathcal X}h(T(x))\,d\alpha(x).
$$

等价地，对任意可测集 $B\subseteq\mathcal Y$，

$$
\beta(B)=\alpha(T^{-1}(B)).
$$

### 6.3 Kantorovich 对偶问题

为避免记号歧义，以下把 Kantorovich 最优值记为 $\operatorname{OT}_c(\alpha,\beta)$。

$$
\operatorname{OT}_c(\alpha,\beta)
=
\sup_{(f,g)\in\mathcal R_c}
\left\{
\int_{\mathcal X}f\,d\alpha+
\int_{\mathcal Y}g\,d\beta
\right\},
$$

其中

$$
\mathcal R_c=
\left\{
(f,g)\in C(\mathcal X)\times C(\mathcal Y):
f(x)+g(y)\le c(x,y),\;\forall(x,y)
\right\}.
$$

在紧致空间和连续代价等常用假设下，可取 $f\in C(\mathcal X)$、$g\in C(\mathcal Y)$。对任意 $\pi\in U(\alpha,\beta)$，

$$
\int f\,d\alpha+\int g\,d\beta
=\int(f(x)+g(y))\,d\pi(x,y)
\le\int c(x,y)\,d\pi(x,y),
$$

这说明了弱对偶；在标准正则条件下等号成立。

### 6.4 熵正则化最优传输

定义熵正则化 OT 值（$\varepsilon>0$）为

$$
\operatorname{OT}_{c,\varepsilon}(\alpha,\beta)
=
\min_{\pi\in U(\alpha,\beta)}
\left\{
\int_{\mathcal X\times\mathcal Y}c(x,y)\,d\pi(x,y)
+\varepsilon\,\mathrm{KL}(\pi\mid\alpha\otimes\beta)
\right\}.
$$

其中 $\alpha\otimes\beta$ 是乘积测度。对有限正测度，KL 散度写为

$$
\mathrm{KL}(\pi\mid\xi)
=
\int_{\mathcal X\times\mathcal Y}
\log\!\left(\frac{d\pi}{d\xi}(x,y)\right)d\pi(x,y)
+\xi(\mathcal X\times\mathcal Y)
-\pi(\mathcal X\times\mathcal Y).
$$

若 $\pi$ 不关于 $\xi$ 绝对连续，则约定 $\mathrm{KL}(\pi\mid\xi)=+\infty$。

其对偶形式为

$$
\operatorname{OT}_{c,\varepsilon}(\alpha,\beta)
=
\sup_{f,g}
\left\{
\int_{\mathcal X}f(x)\,d\alpha(x)
+\int_{\mathcal Y}g(y)\,d\beta(y)
-\varepsilon
\int_{\mathcal X\times\mathcal Y}
\exp\!\left(
\frac{f(x)+g(y)-c(x,y)}{\varepsilon}
\right)
d\alpha(x)\,d\beta(y)
+\varepsilon
\right\}.
$$

末项 $+\varepsilon$ 来自上述 KL 归一化方式。若将该常数吸收到正则化目标中，对偶式可省略 $+\varepsilon$；两种写法得到的最优耦合相同。

#### 关于参考测度的注意事项

若 $\pi\in U(\alpha,\beta)$，并且 $(\alpha',\beta')$ 与 $(\alpha,\beta)$ 分别具有相同的支撑，使相应测度彼此有密度，则

$$
\mathrm{KL}(\pi\mid\alpha\otimes\beta)
=
\mathrm{KL}(\pi\mid\alpha'\otimes\beta')
+\mathrm{KL}(\alpha'\otimes\beta'\mid\alpha\otimes\beta).
$$

右边第二项与 $\pi$ 无关，所以用 $\alpha'\otimes\beta'$ 代替 $\alpha\otimes\beta$ 作为 KL 的参考测度，不会改变最优耦合，只会使最优值平移一个常数。参见 Peyré 与 Cuturi 的 [*Computational Optimal Transport*, Proposition 4.2](https://optimaltransport.github.io/pdf/ComputationalOT.pdf)。

### 6.5 Sinkhorn divergence

记 $\operatorname{OT}_{c,\varepsilon}$ 为第 6.4 节定义的熵正则化最优传输值。Sinkhorn divergence 用于校正其自耦合代价（self-cost）偏差：

$$
S_{c,\varepsilon}(\alpha,\beta)
=
\operatorname{OT}_{c,\varepsilon}(\alpha,\beta)
-\frac12\operatorname{OT}_{c,\varepsilon}(\alpha,\alpha)
-\frac12\operatorname{OT}_{c,\varepsilon}(\beta,\beta).
$$

当 $c(x,y)=d(x,y)^p$ 时，定义正则化的 Wasserstein 型代价

$$
W_{p,\varepsilon}(\alpha,\beta)
=\operatorname{OT}_{d^p,\varepsilon}(\alpha,\beta)^{1/p}.
$$

它一般不严格满足距离公理。相应的去偏版本可定义为

$$
\widetilde W_{p,\varepsilon}(\alpha,\beta)
=S_{d^p,\varepsilon}(\alpha,\beta)^{1/p}.
$$

有些文献将 $\operatorname{OT}_{d^p,\varepsilon}$ 直接记为 $W_{p,\varepsilon}$。在这种记号下，减去两个 self-cost 的表达式就是 $S_{d^p,\varepsilon}$。使用 $W_{p,\varepsilon}$ 时，需要先确认其定义是否包含 $p$ 次方根。

### 6.6 Wasserstein 距离

令 $(\mathcal X,d)$ 为度量空间，$p\ge1$，并取

$$
c(x,y)=d(x,y)^p.
$$

则 $p$ 阶 Wasserstein 距离为

$$
W_p(\alpha,\beta)
=
\operatorname{OT}_{d^p}(\alpha,\beta)^{1/p}
=
\left(
\min_{\pi\in U(\alpha,\beta)}
\int_{\mathcal X\times\mathcal X}d(x,y)^p\,d\pi(x,y)
\right)^{1/p}.
$$

更一般地，对 $1\le q\le p$，有

$$
W_q(\alpha,\beta)\le W_p(\alpha,\beta).
$$

**证明。** 对任意耦合 $\pi\in U(\alpha,\beta)$，在概率空间 $(\mathcal X\times\mathcal X,\pi)$ 上对非负随机变量 $D=d(X,Y)$ 使用 Jensen 不等式或 $L^p$ 范数的单调性：

$$
\left(\int D^q\,d\pi\right)^{1/q}
\le
\left(\int D^p\,d\pi\right)^{1/p}.
$$

令 $\pi_p^*$ 是 $W_p$ 的最优耦合，则

$$
\begin{aligned}
W_q(\alpha,\beta)
&=\inf_{\pi\in U(\alpha,\beta)}
\left(\int d(x,y)^q\,d\pi\right)^{1/q}\\
&\le
\left(\int d(x,y)^q\,d\pi_p^*\right)^{1/q}\\
&\le
\left(\int d(x,y)^p\,d\pi_p^*\right)^{1/p}\\
&=W_p(\alpha,\beta).
\end{aligned}
$$

取 $q=1,p=2$，可得

$$
W_1(\alpha,\beta)\le W_2(\alpha,\beta).
$$

本节的定义与正则化形式参考 Gabriel Peyré 和 Marco Cuturi 的 [*Computational Optimal Transport*](https://optimaltransport.github.io/book/)。
