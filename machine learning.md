1. ### introduction


> **Pre-train**:提前训练一些能够处理初级任务的模型 进而”通性“的downstream task
>
> 提前训练的模型和实际使用的模型 类似于Operating system 和Applications
>
> 其中比较出名的Pre-train的模型 BERT
>
> 
>
> **Generative Adversarial Network**:生成式对抗网络 作用是给出一大堆X性质样本 和Y性质样本 可以自动找到之间的Function
>
> 
>
> **Reinforcement Learning**:强化学习 用于无法给出样本 无法标注资料 但是可以标注好坏时 可以标注成功时 比如下棋 人也不知道下载哪最好 就可以使用强化学习 因为下赢总是好的 总是可以确定的
>
> 
>
> **Anomaly Detection**:异常检测 允许AI不告知答案
>
> 
>
> **Explainable** AI:可解释性AI
>
> 
>
> **Model Attack**:本质上是对样本差异化导致model无法辨识样本
>
> 
>
> **Domain Adaptation**:领域适应性建模
>
> 
>
> **Network Compression**:模型压缩
>
> 
>
> **Life-long Learning**:
>
> 
>
> **Meta Learning**:元学习 Few-shot learning 通过很少样本就可以进行学习

1. ### 机器学习基本概念


##### Machine Learning equals Looking for Function



**Different types of Function**

> Regression:The function outputs a scalar(标量)
>
> Classification:Given options(classes),the function outputs the correct one/ones 
>
> Structure Learn: 创造具有结构的事物 比如文章 

**Function brief introduction**

> 1.**Y=f(Xs)**
>
> predict:y=b+w$x_{1}$ based on domain knowledge 
>
> $x_{1}$->feature b->bias w->weight
>
> 
>
> 2.**next Define Loss from Train Data**
>
> 1. Loss if a function of parameter
>
> 2. to evaluate how good a set of value is
>
>    根据上一天的订阅量推测明天的订阅量
>
>    feature设定成上一天的订阅量 y设为明天的订阅值
>
>    Function假设为0.5k+$1x_{1}$ Loss函数即预测误差和的平均值 $L=\frac{1}{N} \sum_n{e_{n}}$
>
>    两种计算误差的方式 
>    $$
>    e=|y-\hat{y}|\ mean\ absolte\ error(MAE)\\
>    e=(y=\hat{y})^2\ mean\ square\ err(MSE)
>    $$
>
>
>    <img src="C:\Users\Lenovo\Desktop\Typora\pictures\3cb2d42fb8b00a18f2ee8f1c1c247a4.jpg" alt="3cb2d42fb8b00a18f2ee8f1c1c247a4" style="zoom: 15%;" />
>
>    如上图 使用真实数据计算loss 画出的等高线 色系越暖计算出的loss越大 这个图也叫做 Error Surface
>
> 3.**Optimization**
>
> 1. 使用Gradient Descent构建Linear Model(梯度下降 迭代优化算法 用于寻找可微函数的局部最小值) 以下是一元的例子
>
>    1. (Randomly) Pick up an initial value $w^{0}$
>
>    2. Computr $\frac{\partial L}{\partial w}|_{w=w{0}}$​
>
>    3. 根据斜率进行w的变化update w iteratively $w^{1}=w^{0}-\eta\frac{\partial L}{\partial w}|_{w=w{0}}$  其中$\eta 叫做hyperparameter(人为设定的参数\ 超参数))$​
>
>       上述三步就叫做training
>
>       the way to stop:1.set the total steps
>
>       ​			     2.找到局部最小点
>
>       <img src="C:\Users\Lenovo\Desktop\Typora\pictures\cd9735f30a4171bf003d3899e637f66.png" alt="cd9735f30a4171bf003d3899e637f66" style="zoom:15%;" />
>
>       通过这张图1. loss是人为设定的函数 不一定需要是MAE MSE 2. Gradient Descent 存在漏洞 可能找到局部最小点(local minima) 而不是最小点(global minima)
>
> 2. 二元实例
>
>    分别对单个进行考虑 再取两者Gradient Descent计算后得到的值
>
>    这一部分得到的是单输入 多参数的模型
>
> 3. 添加多个输入
>
>    <img src="C:\Users\Lenovo\Desktop\Typora\pictures\4b865ac98c84208e71a5a0b2d6f874d.png" alt="4b865ac98c84208e71a5a0b2d6f874d" style="zoom:15%;" />
>
>    红色是实际数据 蓝色是预测数据 可以看到预测值几乎和前一天值相差无几 并且数据的变化存在类似的周期性 7天之内均经历一个先上升再下降的结果 所以希望添加前7天的值作为输入(而对模型的修改依靠的就是对模型的理解 domain knowledge)
>
>    $y=b+\sum_{j=1}^{7} w_{j}x_{j}$
>
> 
>
>    考虑天数越多 $L^{\prime}$​越接近临界值
>
> 
>
>    The type of those models are called lineal model, we need more sophisticated and flexible modes

**elaborate Function**

> 1. **define a Function**
>
> 对于更加复杂的模型 可能存在多段线性模型叠加的情况 x在不同段可能存在不同的线性模型
>
> <img src="C:\Users\Lenovo\Desktop\Typora\pictures\11dab190f4c345c8a64cc8e32e94c15.png" alt="11dab190f4c345c8a64cc8e32e94c15" style="zoom:15%;" />
>
> 这个方法对于表示piecewise line很有优势 你会发现 加上Heaviside函数后 原函数就类似于转折了 
>
> 
>
> 折线段越多 折现越接近曲线(Approximate continuous curve by a piecewise linear curve) 
>
> 现在的问题是怎么确定转折点？ 怎么写出Heaviside(hard sigmoid)函数？(均通过训练得来)
>
> <img src="C:\Users\Lenovo\Desktop\Typora\pictures\96f03881ea089dd4a855c5877707e5f.png" alt="96f03881ea089dd4a855c5877707e5f" style="zoom:15%;" />
>
> 改变w 改变阶跃斜率(w和实际斜率方向相反)
>
>  改变b 曲线平移(提括号 b影响x)
>
>  改变c 曲线上下值改变
>
> 代码中写作  $c_{i}\ sigmoid(b_{i}+w_{i}x_{1})$
>
> $y=b+\sum_{i} c_{i}\ sigmoid(b_{i}+\sum_{j} w_{ij}x_{1})$ 这里注意 不是对每一个xi求sigmoid 而是对整体使用sigmoid 也就是整体为自变量进行多段累加​
>
> $ 由xi得到ri\ ri=bi+\sum xi$
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/b33547d5630b242981d1df0f11f6a91-1724418490373-12.png" alt="b33547d5630b242981d1df0f11f6a91" style="zoom:15%;" />
>
> ​		使用向量来展示$ 由xi表示ri$
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/be1d7355990b9dde5205e10592c822f.png" alt="be1d7355990b9dde5205e10592c822f" style="zoom:15%;" />
>
> ​		使用$r_{i}和sigmoid表示a_{i}$
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/4cb01a34c6b6219050ae04e887111ba.png" alt="4cb01a34c6b6219050ae04e887111ba" style="zoom:15%;" />
>
> ​		使用$a_{i}来表示y$
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/d7312726b740cfff576530db613917a.png" alt="d7312726b740cfff576530db613917a" style="zoom:15%;" />
>
> ​		最终矩阵形式
>
> ​		$y=b+C^{T}\delta(B+WX)$​
>
> ​		<img src="C:/Users/Lenovo/Desktop/Typora/pictures/aeb8a045450ad890586261f054c4f61.png" alt="aeb8a045450ad890586261f054c4f61" style="zoom:15%;" />
>
> ​		综上 实际的抽象存在两层 从多输入抽象成单输入 单输入抽象成		digmoid叠加
>
> 2. **determine loss**
>
>    使用$\boldsymbol{\theta}来抽象各种参数值的集合 \boldsymbol{\theta}|= \begin{bmatrix}\theta1\\ \theta2\\\theta3\\... \end{bmatrix}(粗体代表集合 \ 浅体代表参数)(参数列表应该是W\ 或B 不一定是列列表 但是这里将所有参数写成一列不影响计算梯度)$
>
>    $\boldsymbol{\theta}^{\star}(\star\ \ means\ \ optimal)=argminL$
>
>    (Randomly) Pick up initial values $\boldsymbol{\theta^{0}}$
>
>    $\boldsymbol g =\begin{bmatrix}\frac{\partial L}{\partial \theta_{1}}\\\frac{\partial L}{\partial \theta_{2}}\\ ...\end{bmatrix}$
>
>    $\boldsymbol g=\nabla L(\boldsymbol \theta^{0})$​
>
>    $\boldsymbol \theta^{1}=\boldsymbol \theta^{0}-\eta\ \nabla L(\boldsymbol \theta^{0})(上标表示不同集合 下标表示一个集合内不同index)(这里的意义是L对\boldsymbol \theta中的元素进行偏导 实际意义也是一个列向量)$    $\begin{bmatrix}\theta_{1}^{1}\\ \theta_{1}^{2} \\\theta_{1}^{3} \\ ...\end{bmatrix}=\begin{bmatrix}\theta_{0}^{1}\\ \theta_{0}^{2} \\\theta_{0}^{3} \\ ...\end{bmatrix}-\eta\begin{bmatrix}\frac{\partial L}{\partial \theta_{1}}|_{\boldsymbol\theta=\boldsymbol\theta^{0}}\\ \frac{\partial L}{\partial \theta_{2}}|_{\boldsymbol\theta=\boldsymbol\theta^{0}}\\\frac{\partial L}{\partial \theta_{3}}|_{\boldsymbol\theta=\boldsymbol\theta^{0}}\\ ...\end{bmatrix}$​
>
>    $\boldsymbol \theta^{2}=\boldsymbol \theta^{1}-\eta\ \nabla L(\boldsymbol \theta^{1})(上标表示不同集合 下标表示一个集合内不同index)$
>
>    $\boldsymbol \theta^{3}=\boldsymbol \theta^{2}-\eta\ \nabla L(\boldsymbol \theta^{2})(上标表示不同集合 下标表示一个集合内不同index)$
>
>    $...$
>
> 3. **Optimization of New Model**
>
>    实际上样本的使用存在各种变种
>
>    Batch Gradient Descent(批量梯度下降)当此的偏导值根据所有样本的偏导值求平均
>
>    Stochastic Gradient Descent(随机梯度下降) 当此的梯度只选取一个样本
>
>    Mini-Batch Gradient Descent(小批量梯度下降) 当此的梯度选取部分样本求梯度取平均值
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/f5c5573121d998eb3c05874552332f8.png" alt="f5c5573121d998eb3c05874552332f8" style="zoom:15%;" />
>
> 
>
>    epoch= see all batchs once
>
>    update: the process of calculating partial using samples 
>
> 
>
>    ​	*Example 1:*
>
>    *​		10000 parameters (N=10000)*
>
>    *​		10 parameters per batch(B=10)*
>
>    *​		There are 1000 times update in **1 epoch**(注意 batch的大小本身也是一个hyperparameter)*
>
> 4. **Activation function**
>
>    常用的两个激活函数 sigmoid 和ReLU(Rectified Linear Unit)
>
>    ReLU的实现 $cmax(0,b+wx_1)+c'min(0,b'+w'x_1)$ 或者 $constrain(0,max,b+wx_1)$​ constrain(min,0,b+wx_1)$
>
> 5. **layer**
>
>    从x->r->a->a'->a''->... a本身就是层
>
>    对多层的样本传递思想:只有最后一层才得到唯一a 也即只有这一层update的partial计算中才取平均 
>
>    每层均可以使用不同的梯度下降变种方法 使用Batch当前层输出的个数变多 但是最后一层一定使用的是Batch Gradient Descent
>
>    *Example*
>
>    *<img src="C:/Users/Lenovo/Desktop/Typora/pictures/397db17660e7a24b5650a8107f8c40b.png" alt="397db17660e7a24b5650a8107f8c40b" style="zoom:15%;" />*
>
>    *通过训练 模型就知道周期性质的预测 错误点是因为数据异常 这一天在过年 是独立于周期点外的一点*
>
>    Terms
>
>    ![25fd3a65eac21fea5a2836788dde005](C:/Users/Lenovo/Desktop/Typora/pictures/25fd3a65eac21fea5a2836788dde005.png)
>       hidden layer:隐藏层,神经网络输入层和输出层之间的神经元或者层
>
>   question:Why don't we go deeper
>
>   会出现Overfitting(过拟合)现象 也即随着层数的增加 在训练资料上的表现更好 但是预测上表现更差

1. ### Ups and downs of Deep Learning


> Perceptron(感知器)->perceptron has limitaion->Multi-layer perceptron->Backpropagation->RBM initialization->GPU->

1. ### 对神经网络前向通道和反向通道(forward propagation and backpropagation or forward pass and back pass)的理解

2. **为什么有前向和反向这个说法？**

> 前向是指通过神经网络计算输出 反向是指通过已知的输出计算神经网络(两个输入均已知)

2. **怎么训练神经网络？**

> 首先需要选定激活函数(sigmoid或者ReLU)
>
> 流程是：x使用线性和表示r_1(B和线性W)，r_1通过ReLU表示a_1(B和卷积C)，a_1再线性化表示r_2,r2通过ReLU表示a_2...，最后一层输出表示output
>
> **或者省去a 直接使用sigmoid处理r而非再对这个结果线性拟合作为传递**(一般按此处理)

3. **怎么求偏导 如果非线性各个参数之间不是相互影响吗？**

> 因为所有参数值均有initialization 所以计算某一个参数的偏导时 均带入其他参数的实值 并且输入也已知 输出也已知 L可以视作只含有目标变量的函数 自然可以求偏导计算

4. **各个层的参数和连接方式怎么确定？**

> 后一层的参数数目可以由前一层的weight和bias矩阵格式决定
>
> 连接方式通过x->r a->r 这一步线性化来决定怎样使用上一步的数据

5. **为什么要使用激活函数来引入a 只使用r不可以吗 引入激活函数有什么用？**

> 不使用激活函数 整个计算本质还是线性计算 只有使用激活函数才可以引入非线性

1. ### Multi-Class classification，Softmax formula and sigmoid


**Multi-Class classification**

> Definition：是机器学习中的一种学习任务，用于将输入数据划分为三个或者更多个类别 比如在图像处理中可能将图像分类成"猫，狗，鸟"等等类别

**Softmax**

> Definition：和sigmoid一样 可以当作激活函数(将输出进行非线性变换) 用于多分类问题的激活函数 可以用于将输出转化为概率分布  即将神经网络输出的最终层(实数向量)转化成一个概率向量 其中每个元素的值都在0-1之间，并且所有元素的和为1 
>
> formula：$ \Large \frac{e^{z_i}}{\sum^n_{j=1}e^{z_i}} $(其中z_i是第i个类别的未归一化输出值 n是类别的总数)

1. ### The determination of the “best” structure


> Q: How many layers？ How many neurons for each layer
>
> ---Trial and err+Intuition.
>
> Q: Can the structure be automatically determined?
>
> --- Yes, but not regular.

1. ### Elaborate on Backpropagation


> 1. Chain Rule
>    $$
>    Case1:\ \ y=g(x)\ \ z=h(y)  \\\Delta x->\Delta y->\Delta z\ 则 \frac{dz}{dx}=\frac{z}{y}\frac{dy}{dx}
>    \\Case2:\ \ x=g(s)\ \ y=h(s)\ \ z=k(x,y)
>    \\\frac{dz}{ds}=\frac{\delta z}{\delta x}\frac{dx}{ds}+\frac{\delta z}{\delta y}\frac{dy}{ds}
>    $$
>
> 2. Compute
>
>    假设数据的传输为 $\large x^n(sample)\xrightarrow{}NN(\boldsymbol \theta\ \ hidden layer)\xrightarrow{}y^n(output)\xleftrightarrow{c^n}\hat y^n(概率预测量 经过Softmax函数归一化)$
>
>    $L(\theta)=\sum^n_{n=1}c^n(\theta)(Loss单一样本计算损失\ \ Cost对所有样本计算损失)$​
>
> 
>
>    **核心思想是一层一层求偏导 引入后一层变量对前一层变量求偏导乘Cost函数对后一层变量求偏导 而后一层又可以视作前一层 往复循环 并且每往后一层 新的引入项数目就会变多(这个数目和前一层的流向数目相同) 而这时可以视作最后一层反过来一步步流向输入层 一言蔽之 就是将最远层和基础层偏导 化作后一层和基础层偏导乘最远层和后一层偏导 以此类推**
>
>    其中的一些细节：
>
>    (i)  如果没有经过激活函数 仅仅是x->r 这一步求偏导直接得到对应的w(这里需要解释 最开始一层求偏导确实是对w求偏导 而后使用链式法则求偏导后 中间出现的都是w了)
>
>    (ii)  对于sigmoid等激活函数求导 这里中间变量是可以计算出实值 对一般形式的sigmoid求导代入实值即可
>
>    (iii)  实际上的backpropagation 也会前向跑一遍节点 将所有节点的数记录下来 方便sigmoid计算倒数
>
>    (iiii)  最远层和当前层所有(指定方向)偏导相加得到下一层的最远层和当前层偏导 
>
>    (iiiii)  形象的理解 可以理解成分叉的树干 
>
>    (iiiiii) sigmoid导数在这里可以视作一个op-amp(运算放大器) 值可能是倒数？？？ 
>
> 通过以上步骤 将整体的求解偏导变成了树状的加和与经过运放的乘 避免了运算的重复(最易重复的计算就是输出层与前一层的偏导 这里最先计算 同理除了第一层偏导都会重复计算)
>
> 

1. ### 宝可梦cp分辨实例


> Step1:建立模型
>
> ​	1.线性回归 $y=b+wx_{cp}$
>
> Step2:Goodness of Function
>
> ​	2.定义Loss function(输入一个function 输出how bad it is)
>
> Step3:BackPropagation
>
> 
>
> 引入的问题：
>
> 1. 过拟合 模型的次数一步步上升 训练出的模型对Training Data有更好的结果 但是Testing Data的结果甚至更差 $\xrightarrow{原因}$数据量太少导致模型建立不精确
> 2. 会出现同一个cp值出现两条线 原因是物种也会影响进化的cp值 $\xrightarrow{原因}$​输入量不足以描述整个系统
> 3. 添加weight和height变量去描述模型之后 Loss增大了$\xrightarrow{原因}$机器学习再怎么样也是去推测实际的模型 如果建立模型那一步就极大的偏离实际模型 不能可能在Test Data上表现很好 即使很好的符合Training的数据 
>
> 对于问题二 李采用了分类判断类别使用不同模型的方式 但是我觉得可以将物种分类变成一个元组 将物种直接变成数带入模型
>
> 对于问题一 引入Regularization

1. ### 范数正则化(Norm Regularization)

   **Norm(范数)**


> Definition:通过向损失函数中添加惩罚项，限制模型参数的大小，从而迫使模型选择较小的参数，避免参数过大导致的过拟合
>
> Purpose:避免过拟合或者降低过拟合程度(达到同样的对训练集的预测效果 整体数据越小越好)
>
> Method:通过向惩罚函数内添加有关参数的项 使loss函数也考虑参数大小
>
> Classification:
>
> L1正则化(Lasso回归) 引入惩罚项$\lambda sum_i|\theta_i|(其中\lambda为权重)$
>
> L2正则化(Rideg回归) 引入惩罚项 $\lambda \sum_i\theta_i^2(其中\lambda为权重)$​
>
> Dropout 用于训练更新参数随机的让神经元的输出置0 即前向传播置0 反向传播输出0 但实际使用和评估时不做改变

**Regularization(正则化)**

> **details**
>
> 1. 通过Norm降低整体参数大小
>
>    如果过拟合 降低的最开始部分会提高训练数据的Loss 但是降低测试数据的Loss(曲线变平滑)
>
>    如果继续下降 两个训练集的Loss都会上升(数据不够强) 
>
> 2. 不需要考虑bias 因为其和曲线平衡无关

1. ### Validation(验证)


> 引入验证集(Validation set)的概念进行Validation
>
> Function:验证集用于在训练中评估模型的性能 在必要的步骤停止 防止过拟合
>
> details:
>
> 1.验证集的数据不进行反向传播更新
>
> 2.验证集对训练起作用 但是测试集是最终用来评估模型性能的 对模型没有作用
>
> 3.使用验证集的方式是将一份数据每次epoch都分成k-1 和1份 使用那一份作为验证集合(每次其他数据进行反向传播更新 但是对于验证集只前向传播验证Loss)

### 分类宝可梦实例

**课程跟随**

> 1.regression和classification的认识：均是"任务",前者输出"可"连续数据,后者输出理算数据---可连续的意思可以输出连续的数据 不像分类任务输出的数据是固定分隔的
>
> 2.引入问题:使用regression预测样本数据,期望同一类别输出同一值或者至少同一值附近 但是这个思想的致命点在于假如某些特别符合样本的值 会导致其特别偏离基准值 使用这些数据训练的效果判断边缘数据时会出现误判---根本原因是边缘数据导致Loss最小的情况并不能完全区分binary-class 
>
> 但是这个情况的模型拟合易理解 使用人为设定的数值(class对应)和计算结果计算Loss和梯度
>
> 3.引入一个理想模型 
>
> Function:f(x) 根据内含的g(x)的正负输出分类结果 内部的g(x)根据x(某一个对象的各种特征值)输出正负 注意 这里g(x)和第二点遇到的问题不同在于 这个函数g(x)不使用Loss的方法得到 可以将数据完全的区分
>
> 然后Loss设置为f(X)判断失败的次数
>
> 4.一些公式概念$P(x|c1)\ 类别1中有x特征的概率\ P(x|c2)类别二中有x特征的概率\ P(c1|x)有x特诊 是类别1的概率...\\那么 已知x特征 求其是c1类的概率公式为\ c1中有x的概率除以c1c2中有x概率的概率和P(c1|x)=\frac{P(x|c1)P(c1)}{P(x|c1)P(c1)+P(x|c2)P(c2)}$
>
> 5.generative model(用于概率函数的X是vector\ \mu也是vector\ \sum是covariance\ matrix(其为对角矩阵\ 非对角线元素均为协方差\\ \ 对角线上点为方差)\\ 引入L(代表Likelihood) L_{\mu,\sum}=f_{\mu,\sum}(X^1)f_{\mu,\sum}(X^2)...从给定的数据中生成新的数据样本) 找到方法得到$P(c1) P(C2) P(x|c1) P(X|C2)\ 就是找到方法搭建了一个generative\ model$ 现在存在的问题有1. P(c1)单纯从样本数去确定吗？ 2.在某个类下特征的概率怎么确定？
>
> 6.假设数据是按高斯分布(本步是在找特征在对应类下的概率) 对于多维数据 $概率函数的X是vector\ \mu也是vector\ \sum是covariance\ matrix(其为对角矩阵\ 非对角线元素均为协方差\\ \ 对角线上点为方差)\\ 引入L(代表Likelihood) L_{\mu,\sum}=f_{\mu,\sum}(X^1)f_{\mu,\sum}(X^2)...$
>
> 7.Maximum Likelihood(极大似然法) 数学推导后(求导求极值点 其实就是在做SGD在做的事情) 高斯公式中的变量$\mu和_sum其实就是数据的平均值和方差$​
>
> 8.经过上述的铺垫 由数据搭建特征在类别下的高斯概率模型(极大似然法 将点和概率联系) 进而可以得到类别在特征下的概率 只要类别在特征下的概率大于0.5 就认为其在此类中 那么在两个特征值的情景下 就可以在二维图中画很多"等高线",其中一个等高线代表c1在x下的概率是0.5 在线外是c1 线内是c2
>
> 9.扩展到高维 带入公式的$\mu和\sum都是高维量 但是带入x进入高斯f分布公式中计算得到的概率都是标量\ 在L内对\mu和\sum求极值就得到其实际意义（极大似然法）$
>
> 10.分别根据数据点和极大似然法得到相应的高斯分布 再得到P(c1|x) 至于为什么使用高斯分布以及高维下的高斯分布意义这些不需要考虑
>
> 直接使用维度的高斯分布进行判断效果不太好 只有50%(判断的过程是 多维x分别求解$\mu\ \sum$带入公式计算出高斯分布得到P(x|c1)同样的得到P(x|c1)再得到P(c1|x)进行曲线绘制以及类别判断)
>
> 11.引入两个类别高斯分布使用相同协方差来减少参数 怎么处理相同的协方差矩阵求其具体值呢？将两个数据的所有概率在一起写成新L 这个新L中P(x|c1)P(x|c2)有不同的均值和相同的协方差 使用极大似然法 求得$\sum=\frac{n1}{n1+n2}\sum_1+\frac{n2}{n1+n2}\sum_2$rz1
>
> 12.使用相同的协方差得到P(c1|x) 用其等于0.5得到判断线就从曲线变成直线 高维场景下判断效果变得更好
>
> 13.引入naive bayes classifier 也就是说协方差矩阵只考虑方差不考虑协方差 假设各个feature之间是independent
>
> 14.将高斯分布概率带入P(c1|x) 化简得到使用高斯分布估计概率的数学模型可以写成
>
> $P(c1|x) =\frac{1}{1-e^{-z}}\ z=\frac{P(c2|x)}{P(c1|x)}=WX+B$
>
> $继续化简\ 有\ P(c1|x)=\sigma(f(X))$
>
> **注意 只有假设两类数据拥有相同协方差的情景下 极大似然才可以化简成线性回归加sigmoid的形式**
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/bda41da47795575f1706d993e55940d.png" alt="bda41da47795575f1706d993e55940d" style="zoom:10%;" />
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/6439079091baca38f8545c1286c137e.png" alt="6439079091baca38f8545c1286c137e" style="zoom:10%;" />
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/833be940fc0baf1c9855c5660b2b4de.png" alt="833be940fc0baf1c9855c5660b2b4de" style="zoom:10%;" />
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/cd6063f3b4b3ac78e5514add3671cf3.png" alt="cd6063f3b4b3ac78e5514add3671cf3" style="zoom:10%;" />
>
> 15.使用机器学习方法求$\mu\ \sum$​而非数学方法
>
> 首先对原式取(-ln)方便计算 将乘项变成加项 将求最大值变为求最小值 $ -ln(L_{\mu,\sum})=-(ln（f_{\mu,\sum}(X^1))+ln(f_{\mu,\sum}(X^2))...)$
>
> 给出$ -ln(L_{\mu,\sum})=-(ln（f_{\mu,\sum}(X^1))+ln(f_{\mu,\sum}(X^2))...)$式后 需要带入使用w b表达的func 这就牵扯到带入c1还是c2的问题 为了方便统一 即使是c2类也选择带入c1概率式 其是c2的概率是1-c1式概率
>
> $ L_{\mu,\sum}=f_{\mu,\sum}(X^1)f_{\mu,\sum}(X^2)...=f_{w,b}(X^1)f_{w,b}(X^2)(1-f_{w,b}(X^3)...)$(假设1，2类是c1 3类是c2...)但是这样又牵扯到新问题 式子的写入需要人为介入 于是引入同意异名形式
>
> $-lnP(X^1)=-lnf_{w,b}(X^1)=-(\hat ylnf_{w,b}(X^1)+(1-\hat y)(1-lnf_{w,b}(X^1)))$
>
> 则原式为$-ln(L_{\mu,\sum})=\sum_n-(\hat y^nlnf_{w,b}(X^n)+(1-\hat y^n)(1-lnf_{w,b}(X^n)))$ y对于class1 输出1 对于class2 输出2 
>
> 同时 此式也代表cross entropy
>
> 16.问题为什么要使用cross entropy来评判估计效果(最小化量?) 因为如果使用ESM这种方法 问题见第2点 也可以从另一个视角判断 求出ESM下的求导结果 可以判断出$|\hat y^n-f_{w,b}(x^n)|$很大时 偏导结果很小 但是此时应该结果比较大才能距离越远训练越快这样
>
> 17.此部分对$-ln(L_{\mu,\sum})求偏导\ 经过使用链式法则后 偏导的化简有$
>
> $-\frac{\part ln(L_{w,b})}{\part w_i)}=\sum_n-(\hat y^n-f_{w,b}(x^n))x_i^n$
>
> $wi=w_{i-1}-\sum_n-(\hat y^n-f_{w,b}(x^n))x_i^n$​
>
> 18.提到作业一也是存在$\hat y^n-f_{w,b}(x^n)$ 作业一可能就是手动定义L(ESM)手动带入导数 手动带入值 手动计算梯度(线性回归的式子求导可以直接化简出计算公式 $\frac{\part (L_{w,b})}{\part w_i}=-\frac{\part (\frac{1}{2}\sum_n(f(x^n)-\hat y^n)^2}{\part w_i}=\sum_n-(\hat y^n-f_{w,b}(x^n))x_i^n$​)（依旧是链式法则去求 括号内对wi求导结果是和wi相乘的项 对于线性回归简单模型也就是xi）
>
> 19.作业中又提到解least square err 猜测意思可能是带入数据点后 L变成含有多个变量的函数 变成解这个多变量函数的极值问题
>
> 20.问题 为什么ESM求偏导最终的化简形式和极大似然相同？既然L的形式都不一样--->偏导结果需要考虑两个形式 一个是L的形式 一个是内部func的形式
>
> 21.给出一个区分生成式和区分式模型的区别 给了二维的10个数据点使用贝叶斯公式 但是例子中(1,1)在c2中的概率我认为是0而不是1/3x3 因为特征应该是整体的而不是分立的
>
> 22.输出为三维,相当于使用三个维度分别对参数更新,但是使用cross entropy的方式更新,首先进行数学上求导 最后式子包含y,所以相当于只使用1的维度进行更新了

**贝叶斯公式 极大释然估计 极大似然估计** 

> **1. 贝叶斯公式**
>
> ​       假设存在两个事件 事件A和事件B且事件A先于事件B发生
>
> ​       则贝叶斯公式的数学形式如右$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$
>
> ​       $其中:\\P(A|B)称作后验概率(已知事件B发生 验证事件A的概率)\\P(B|A)称为条件概率\\P(A)称为先验概率(在没有观察到B的情况下 对事件A发生的先验信念)\\P(B)成为边缘概率$
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20240919225636112.png" alt="image-20240919225636112" style="zoom:040%;" />
>
> **2.极大释然估计**
>
> ​        核心思想是
>
> ​        计算时考虑的不是当前已知数据在未知参数下的概率 而是当前未知参数在已知数据下的概率
>
> ​        使用贝叶斯公式 $P(\theta|X)=\frac{P(X|\theta)P(\theta)}{P(X)}$
>
> ​        释然函数近似表示为$L(\theta)=P(\theta|X)=\prod^n_{i=1}P(\theta|X)$
>
> ​           $\hat\theta_{MLE}=argmax_{\theta}L(\theta)(即\theta选取一组使其最大化的参数)$​
>
> ​        实际计算中可以将这个值视作极大似然估计和先验概率的乘积(分母为$P(X)由数据中直接得到\ \boldsymbol{问 怎么得到？}$)
>
> ​        其中先验概率一般由于经验给定
>
> **3.极大似然估计**
>
> ​        核心思想是
>
> ​        使用外部条件(比如给定正态分布就已知数的概率和参数的关系)得到已知结果在未知参数下的概率之和 再对其求导 得到使已知结果发生概率最大的一组未知参数
>
> ​         似然函数近似表示为$L(\theta)=P(X|\theta)=\prod^n_{i=1}P(X|\theta_i)$
>
> ​         $\hat\theta_{MLE}=argmax_{\theta}L(\theta)(即\theta选取一组使其最大化的参数)$
>
> ​         最大化的是当前已知数据在未知参数下的概率 
>
> 极大似然是找到$\theta$使X出现的概率最大
>
> 极大释然是找到$\theta$使$\theta$出现的概率最大

**Logistic regression**

> definition: A task that can describe the reflection from various X to two-calss Y

### Discriminative versus Generative

> **distinction:**
>  1. Discriminative寻找w和b 先设定func 再找L 最后SGD
> ​	Generative寻找实际模型的具体参数 根据数据点计算$\mu\ \sum...$
>
>  2. 使用Discriminative找到的parameter和Generative一般不一样 因为Generative是基于意义模型的 是由假设的 但是Discriminative只有形式
>
>  3. benefit of G
>
>     ​	存在对数据的假设 对噪音不像D一样受影响(见2点)
>
>     ​	达成更好的效果需要更少的数据(如果假设合理)
>
>     ​	可以引入前提条件 将前提和类别分析分离开来 例子:语音辨识 类似gpt优先获取说话概率就是prior DNN才真正进行语音识别这种"类型辨识"
>
>     ​	

### Multi-class Classification

**softmax**

> definition：given variate z1=wx1+b1 z2=... z3=... , softmax is to calculate $s(z1)=\frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}}$​
>
> why softmax? :三变量经过二变量高斯分布使用相同协方差的数学推导结果

**procedure**

> step1：特征均带入logic regression计算(输出三维z1 z2 z3) 
>
> step2：带入softmax(二维是sigmoid)
>
> step3：求cross entropy作为criterion, formula seen as$\sum_{i=1}^n\hat y_ilny_i$(n为类别数 帽子是真实值)
>
> **问题**:为什么logic regression中其形式和其不一样？---重要概念 二维的例子中是用一个变量0，1表示两类 这里是使用三个变量分别表示三类 对于每一类 Y都是一个n行vector 其中只有代表其类的行是1 其他均为0
>
> 注:
>
> 1. L的形式是三维下的cross entropy 这个形式本身就代表着只有当前类数据才可更新当前类对应的一维linear regression
>
> 2. limit of Classification based on linear model
>
>    以二维为例
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/4c45680ff267141b1d3777a6fe3eb7f.png" alt="4c45680ff267141b1d3777a6fe3eb7f" style="zoom:10%;" />
>
>    因为同协方差情景下的高斯Gaussion本质还是线性的(什么是本质 为什么是线性见随记32) 区分线是一条直线的话肯定不可能正确率高的将这四个点区分 只能做Feature Transformation 对其转化后的数据进行分类(肯定不能由人去提供转化方式) 这个问题怎么解决？
>
> 3. Solution to point two
>
>    我们需要机器去得到transform数据的方式 
>
>    在数据的输入阶段cascade一个n输入n输出的线性层 这一层就作为data transform 通过对Loss的训练 得到一个好的data transform结果(Loss变低就是data transform在变好)
>
> 4. optimize
>
>    在数据输出前不只使用一个linear+sigmoid 而是多个cascade多个 其中每一个linear+sigmoid就叫做neuron

### 机器学习任务攻略

> 课程跟随
>
> 1.  <img src="C:/Users/Lenovo/Desktop/Typora/pictures/cfbeebb9f7ebdbd17968d4e66a6befd-1727252337062-2.png" alt="cfbeebb9f7ebdbd17968d4e66a6befd" style="zoom:10%;" />训练效果不好分析流程图
>
> 2. 首先对训练集就训练不好的问题分析 
>
>    1.可能是**模型不好** 可以通过增加输入量 替换激活函数 增加连接层的方式实现
>
>    2.可能是**optimization做的不好** 证明这一点可以先拿好训练的小模型比如说linear看看效果 如果比它们都差肯定是没有optimize好 
>
>    同时又给出training data中5层没有四层好的例子(在test内才是过拟合的现象 在训练集内就是没有optimize好) 又说了SGD的局限性(但是为什么之前的课程提出不需要考虑这一点?)
>
> 3. 复杂的模型更加flexible 但是如果数据量不大 就可能导致overfitting(Train上更好的前提下Test更差)
>
>    解决方法有 
>
>    **增加数据量** 
>
>    或者使用**Data augmnetation**(在原有数据以及人为基础认知上 对原有数据进行扩增 比如说图像翻转 局部放大等等 但是对原数据的修改一定要合理 比如说不能将图像上下翻转训练 因为现实中很少有上下翻转的图片导致机器认为这种图片就是正向的)
>
>    或者1.**降低模型的复杂度** 2.**明确模型** constrain模型(训练资料优先的情况下 类似于generative model) Less parameter or sharing parameters(这种方法比如说CNN在影像识别中的作用) 3.**Early stopping** (实时监控Loss 变化不大时就停止 防止过拟合)4.**Regularization** 5.**Dropout**(随机选取某些神经元 使其不更新参数)
>
>    4.提到Model complexity 从怎么判断应该卡在哪个复杂度停止训练模型 到引入kaggle 又说到测试集的public和private 最后提出观点 在训练集上效果好的模型可能本身模型很史 只是运气好数据集和部分测试集效果好(另外为什么作业不给private test 是因为模型重要在预测 如果test都给你了就不用考虑overfitting了 直接加大强度疯狂练就好)
>
>    5.Cross Validation 每一个epoch中都将数据分成Training set 和Validation set(引入问题 Va和test的区别 前者参与模型训练 即使不参与反向传播 但是可以参与Early stopping 后者单纯判断效果) 
>
>    6.其实无论public data还是private data 无论测试集还是训练集 本质上都是数据 我们的策略应该是将这些本质一样的数据进行分类使用 对test data不用太过考虑其Loss 有时其Loss为零反而不如存在一些Loss对真实的模型描述更好
>
>    7.提出N-fold Cross Validation 原数据分成n等分 每一个epoch更换一次 
>
>    8.又提出validation的另一个作用 除了对一个模型进行early stopping 还可以对不同模型进行compare
>
>    9.mismatch 定义为训资料和测试资料分布不一样 可以认为是overfittting 但是李认为overfitting是可以通过增加数据量来克服(提取出的意思就是说 overfitting是对模型没有拟合成功 而mismatch是训练集即使集合成功了也和测试集的模型不一样(因为资料中可能存在少量的噪音)) 比如说20年的covid数据训练预测21年covid数据 
>
>    这一步就需要人去介入了 通过field knowledeg人为判断数据是不是描述一个模型
>

### 类神经网络训练不起来怎么办

**课程跟随**

> **p1介绍了为什么会出现Loss不变 以及其中的一个情况怎么处理**
>
> 1.The reason for failure of the optimization
>
> The function encountered a critial point, which contains local minima or saddle point where we get the gredient as zero
>
> 2.The deduction of math to tell the difference of local minima(向各个变量方向行走变化趋势均变小) and saddle point(向外往某些变量方向变大 往某些方向变小)
>
> First using Tayler Series Approximation to get the proximity result of the point which is extremely near the point we are studying
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/28e557228d1767f495657b43e85940b.png" alt="28e557228d1767f495657b43e85940b" style="zoom:15%;" />
>
> Second in the point we are studying the value of the gredient is zero, so we can simplify the form of the formula, then we need to study the trait of the quadratic item to understand whether the critical point is local minima or saddle point
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/87f371b9e5b8e641498d222f750fe10.png" alt="87f371b9e5b8e641498d222f750fe10" style="zoom:15%;" />
>
> > 补充关于特征值 特征向量 特征值矩阵的一些知识点
> >
> > 1. 构建特征方程$(\boldsymbol A-\lambda \boldsymbol I)\boldsymbol V\ 其中\boldsymbol A为目标矩阵\ \lambda为特征值\ \boldsymbol V为特征值对应的特征向量$(引入概念 一个特征值对应一组线性相关的特征向量 但是特征值的大小是固定的)
> > 2. 求解特征值 使用矩阵乘积的行列式的值等于矩阵分开行列式的值的乘积这个形式 求解$|\boldsymbol A-\lambda \boldsymbol I|(也就是det()$ 求解这个方程的解即得到特征值标量的集合
> > 3. 求解特征向量 将目标特征值带入原式进行矩阵运算 得到0矩阵 求解特征向量各个元素的值
> > 4. 性质 由特征方程$(\boldsymbol A-\lambda \boldsymbol I)\boldsymbol V=\boldsymbol 0$展开化简得到$\boldsymbol A\boldsymbol V=\lambda \boldsymbol I\boldsymbol V=\lambda \boldsymbol V$​
> > 5. 对所有矩阵 特征值全为正为目标矩阵为正定矩阵的必要条件 但是对实对称矩阵 这个是充要条件
> > 6. Hession一定为实对称矩阵(在某域内 如果函数二阶偏导数连续 则对ji两个变量求偏导 顺序不影响结果)
>
> Third 使用上面的结论 假设存在$u^{T}Hu\ u为H的特征向量 其为负值 这一步是直接使用上述的结论$
>
> $而需要判断正负的是(\boldsymbol \theta-\boldsymbol \theta')^TH(\boldsymbol \theta-\boldsymbol \theta')\ 如果特征值全大于0 H正定 乘积必正 反之全小于零 \\H负定 乘积必负\ 如果特征值有正有负 那乘积也是有正有负$
>
> $如果此时已知有正有负\ 需要找到能够降低Loss的方向\ 则这里有一个巧妙的代换\\ 首先需要证明u带入计算乘积为负值 只要性质4进行化简即可 \\也就是说\boldsymbol \theta变化的方向是u即可保证带入乘积为负 那么变化后的点就已知 这种方法可以找到需要变化的方向$ 
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/bfb81f7e65f837d179899beb641cad9.png" alt="bfb81f7e65f837d179899beb641cad9" style="zoom:15%;" />
>
> 2.介绍说实际使用进行SGD方式训练的model 在多模型feature的情况下$Minimum\ ratio=\frac{Number\ of\ Positive\ Eigen\ values}{Number\ of\ Eigen\ values}$越小Training Loss越大 且几乎找不到所有eigen values都是正的模型
>
> 3.综上的理解 一个使gradient为零的feature集合对应一个H 通过分析H的这个特征值来判断是local minima还是saddle point 并且可以向特征向量方向进行最小化
>
> 并且特征值越接近全正 点越接近local minima(这个性质理性认识就好) 
>
> 4.Loss卡在某一值可能就是卡在了梯度为零的点
>
> **p2 介绍了batch size对训练的影响和另一种解决Loss不变的方式**
>
> 1.引入batch的概念 就是将所有样本分成均分就叫做batch batch定义的核心在于更新参数只在batch最后完成 一整个batch去计算Loss使用的原参数都一样
>
> 2.分析使用batch和不使用的区别
>
> 使用batch:long time for calculating(but slightly due to parallel calculating )but powerful
>
> 不使用batch:short time for cooldown(shortly also) but noisy
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/a3bd2d711b5edd51004201a5104694e.png" alt="a3bd2d711b5edd51004201a5104694e" style="zoom:15%;" />
>
> 3.引入平行计算的概念 
>
> 计算一个batch和一个数据的SGD不一定前者比后者慢很多 因为gpu加速可以使用平行计算 如果batch特别大才会跟单数据有区别
>
> 这一注意一个重要的点 一个epoch 一个个计算和分batch计算 都是分batch计算大大领先 并且batch分的越多总时间越小 因为batch越大 相当于平行计算同时计算的单元更多 使用gpu的效率更高
>
> 4.小的batch size(noisy)反而在训练效果上更好 解释原因是optimize的问题
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/182ddcc914dcb956c8246d616292919.png" alt="182ddcc914dcb956c8246d616292919" style="zoom:15%;" />
>
> Full batch
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/deb6bbfc84cb1080a6be90f626fcf4e.png" alt="deb6bbfc84cb1080a6be90f626fcf4e" style="zoom:15%;" />
>
> 5.同时小的batch对test data的拟合也有帮助 小的batch不容易造成overfitting
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/888bae54786ad1713214f51eeb629dc.png" alt="888bae54786ad1713214f51eeb629dc" style="zoom:15%;" />
>
> 第一层解释:图像怎么理解?--->横坐标是一个模型参数 纵坐标是Loss
>
> 第二层解释:为什么宽的峡谷更好？--->因为testing data可能分布和training data不一样
>
> 对于窄“gorge” 在Loss-parameter模型改变时(假设即使模型改变 该点变化的快慢这个性质不会变的太大) Loss差异不大
>
> 第三层解释:为什么Small Batch更容易落到宽"gorge"? 对于Small Batch 其值存在跳跃性 可能存在极端点导致当前参数计算出的gradient更大 对于窄gorge更容易跳出
>
> 6.总结 big batch快但是容易过拟合 small batch训练慢但是testing data效果更好
>
> 7引入momentum(冲量 冲力 动量) 类似于小球滚下坡 不一定就会在一个坡上的坑停住
>
> 实际的操作类似于引入i量 当前参数走的步数会考虑之前所有gredient的计算值累加 具体的公式时
>
> $m^n=\lambda m^{n-1}-\eta g^n$(当前需要移动的步数就是上一次移动的步数减去当前的梯度更新($\lambda可以调整过去值的衰减$​)
>
> **p3介绍了对学习率的两种修正方式** 
>
> 1.介绍了Loss卡住可能是在峰低来回震荡(learning rate可能太大了)
>
> 2.对于最普通的convex图形 learning rate会出现稍微大一点 就会在峰底内震荡(当处计算的梯度更新后的点又根据梯度更新返回最开始的点) 但是在接近Loss后训练速度又会太小小到接近目标值的梯度都不足以其达到接近目标值的地方(更新太慢以至于没有实际意义)
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/edb6cbd1ddd45a780c7f7e423e480e9.png" alt="edb6cbd1ddd45a780c7f7e423e480e9" style="zoom:15%;" />
>
> 3.引入变化大learning rate小 变化小 LR大的方法(AdaGrad(Adaptive Gradient Algorithm)) Root Mean Square
>
> $对学习率\eta进行修正 除以修正系数\sigma 第n次更新的\sigma=\sqrt{\frac{1}{t+1}\sum^t_{i=0}(g_i^t)^2}(上标针对epoch 下标针对特定参数)$
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/3cbd3f9cbd772c066a8d0521d8a55e2.png" alt="3cbd3f9cbd772c066a8d0521d8a55e2" style="zoom:15%;" />
>
> 这里解释为什么这样做会解决2中的问题 对于蓝线 gradient均比较小 修正后的学习率更大 也就是平缓阶段学习率更大 防止出现到不了L为0的情况 绿线和蓝线相反 陡峭部分学习率更小 减小了震荡的情况(引入训练的技巧 可以打印L和gradient随函数迭代的情况)
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/e07c945b4109f8cfe2158326014eb3f.png" alt="e07c945b4109f8cfe2158326014eb3f" style="zoom:15%;" />
>
> 4.RMSProp
>
> 和上诉思想一样 区别是修正系数考虑了对过去数据和当前数据的区分度
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/72e158e2217999b146601451e3ee4dc.png" alt="72e158e2217999b146601451e3ee4dc" style="zoom:15%;" />
>
> RMSprop这个方法相较于均方的有点在于 对学习率的修正反应更快($如果\alpha给小一点的话$)
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/c1bf74621b8d0babfd0df2098dac23f.png" alt="c1bf74621b8d0babfd0df2098dac23f" style="zoom:15%;" />
>
> 5.介绍Adam策略 就是RPMSProp+Momentum
>
> 同时说pytorch有配套的方法可以直接使用 各种参数大部分使用预设值就好
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/6ec1615fba681bc151a0ef1e8d4079d.png" alt="6ec1615fba681bc151a0ef1e8d4079d" style="zoom:15%;" />
>
> 6.真正使用了Adagrad的思想后 实际的效果是右下角的图像
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/e423d5b3711fbe1d08e5e3772665567.png" alt="e423d5b3711fbe1d08e5e3772665567" style="zoom:15%;" />
>
> 第一层解释: 为什么要考虑二元？ 图示的情况需要在w已经达到目标的情况下继续train 也就是说需要一个没有train到位的参数让train继续下去
>
> 第二层解释:震荡的是w 单独考虑w即可 w在本身梯度很小的情况下 被迫计算了太多次 导致学习率修正的过大 导致震荡 但是震荡是收敛的(因为修正系数随着震荡也在不断增大)
>
> 第三层解释:越向后震荡的频率越大是因为 越向后b训练的越慢 对train相同次数可能b变化越来越慢 导致看起来频率变小
>
> 7.为了解决6中问题 可以引入Learning Rate Scheduling内(其和learning修正的区别是 前者和训练次数有关 后者和当前点的梯度有关)
>
> ​	1. learning Rate Decay
>
> ​		lrate根据epoch单调降低
>
> ​	2. Warm Up
>
> ​		lrate根据epoch先变大再变小		
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/468210a528f5a92930ae8c1ae3fd820.png" alt="468210a528f5a92930ae8c1ae3fd820" style="zoom:15%;" />
>
> 8.Summary of Optimization
>
> (Vanilla) Gradient Descent
>
> $\theta^{t+1}_i=\theta^{t}_i-\frac{n^t}{\sigma^t_i}m^t_i(上标是epoch序号\ 下标是参数序号\\对\eta有根据epoch改变学习率的Learning\ rate\ schedule\\对于sigma有Momentum(consider\ direction)(或者很少用的计算H的负特征值对于的特征向量方向)\\对学习率修正系数\ 有root\ mean\ square\ of\ gradients(only\ magnitude)$
>
> 
>
> ·····················``

###   再谈宝可梦

> 1. 引入了新的区分二类的方式 找中间值 使模型输出大于此值为一类 小于此值为一类
>
> 2. Loss的形式是每个数据判断的和 判断的形式是判断正确返回1 判断错误返回0
>
> 3. 引入思想 部分集训练的结果在训练集上的L可能比全集训练的结果在全集上面的结果可能还低 
>
> 4. 又进行了数学计算 讨论训练集和全集训练模型在假设下的L的差距
>
>    对h的理解是对训练参数效果最好的模型(包括模型参数大小 参数连接方式等等)
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/012d00dbb534a6748274a575c6ecef7.png" alt="012d00dbb534a6748274a575c6ecef7" style="zoom:15%;" />
>
>    5.经过概率学的推导 减低模型复杂度 训练集所含样本变大 坏集合的概率就变小
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/20bba0892811b77632dccb9c0b7cee9.png" alt="20bba0892811b77632dccb9c0b7cee9" style="zoom:15%;" />
>
>    这里存在一个模糊的点 H H是什么？H是所有可能的模型数 对神经网络 H的个数取决于参数排列组合
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/95759dc57b95ce5c32e26675f7964f5.png" alt="95759dc57b95ce5c32e26675f7964f5" style="zoom:15%;" />
>
>    这里的字母$\delta代表概率的限制\ \sigma代表L的限制$
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/9c8e5f2b6d70aa2dac49bccbb85247e.png" alt="9c8e5f2b6d70aa2dac49bccbb85247e" style="zoom:15%;" />
>
>    6.对上述进行总结 也是最关键的一点 **复杂模型容易过拟合 简单模型上限不高**
>
>    复杂模型容易过拟合的原因是 复杂模型 H大 使用训练集训练出的模型可能就是差模型 和全集的模型相差较大 这也就是过拟合的根本原因(训练集效果好 测试集效果差)
>
>    简单模型虽然使用局部样本训练的模型和全集训练的模型相差很大的概率小 但是容易L本身很大
>
>    这一节根本就在使用概率学解释上述两解
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/f4c95794137aeb7c6856b6790762efc.png" alt="f4c95794137aeb7c6856b6790762efc" style="zoom:15%;" />
>
>    
>
>    后续介绍在P33 鱼于熊掌不可兼得 告知了深度学习可以既实现训练集和全集最优模型L相差不大 同时两者都不算太高

### CNN(卷积神经网络 Converlutional Neural Network)

**课程跟随**

> 1.一种Network的架构 以此介绍network的架构以及有什么用
>
> 2.专门用来处理图像使用 假设所有图像的分辨率都是一样的 (输入模型的图像需要预处理)
>
> 3.使用独热编码来对目标进行分类
>
> 4.使用的模型是正常的线性加sofxmax(二维就是加sigmoid 数学可证明这样底层就是默认高斯) 再进行输出和独热编码的cross entropy的评估
>
> 5.图片可以分成三个通道 (RGB) 三个Tensor合成一个vector(假设图片分辨率100x100 这个vector的行数就是100x100x3)
>
> 6.因为图像的输入数据过大 导致模型过于复杂 如果每个参数和每一层的神经元都有联系 参数过于庞大 容易overfitting加优化困难
>
> 这里有一个问题 怎么理解图示中连接的数学运算以及神经元的个数？---神经元的个数就是列的个数 连接就是相乘 汇合就是相加 这里其实就是($A_{1\times n_{feature}}\times B_{(n\times n)}\times C_{(n\times n)}\times ...$)的运算结果 输入视作行向量(如果右乘就是列向量) 所以理解参数个数有两个方式 第一个方式是看连接的线 一个线一个参数(每一层都有$(input\times output)$个线 ) 第二个方式是看矩阵 将线连接抽象乘矩阵相乘 直接看矩阵的个数(每一个矩阵都有$(input\times output)$个参数 一共n个矩阵)
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/3d097db7c68aae0434d11a678081037.png" alt="3d097db7c68aae0434d11a678081037" style="zoom:15%;" />
>
> 7.to simplify
>
> Observation 1:一个神经元可能不需要看完全部的图片 只需要看到特征就好了
>
> Simplication 1: 引入**Recptive field** 也就是将原图像参数分块 对每个块进行计算(Receptive field之间是可以重叠的) 这部分的计算还是神经网络的思想 将一个方格内的参数输入到一个线性网络之中 输出作为这个方格的代表(而下是矩阵点乘filter 结果参数的sum作为新值)
>
> 
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/0c1cbb5304a86b0112b93905cfe72f1.png" alt="0c1cbb5304a86b0112b93905cfe72f1" style="zoom:15%;" />
>
> 
>
> question: RPG分开考虑能不能？ Recptive field能不能分块面积不一样？ 能不能使用长方形的？ 能不能不全部使用图片中所有像素？
>
> ---Typical Setting
>
> > 1.一般设置kernal大小3x3 每一个""感受野"都有多个neurons守备"(对这一点有比较重要的了解 多个神经元相当于对分类特质进行二分类 这一步本身就是机器对图像数据的处理 所以神经元一般都是linear + sigmoid)
> >
> > 2.Receptive field的移动一般要保证相邻的重叠(移动称作stride)
> >
> > 3.如果移动超出图片 可以向kernal 中padding zero
> >
> > 4.这一层image的channel数就是上一层neuron的个数(同下 上一次多channel的处理还是全部计算出后取和)
>
> Observation 2:同一个特征在不同的region出现
>
> Simplication 2: 存在不同的kernal对应的模型参数均是一样的 此时这种神经元叫做filter
>
> 也就是存在不同的kernal判断同一个特质的方式是一样的(也就是对应的神经元参数是一样的)
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/ee5f4c86997586648ecb0bf05f79763.png" alt="ee5f4c86997586648ecb0bf05f79763" style="zoom:15%;" />
>
> 对下图
>
> > Pattern指图形样式 即图形特征
> >
> > 模型越小越不复杂 model bias越大(上限低) 但是也不容易overfitting(下限高)
> >
> > CNN专门为影像辨识设计的(分块加不同块可以同参数 均是针对影响来设计的 对其他方面的使用需要考虑合理性)
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/4b8d8f0fc023a48bb5d95e48ff48e6d.png" alt="4b8d8f0fc023a48bb5d95e48ff48e6d" style="zoom:15%;" />
>
> 8.对CNN的另一种解释
>
> > 1. filter的新定义:上为一个神经元 这里原图有多少通道 有多少大小 就有多少通道多少大小的一个Tensor 也就是原来是3个channel的3x3 之前的定义一个filter只是单矩阵 但是现在是3个矩阵
> > 2. kernal定义一样 只不过stride设置成1 只不过其被filter处理的过程从矩阵乘法变成点乘(原视频这里介绍时用了黑白的例子 没有说明原图是3通道 3个tensor处理后的矩阵怎么处理 合并？)
> > 3. filter的运算决定了生成的新image的大小 filter的个数决定了新image的channel(这里注意一点 对于第一层filter 其维度应该是3 和上一层(也就是原本图像的RGB三通道数量对上 也就是一个通道和一个通道的filter对应相乘 最后三个通道得出的结果再相加作为输出特征图再(i,j)上的值)) 但是filter输出不同的层 多少个filter就输出多少个不同的层 所以下一层image的channel就是上一层filter的个数
> > 4. Network只要够深 每个节点都将包含原图足够大面积的信息(压缩再压缩...)
> > 5. CNN的基本操作就是 filter扫描原图 叠加kernal大小的图层(3or多通道)输出一个node filter换行node就换行 不同的filter叠加图层 得到的就是CNN处理后的结果
>
> 8.两个模型的根本区别就是 一个用神经元的方式去压缩 一个用filter卷积的方式去压缩
>
> 9.Observation3 将一个图片横纵列去除奇or偶项 整体图像保留的信息大体不变(这就较subsampling)
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/2ba24b8dc6af462a934330786c9bd68.png" alt="2ba24b8dc6af462a934330786c9bd68" style="zoom:15%;" />
>
> 引入Pooling
>
> 课程中使用的是Max Pooling 也就是新生成的image 不管channel 将平面再分成不同小格 取每一个方格中最大的参数代表整个方格来减少输入参数
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/4a5e1f153babb079d616f48c892d14c.png" alt="4a5e1f153babb079d616f48c892d14c" style="zoom:15%;" />
>
> 注意Pooling是为了减小参数 如果算例足够 使用full-convolution可能效果更好
>
> 10.CNN最终的实现 本质上是对图像进行线性变换再输入到正常的Multi-Classification Model内 得到独热编码区分
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/64f2533e9c93265e14ad55504a587cd.png" alt="64f2533e9c93265e14ad55504a587cd" style="zoom:15%;" />
>
> 11.CNN的另一应用---下围棋
>
> 为什么下棋可以使用CNN？
>
> > 因为围棋 一特征可能小范围 二多处可能有相同特征
> >
> > 但是注意下围棋不可以使用池化 因为去除列或者行信息直接被打乱了 这和图像有根本区别
>
> > 初始图像将每一个点位作为pixel 其有17个channel描述其信息 包括有无子or自身子色 周围信息等等
> >
> > filter输出的卷积层图像有48个channel(说明有48个channel数为17的filter对图像进行处理)
>
> 12.对filter的加深理解 其通过对初始图像(RGB 3个channel)进行处理 得到新的channel(对图像的分析视角)下的图像情况 机器更适合对这种数据处理

### 鱼与熊掌不可兼得(深度学习兼容参数少效果好)

**课程跟随**

> 1.又介绍了一遍relu hard-sigmoid sigmoid
>
> 2.由引言 又引出为什么不使用大量并联结构而是神经网络的串联结构？从数据上看 还是深度学习的串联更具有效率 相同的参数量效果更好
>
> 3.引入数电 数电回顾各种门的名称 图形 再引入一个逻辑电路的设计(理解这个逻辑电路从底层10 01 101 010向外扩展即可  )
>
> 4.这一节没说什么特别重要的 核心就是复杂且重复的模型深度学习更有优势 因为其能使用达到更少的参数达到这个效果 而生活中语言辨识等等都具有这种特点 所以deep比shadow好;

### 自注意力机制

课程跟随

> p1
>
> 1.原来的架构都是输入一个固定长度的vector 现在如果想输入多个 不固定长度的向量怎么办(比如说语音识别 个人认为多个不是重点 一个个输入即可 但是不固定长度是个痛点)
>
> 2.引入unique-hot Encording 和word Embedding 前者看不出语义联系 后者存在语义联系
>
> 3.引入语言辨识的一些基础知识 一般25ms作为一个frame作为描述的基础 令stride为10ms...语言辨识这个任务就是多输出不固定长度的向量
>
> 4.输出的类型分类 A:每一个vector都输出一个label B:整个sequence输出一个label C:sequence2sequence(与A不同在 输出的不一定是label 可能是vector 且前后数量不一定对等)
>
> 5.对1中一个个输入的思想提出批判 比如说词性辨识 前后输入有关系 不能独立的输入进模型进而分别得到label 需要模型考虑上下文---解决这个问题的方式是每一步都给一个window(而非一个个输出单个vecdor 而是当前vector的相加输入进模型) 但是这个方法仍然有缺陷 对于需要考虑整个sequence的场景 此时需要window将整个sequence盖住 但是sequence有长有短
>
> 6.self-attention的框架 
>
> A 特别的机制 考虑向量之间的关联程度 也就是Dot-product(也有其他的方法 如下图)
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/18805533676abba14f43ad7293a4f56.png" alt="18805533676abba14f43ad7293a4f56" style="zoom:15%;" />
>
> B 存在3个中间向量 q k v 每一个输入都有自己对应的q k v矩阵
>
> 矩阵乘q为主动方 矩阵乘k为被动方 此双结果再做inner product**(!!! $q^1\cdot k^2\ equals\ \ q^1(k^2)^{T}$)**就得到attention score
>
> 注意 下图没有写自己和自己的相关性 实际操作是有这一步的
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/439670701f6afd30256aa8121efd874.png" alt="439670701f6afd30256aa8121efd874" style="zoom:15%;" />
>
> C 将每一个注意力分数和其对于的向量的v矩阵相乘 再相加 就得到输出$b_1$
>
> <img src="C:/Users/Lenovo/Documents/WeChat%20Files/wxid_ozepkuh1ai8l22/FileStorage/Temp/36be3be90e6b51b7786eae41de6f545.png" alt="36be3be90e6b51b7786eae41de6f545" style="zoom:15%;" />
>
> 7.整体的架构是输入到self-attention 输出对应数量的vector 再输入full connection 再输入到self-attention 最后由FC输出需要的label/vector
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/9fa49fd3294c8347d358f07d2a3bb9c.png" alt="9fa49fd3294c8347d358f07d2a3bb9c" style="zoom:15%;" />
>
> 8.将上述分散的运算整合 通过组合输入的q k v矩阵得到$W^q\ W^k\ W^v$​​​三个矩阵 宏观的运算可以得到整体的注意力权重矩阵 再整体softmax(对每列)
>
> 此为第一步 这一步没有矩阵的抽象 所有输入共用一个$W^q\ W^k\ W^v$
>
> $q1_{n\times 1}=W^q_{n\times n}a1_{n\times 1}(深体为系数矩阵)$
>
> 则$\begin{bmatrix}\ q1\ q2\ q3 \ q4\ \end{bmatrix}_{n\times n}=W^q_{n\times n}\begin{bmatrix}\ a1\ a2\ a3\ a4\ \end{bmatrix}_{n\times n}$
>
> k,v同理
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/2acb0ed33f6de7c7354af751655f44a.png" alt="2acb0ed33f6de7c7354af751655f44a" style="zoom:15%;" />
>
> 此为第二部 将q k转化为$\alpha$    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241007010819504.png" alt="image-20241007010819504" style="zoom:45%;" />
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/74c5e4f5221e0c5144df0a0d35517b9.png" alt="74c5e4f5221e0c5144df0a0d35517b9" style="zoom:15%;" />
>
> 此为第三步 将$\alpha$ v转化成b 注意这里$\alpha1，2代表1对2的注意力 对谁的注意力乘上自身的v$
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/1eb3e58f629ea14915b1dba7f72e964.png" alt="1eb3e58f629ea14915b1dba7f72e964" style="zoom:15%;" />
>
> 只有$W^q\ W^k\ W^v$中的参数是未知的
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/5377bf69fbc65f85a3dd16e837b0d4d.png" alt="5377bf69fbc65f85a3dd16e837b0d4d" style="zoom:15%;" />
>
> 9.Multi-head Self-attention(Different types of relevance)
>
> 什么是head？ 也就是对一个输入存在多个$q_i^1,q_i^2$​(下标是输入参数index\ 上标是head)
>
> 这里分别计算了两个head的=$b^1_i\ \ b^2_i$(上标代表head 相当于并行操作了) 
>
> 值得注意的就是 计算步骤是先由$W^q$计算出$q^i$ 在用$q^i$乘上两个矩阵得到$q_i^1,q_i^2$ k v的head并行计算方法同 再不同的head内分别计算$b^1_i\ \ b^2_i$
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/ae044488d97ba04d5efc12e843c9b41.png" alt="ae044488d97ba04d5efc12e843c9b41" style="zoom:15%;" />
>
> 再根据得到的$b^1_i\ \ b^2_i纵向合并得到2n\times 1的列矩阵 乘上n\times 1的系数矩阵就得到b^i$
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/b24a002ba027f3b0fe3fd50a2ca72b2.jpg" alt="b24a002ba027f3b0fe3fd50a2ca72b2" style="zoom:15%;" />
>
> 10. Positional Encoding
>
>     对于self-attention框架来说 输入没有位置信息(这个点比较抽象 因为本身矩阵的前后不能代表位置这个概念)
>
>     解决方法:Each position has a unique positional vector $e^i$​
>
>     下图中的颜色代表什么还不太清除
>
>     <img src="C:/Users/Lenovo/Desktop/Typora/pictures/47c731d8809f25ca58b9e23c0447809.png" alt="47c731d8809f25ca58b9e23c0447809" style="zoom:15%;" />
>
>     注意一点 b图是把$e^i$当作参数训练出来的图像
>
>     <img src="C:/Users/Lenovo/Desktop/Typora/pictures/d2402f2c4debf50e1355c3aa7ed9615.png" alt="d2402f2c4debf50e1355c3aa7ed9615" style="zoom:15%;" />
>
> 11. 介绍Transform和BERT(用于nlp 自然语言处理)均使用到self-attention
>
> 12. 语言辨识10ms为一个stride 1s的语音就产生上前个输入 也就是上百万个inner product($a^1$为主动计算1000次 这种计算也循环1000次)
>
>     <img src="C:/Users/Lenovo/Desktop/Typora/pictures/ba6f82d369e771600db753326d52321.png" alt="ba6f82d369e771600db753326d52321" style="zoom:15%;" />
>
> 13. Truncated self-attention 也就是不考虑全部的样本 只考虑一部分？
>
> 14. 提出观点 self-attention是CNN的全集 CNN人为规定了关注度 也就是邻近的几个pixel的关注度绝对高 实际的表现也支持这个观点 一个任务使用self-attention+FC和使用CNN+FC 使用相同数据训练的结果 在test上的表现 数据增加时 开始CNN好(参数少 模型相对不复杂) 后来self-attention的效果更好(模型更复杂)
>
>     <img src="C:/Users/Lenovo/Desktop/Typora/pictures/87f1870183fe7f0c083fb24f49138f3.png" alt="87f1870183fe7f0c083fb24f49138f3" style="zoom:15%;" />
>
>     15.引入RNN 
>
>     RNN对比 self-attention的两个劣势 
>
>     有次序以及不能平行计算(计算效率低 训练慢))
>
>     <img src="C:/Users/Lenovo/Desktop/Typora/pictures/34cc5474a38be125765aaf6cfc28c74.png" alt="34cc5474a38be125765aaf6cfc28c74" style="zoom:15%;" />
>
>     引入使用self-attention来获取图像信息
>
>     这种方式下 将像素集看作node 只关注相连接的node 计算它们的注意力权重 而不相关的直接赋0即可
>
>     <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241008230953699.png" alt="image-20241008230953699" style="zoom: 33%;" />
>
>     16.最后提了一嘴 使用self-attention为内核的模型大多叫做xxxformer
>



### Batch Normalization(批次标准化)

> 1. 提到在作业三和CNN中能够带来很大的的帮助
>
> 2. 提高一个问题 一组feature中各个量量纲可能存在很大差别 这就导致对于一个feature的输入 其导致的Loss一般比较大 Loss的变化(也就是feature变化比较大) 导致模型的训练不同维度存在差异
>
>    所以这里希望将每个feature的量纲定在同一个范围 同时不丢失参数分布的信息
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241010190914320.png" alt="image-20241010190914320" style="zoom:50%;" />
>
>    Batch Norm的实际操作 是对不同feature的同一个feature进行标准化
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241010191410748.png" alt="image-20241010191410748" style="zoom:50%;" />
>
> 3. 这里提到不止只有最开始输入可以Norm 中间的中间量也可以进行Norm操作
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241010191621754.png" alt="image-20241010191621754" style="zoom:50%;" />
>
> 4. 这里提到几个点 
>
>    Batch最好比较大
>
>    将模型复杂化(添加了Norm的计算 GD计算变复杂)
>
>    输出的输出的每一个变量都相互关联
>
>    
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241010192338567.png" alt="image-20241010192338567" style="zoom:50%;" />
>
> 5. ​        引入对Norm的数据再次普通化(因为Norm本身也是限制) 但是这种普通化并不一定就让效果还原 因为最开始$\gamma就是全1矩阵$
>
>    ​	注意这里的运算是矩阵元素相乘而非矩阵相乘
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241010193123730.png" alt="image-20241010193123730" style="zoom:50%;" />
>
> 6. ​        使用batch训练的模型 本身可以不使用batch处理数据 但是使用batch Norm的模型 理所当然的需要使用batch 但是实际的时候肯定不是时时刻刻按batch输入的
>
>    ​	这里实际上的操作是 使用训练数据去推测测试数据的均值和方差 每计算一次batch train 都会更新一次这个参数 实际使用单数据的时候 就不会要求这个单数据去计算均值和方差了 而是直接使用训练集中本身得到的均值和方差
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241010193512391.png" alt="image-20241010193512391" style="zoom:50%;" />
>
> 7. 提了一嘴batch Norm作用在CNN上 我的推测是 核心是怎么区分输入的feature
>
>    对于图像 一个pixel中的一个通道就是feature 对这个做Norm
>
>    下图的信息 同样的学习率做了BN速度更快 同时BN可以给更大的学习率(因为相当于把error surface变平缓了)
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241010194628110.png" alt="image-20241010194628110" style="zoom:50%;" />
>
> 8. 实践上和理论上 BN都可以改变error surface
>
> 9. 对这部分我认为最重要的理解就是运算过程 你会发现 最开始输入的BN是好了理解的 是独立于计算的 但是中间层的BN在前向计算中 是每层计算先卡一下 算出所有的数值时候 才能进行下一次的计算 所以**中间的计算就不是完全并行的** 而是batch内不同数据运算相关的 
>
>    并且并且计算GD 在反向传播中 卡在最后一次BN(也就是反向视角的第一次BN) 即使使用了链式法则 但是后向前的求导 也不是简单的一次func求导 或者直接得到linear的参数了 而是和上一次的输出全部都有关系 这一步的求导复杂了(这里的理解就是 GD用反向传播和链式的思想也还能求 就是函数形式非常复杂 要包含上一次的所有输入(除本次偏微分下的求导 其他也是简单带入前向计算的实值) 同时函数形式很复杂 对一个输入的求导形式hen'nan)

### Transformer

> 1.首先介绍的就是Transform在sequence2sequence的优势 输入输出数量有关系 但是没有绝对的翻译
>
> 2.介绍了很多相关的用途 语言翻译 语音辨识 语音输出 聊天机器
>
> 3.介绍现在NLP的处理可以归为QA,QA又可以归为seq2seq 但是对于特定的任务 为任务客制化模型效果一般会更好(这部分主要是NLP 不重要)
>
> 4.语法解析也可以seq2seq
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241009222642554.png" alt="image-20241009222642554" style="zoom: 50%;" />
>
> 5.首先区分Multi-class和Multi-label的区别 前者是分类任务 后者是一类多label任务
>
> 如果使用分类问题的思路去解多label任务 即输出的序列取数值比较高的来确定label 也会出现到底要取几个label这个问题
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241009222819989.png" alt="image-20241009222819989" style="zoom:50%;" />
>
> 6.介绍transform架构的根本在于encoder和decoder
>
> encoder：
>
> ​	首先使用简要的encoder模型(如下图) 每一个block并不称作一个layer 因为它在做好几个layer在做的事
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241009223658045.png" alt="image-20241009223658045" style="zoom:50%;" />
>
> ​	接下来引入实际上transform的工作(使用残差网络(residual network))
>
> ​	残差网络的基本做法就是self-attention的输出加上对应输入得到新的输出 这个residual的输出进行归一化(计算均值方差 值减去均值除以方差得到新值) 
>
> ​	这样的输出再输入到FC中 再进行一遍这样的操作
>
> 
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241009224119156.png" alt="image-20241009224119156" style="zoom:50%;" />
>
> 对encoder的原始论文的设计解释
>
> Positional Encoding是对数据位置的解析 扩展原始输入后输入到Multi-Head Attention中 再进行residual和Norm再输入到FC中再进行操作...
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241009224626502.png" alt="image-20241009224626502" style="zoom:50%;" />
>
> ​	下图是对原始架构的修改 和Layer Norm和Batch Norm的选择
>
> <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241009225135962.png" alt="image-20241009225135962" style="zoom:50%;" />
>
>
> **p2**
>
> 1. 对下图 首先Decoder吃两类东西 第一类是Encoder的输出(这个怎么吃的很模糊)
>
>    第二类是由Decoder自己的Output组成的input(先吃一个Begin(一个符号的one-hot 表示该符号该位权重最大 也即此处开启sentence的概率最大))
>
> 
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241011150914950.png" alt="image-20241011150914950" style="zoom:50%;" />
>
> 2. 引入流程图，简要介绍Decoder做的事情(流程上)和Encoder类似
>
>    但是输入进行self-attention时 是Masked的特化 
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241011163653181.png" alt="image-20241011163653181" style="zoom:50%;" />
>
> 3. 这里解释什么是Masked的特化
>
>    这里的a1-4不是Encoder的输入 而是Decoder自己的Output组成的input
>
>    每一个位置只能使用该位和该位前的k矩阵计算注意力权重(很合理 因为先产生的不应该受后产生的数据的影响 只有后产生的数据可以使用先产生数据的信息)
>
>    这里对底层的数学稍微介述一下,最终得到的矩阵为上三角矩阵,最后$V\alpha$的乘积也只有一个输出(即$V乘上\alpha最后一列得到的结果$)补充到输出seq中
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241011151554024.png" alt="image-20241011151554024" style="zoom:50%;" />
>
> 4. 上图解决了Decoder是怎么计算输入的 但是不能输入输出一直循环下去 所以字典内需要存在一个符号可以代表序列停止(和开始的begin符号一个效果)
>           /<img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241011152346938.png" alt="image-20241011152346938" style="zoom:50%;" />
>
> 5. **AT的输入数一直在改变 内部参数维度怎么处理---问这个问题是忘了self-attention的计算特点 内部参数固定的是三个$n\times n$的矩阵$W^qW^kW^v$ 所以外部输出的个数任意变化 内部参数不用改变 只不过输出的注意力权重矩阵维度改变了 最后的输出b个数也和a一起改变了(self-attention的优点 输入改变维度内部参数不用改变)**
>
>       这里再使用Encoder就将变长序列转为定长序列 之后则易处理	
>
> 6. 引入Non-autoregression Decoder
>
>    AT Decoder就是上文的串行计算 最后得到完整的seq
>
>    而NAT与AT的区别就是 输入的大小是定死的(但是能够保证大于等于可能输出的结果)
>
>    确定NAT输出的seq(如果不做这一步 又是n-n 不是seq2seq了):
>
>    1. 让输出中不包含end标志 准备一个classification 输入整体数据 输出希望Decoder输出的序列数(这个不太好train把吧 首先如果将其和Decoder视作一个模型 连接不是连续的函数 无法计算SGD 如果分开来train 就需要人为从NAT中统计数据和目标序列长度再train classification模型)
>    2. 另一种方法是 让输出中还包含end标志 只读begin到end的内容
>
>    NAT的好处:
>
>    1. 最大的好处就是可以并行计算 快
>
>    但是NAT实际效果比AT要差
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241011152517689.png" alt="image-20241011152517689" style="zoom:50%;" />
>
>    这里介绍cross-attention的方法
>
>    **无论对什么attention 都是当前q乘上对应位置的k得到   当前位置对于对应位置的注意力** 
>
>    **再将当前位置对于对应位置的注意力乘上对应位置的v的所有结果加和得到当前位置输出(其实当前位置对于对应位置的注意力就是权重 加和就是总输出 类似于评价矩阵)(attention本质就是用q为主导k为底层算k位置数据对q位置数据的权重 而v代表处理后的数值 k基的权重乘上k位处理后的数值v加和得到q位置的评价数据)**
>
>    cross-attention就是关注Maskd self-attention计算出的结果在Encoder 的输出下的权重输出 输入n个Masked 输出n个b(以Encoder输出为决策)
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241011155007190.png" alt="image-20241011155007190" style="zoom:50%;" />
>
> 7. 下图的方格代表该处声音vector的注意力权重
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241011163904654.png" alt="image-20241011163904654" style="zoom:50%;" />
>
> 8. 下图是train的细节 
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241011164228379.png" alt="image-20241011164228379" style="zoom:50%;" />
>
>    注意这里使用了一种叫做teaching force的技巧 Decoder的输入和输出都使用正确答案
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241011164257616.png" alt="image-20241011164257616" style="zoom:50%;" />
>
> 9. 训练的一些tips
>
>    Copy Mechanism:即机器会学习将输入的一些特殊的信息在输出直接复制
>
>    Guided Attenton: In some tasks, input and output are monotonically aligned.(简单理解就是输入和前一次输入要强相关) 引导模型在训练时 前一级的权重最好是最大的
>
>    Beam Search: 首先引入概念Greedy Encoding(贪心算法)
>
>    也即最好的路(总最大可能性概率和最大)不一定是每一步都选择当前最大的概率
>
>    在有标准答案的任务 使用Beam Search效果一般比较好
>
>    但是在需要创造力的场景(比如说语音合成或者文字生成) 模型训练好后 测试集一般需要加一些随即性
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241011165426338.png" alt="image-20241011165426338" style="zoom:50%;" />
>
> 10. 这里引入Blue score
>
>        存在几层逻辑
>
>        第一层:无论层和层之间是怎么连接的 只要连接的方式是连续的函数(可微分) 并且Loss的形式可微分 就一定可以简单粗暴的用SGD训练 并且底层逻辑也是简单的前向传播 函数梯度运算 反向传播
>
>        第二层:如果Optimizing Evaluation Metrics使用的是不连续的函数形式 无法微分 就不能使用SGD进行简单粗暴计算
>
>        第三层:语言模型训练最终的评估一般使用 Blue scroe 
>
>        这里简要介绍Blue score 提出其的目的是防止机器使用CrossEntropy作为Optimizing Evaluation Metrics作为模型评估时 故意输出较短的句子来获得高精确率 Blue引入了简短乘法 而这个简短乘法本身是不连续的函数 所以无法用其作为criterion来进行SGD
>
>        第四层:训练时使用CrossEntropy训练多个模型参数(应该就是一个模型训练过程中不同阶段的参数) 再使用Blue score进行选取
>        如果遇到Optimization无法解决的问题 把希望使用的metrics当作RL的reward 把Decoder当作agent 就可能可以硬train出来
>
>        <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241011170234539.png" alt="image-20241011170234539" style="zoom:50%;" />
>
>        11. 如果训练的时候 一直给正确的样本 在真实的输出下 如果有一步输出了错误的样本 就会导致error propagation 此时就可以人为的给一些错误feature
>
>            这叫做Schedule Sampling
>
>        12. 总结:
>
>            对卷积核 self-attention视作读取信息的方式 
>
>            对FC cross-selfattention 使用信息的方式
>
>            对Transform CNN 包含读取信息和使用信息的整体框架 中间会包含类似residual    Batch Norm(Sample输入前 对features内部的一个feature类内部进行Norm操作 因为同类) Layer Norm(self-attention的输出进行归一化操作 因为同类(进过self-attention输出后 每一个输出都包含输入信息))
>            
>            Transform是可变输入Encoder, Encoder输出与可变输入Decoder Decoder输出可变(但存在end) 这里其实存在一个缺陷没有将清除 输出和输入的个数被强制绑定了
>

### Deep Reinforcement Learning(RL)

> 1. 提出了之前的框架基本都是supervise learning 自监督学习是机器自己给label 自编码器也类似
>
> 2. RL针对不易给定label的场景(比如说下棋) 但是从外界得到好坏程度的"reward"
>
> 3. RL内部存在一个actor 外部有一个environment environment给出observation(input)作为输入到actor内部得到action(output) 同时environment会根据action做出改变 同时给出actor一个reward
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015161443387.png" alt="image-20241015161443387" style="zoom:40%;" />
>
> 4. 提出RL和machine learning一样的三个步骤
>
>    ​	**这里经过学习之后对RL的框架有了一个更深的理解 暂时只考虑深度强化学习 使用深度学习对状态量(也就是observation)输出期望的action** 
>
>    ​	**先引入一个关于r存在映射关系的Q 已知可以使用Q搭建Loss函数**
>
>    ​	**这时一般会对action作用在环境后 求取状态量 利用此时的状态量求取reward(暂时不考虑G的一些优化总总) 但是这里的reward和模型参数就没有映射关系了(参数和action有映射关系 当前的action已经作用action后的状态之间是没有映射关系的) 所以这里不能简单的计算梯度 即使已经设定了loss函数**
>
>    ​	**所以这里只需要引入a-r的梯度计算(见下大标题分析) RL的整体计算框架就搭好了**
>
>    step1:find a function
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015161758932.png" alt="image-20241015161758932" style="zoom:40%;" />
>
>    这里跟分类很像 只不过分类的选择是动作
>
>    这里提到一嘴(输入的可以是游戏画面 这个更加复杂 使用的可能是游戏内的一些现存状态)
>
>    最后机器选择的action根据数据的几率分布输出 但是这里的输出是sample(所有行为的概率综合) 信息量比给出最优行动更好 可以采取非贪心算法的其他优化方法(比如说beam search)
>
> 5. step2:Define Loss
>
>    下图进行的操作是 根据observation(本为系统的状态量) 输入得到行为输出 输出作用于环境 环境根据目的给出reward 一场游戏结束(die or win)得到总的reward叫做epoch 多个epoch累加后的reward叫做return 
>
>    对于return取负号就是传统意义的Loss
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015162426636.png" alt="image-20241015162426636" style="zoom:40%;" />
>
> 6. step3:optimaztion
>
>    Optimization的过程就是找network的参数 让reward越大越好
>
>    这里有几层的点：
>
>    1.输入到actor内部 输出具有随机性(跟sample有关 可能使用了其他优化算法)
>
>    2.环境和reward都可能是随机的 比如给定action 环境进行的改变可能不一样 
>
>    给定action作用在obseration上 得到的reward也可能不一样(不正确的位置开火就没有分数)
>
>    3.reward是个规则 无法使用SGD
>
> 
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015162820875.png" alt="image-20241015162820875" style="zoom:40%;" />
>
> 7. ###### 这里引入传统的对行为的处理过度到reward
>
>    ​	同时引入控制actor一定不进行某个行为 只要把Loss取反即可
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015164043248.png" alt="image-20241015164043248" style="zoom:40%;" />
>
>    ​	这里将正e和负e联合 表示针对这个情景偏向左不偏向右(结果是输出左>设计>右)
>
>    ​	这里相当于给样本的label多个hot-code 每一个hot-code都有一个e 对这个e取最大或者最小来得到最希望原理还是最希望接近来进行训练(这样的好处是可以区分当前样本下数据的优先级 而原本的one hot-code一次输入带有的信息只有最希望接近的类别)
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015164209723.png" alt="image-20241015164209723" style="zoom:40%;" />
>
>    这里给定输入s1输出a1的评价 
>
>    这里的疑问是怎么还是采用e作为评断Loss的标准 因为e已经没办法求了
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015164433197.png" alt="image-20241015164433197" style="zoom:40%;" />
>
>    这里没看懂有几层(这里是传统深度学习到RL的过度 所以e是什么 A是什么 eA乘积什么意义都没有决定性的意义)
>
>    1.A代表什么？reward？
>
>    2.为什么A乘上e e不是Entropy吗
>
>    3.A本应是描述同一个预测输入和不同hot-code entropy的权重 但是这里放分散的放在了多对输入和输出的后面 是什么意思？难道这里的多e不是同一个预测值和多个目标hot-code的结果 而是每一个输入得到输出的评价？(应该是在这样的)
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015170506269.png" alt="image-20241015170506269" style="zoom:40%;" />
>
> 8. 探讨A(**这里A就已经代表reward的有梯度的表示了 这里A本质就是reward**)
>
>    下图是求解A的第一个版本
>
>    ​	假设存在一个随机的Actor存在可以供我们optimize 让这个Actor随机跑多个epoch 得到的数据之后(也就是蓝框左) 认为的标注右框A 再进行训练(这一步怎么训练没有给出)
>
>    ​	这个求解方式存在缺陷:
>
>    1. 没有Reward Delay:也即当下的行为保证对当下或者很近的结果
>    2. 没有考虑上一步的行为之后reward的影响
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015171348862.png" alt="image-20241015171348862" style="zoom:40%;" />
>
>    下图是处理A的第二个版本
>
>    ​	使用中间变量r代替version0中的A r代表当前行为的初步判断(r只保证当前或者很近的action评价)当前行为的A是当前行为之后所有的reward的总和
>
>    ​	但是这样存在一个缺陷 很远很远的reward可能和当前的a就没有什么关系了 但是这里仍然不加权重的考虑reward
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015172113053.png" alt="image-20241015172113053" style="zoom:40%;" />
>
>    下图是处理A的第三个版本
>
>    ​	考虑reward作用在之前的action的时候存在衰减 越远衰减的越厉害
>
>    这里中间有人提问说这个过程有点像蒙特卡罗:存在大量随机操作后得到的样本 根据这些样本对模型进行处理求解
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015171853890.png" alt="image-20241015171853890" style="zoom:40%;" />
>
>    下图是处理A的第四个版本
>
>    ​	这里提出情景 如果r都是大于10(比较重要的概念了 r是联系实际情景定下的)
>
>    但是这样得到的G也好A也好都会大于0 也就是无论怎么样的行为 model都会鼓励
>
>    ​	所以人为的设定baseline 让G有正有负
>
> 
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015172517001.png" alt="image-20241015172517001" style="zoom:40%;" />
>
> 9. Policy Gradient
>
>    ​	这里提出一个观点 资料的obtain本身在训练的过程中()
>
>    问题:这里L怎么求梯度？---这个问题的核心在于A是否能用输入输出的集合得到---这个观点错了 这个问题的核心是action-r的梯度怎么求 而A是可以从各个步骤的状态量求取出来的 s-r(r是关于s的一个映射)-
>
>    问题:这里为什么还考虑e？---这里的e本质是评估的形式 具体可以选择为当前步的负log值
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015173407751.png" alt="image-20241015173407751" style="zoom:40%;" />
>
>    ​	规定:每一次搜寻的资料 只能训练一次 只能用来训练训练资料所使用的模型的参数
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015173648983.png" alt="image-20241015173648983" style="zoom: 40%;" />
>
>    ​	这里做了一些解释 因为搜集资料的actor和搜集到的数据就直接对应了 但是将本次得到的数据分批训练了多个模型 存在的错误就是训练的actor可能不会按照训练资料行动 这样训练的结果就不太好
>
>    ​	如果使用off-policy 对同一组数据分批训练得到多个模型 这就需要训练的时候让当下的actor to train知道它和actor to interact的差别
>
>    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241015174610268.png" alt="image-20241015174610268" style="zoom:40%;" />
>
> 10. Critic
>
>     用于评判actor在s下的好坏
>



### The way to understand how to update the parameters of model base on using RL framework

> **问题1:a-s映射可能存在无法求导**---
>
> **问题2:s-r映射可能存在无法求导**---
>
> **问题3:Loss无法确定**
>
> 以上两个问题都客观存在 但是都有解决方法 如下

**策略优化类**(对梯度的求解方式做处理)

> **Policy Gradient**
>
> > 对问题1,2 其采取的策略是不考虑a-s s-r 只考虑 a-r 并且a-r只需要知道映射关系
> >
> > 对问题3 不存在Loss r/G本身作为评估标准但是对每一步的概率取log乘上其G(对r进行处理后的 一定要有正负值)
> >
> > 细节见李课程跟随 这种思想的核心思想就是扩大reward好的行动概率 减小reward不好的行动概率
> >
> >
> > ​	这里的细节操作是不是Loss用输出和其对应的hot-key求解乘上reward 这样如果reward大于0 那就意味着希望在当前状态输入下 实际输出可能的概率更高
>
> **ADH(Advantage-Weighted Hierarchical/Policy Gradient)**
>
> > <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241017184353030.png" alt="image-20241017184353030" style="zoom:70%;" />
>

**价值函数类**(引入Loss处理)

> TD
>
> Q-learning
>
> SARSA

**混合策略-价值类**(又对梯度求解方法做处理 又引入loss)

> Actor-Critic
>
> Deep Deterministic Policy Gradient(DDPG)









skip:p51(transform扩展)

