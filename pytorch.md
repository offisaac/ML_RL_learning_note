1. 数据加载intro

   Dataset

   > Dataset:类似于Series 提供一种方式获取数据和label 是pytorch库中的一
   >
   > 个抽象类，Pytorch中的数据集都应该继承这个类并实现它的两个核心方法
   >
   > 1. \__len__() 这个方法返回数据集的大小 即样本数量 值得注意的是 外部调用的时候使用的是len()这个函数
   >
   > 2. \__getitem\_\_() 这个方法获取内部data和label 值得注意的是  使用\_\_下划线__这种“魔法方法”，外部在调用时 是使用方括号的形式、
   >
   >    由上总结一点 下划线方法在外部有宏观接口 内部抽象定义 类似于抽象库的顶层和底层 \_\_init\_\_的顶层就是()这种构造函数
   >
   >    python名称重要的另一个instance
   >
   >    ```python
   >    import torch
   >    from torch.utils.data import Dataset#本身这里就可以直接使用Dataset了 额外引入一步只是不想使用那些前缀
   >    class CustomedDataset(Dataset):
   >        def __init__(self,data,labels):
   >            self.data=data
   >            self.labels=labels#这里self使用的是父类Dataset的数据
   >    class __len__(self)
   >    	return len(self.data)#len可以对data操作 说明data本身也内置__len__()
   >    class __getitem__(self,index):
   >        sample=self.data[index]#[]可以对data操作 说明data本身也内置__getitem__() #这里没有使用self.sample 说明其是临时变量
   >        label=self.labels[index]
   >        return sample,label#返回元组
   >    ```
   >
   >    另一个实例
   >
   >    ```python
   >    import torch
   >    from torch.utils.data import Dataset
   >    from PIL import Image
   >    import os
   >    import os #python中关于操作系统的库 可以实现和系统的交互
   >    # print(torch.__version__)
   >    # print(torch.cuda.is_available())
   >    # print(torch.randn((2,3)))
   >       
   >    class  MyData(Dataset):
   >        # def __init__(self,root_dir,label_dir):#这里输入1.外层文件夹地址 2.内层文件夹名称(label)#这就是形象的处理方法
   >        #     self.root_dir = root_dir
   >        #     self.label_dir = label_dir
   >        #     self.path = os.path.join(self.root_dir,self.label_dir)#得到内层文件夹路径
   >        #     self.img_name_list =os.listdir(self.path)#得到内层文件夹条目列表
   >        def __init__(self, path):#传入目标文件夹地址
   >            self.path = path
   >            self.img_name_list=os.listdir(path)
   >       
   >        def __getitem__(self,index):
   >            img_path=self.img_name_list[index]
   >            img_item_path=os.path.join(self.path,img_path)
   >            img=Image.open(img_item_path)#通过路径返回图片类型对象(包含图片的各种特诊)
   >            label=self.img_name_list[index]
   >            return img,label
   >       
   >        def __len__(self):#长度返回目标文件夹的条目数目
   >            return len(self.img_name_list)
   >       
   >    #总结 init函数输入的是外层文件夹地址和内层文件/文件夹名称 内部还得到内层文件夹内部的图片名称列表(通过os.path.join()进入内层文件夹 再"读取"得到内层条目名称)
   >    # getitem函数输入的是index 返回的是内层文件夹中第index个图片和名称 方式是index读取路径列表得到对应名称 使用os.path.join()方法得到路径 使用open函数打开路径返回图片对象
   >       
   >    #更改后的代码思路 传入目标文件夹路径 得到文件夹内部条目名称列表 index读取对应名字 和目标文件夹路径一同合成图片文件夹路径 通过open函数返回图片对象
   >    img_path = r"E:\code for py\pytorch_learning_note\dataset\cat\cat-4756360_640.jpg"
   >    img = Image.open(img_path)
   >    img.show(command='start')
   >       
   >    cat_dataset=MyData(r"E:\code for py\pytorch_learning_note\dataset\cat")
   >    print(len(cat_dataset))
   >    print(cat_dataset[1][1])#返回第二张图片的名称
   >    print(cat_dataset[1][0])#返回第二章图片对象
   >    print(cat_dataset[2])
   >       
   >    dog_dataset=MyData(r"E:\code for py\pytorch_learning_note\dataset\dog")
   >       
   >    animal_dataset=cat_dataset+dog_dataset#相当于叠合了 dog数据集index紧跟着cat
   >    ```
   >
   > 
   >
   > 通过继承Dataset，可以自定义处理各种形式数据的逻辑(图像，csv，datapool(数据库))，并且可以在\_\_getitem_\_()方法中对数据进行预处理

   Dataloader

   > Dataloader:将数据集打包 
   >
   > ```python
   > DataLoader =DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)
   > 其中 
   > shuffle表示在每一个epoch时是否对sample进行打乱(只要从头开始遍历就会打乱, 而不是遍历完才会打乱)
   > num_workers表示处理数据时使用的子线程个数（给0则只使用主线程）
   > drop_last表示在对sample分batch时 如果无法整除 是否使用余留的项
   > ```
   >
   > Dataloader返回可迭代对象用于batch
   >
   > ```python
   > def MyDataset(Dataset):
   >  def __init__(self,xxx):
   >      xxx
   > mydataset=MyDataset(xxx)
   > dataloader=Dataloader(mydataset,batchsize=,shuffle=)
   > criterion=n.MSELoss()
   > optimizer=torch.optim.SGD(model.parameters(),lr=,decay=,momentum=,)
   > criterion=nn.MSELoss()#这一步记住是nn,MSELoss() 其内没有参数
   > for i in epochs:
   >  for data1,data2...in dataloader:
   >      outputs=Model(Xs)
   >      Loss=criterion(outputs,Ys)
   >      optimizer.zero_grad()
   >      Loss.backward()
   >      optimizer.step()
   > ```
   >
   > 

   Grid 

   > Grid：用于输出打包后的batch
   >
   > ```python
   > img_grid = torchvision.utils.make_grid(imgs)
   > ```
   >
   > 

2. Tensor(张量)

   Definition

   > Pytorch中，Tensor是一个多维数组，类似于Numpy中的ndarray 可以支持GPU加速计算能力 和自动微分 是PyTorch的核心数据结构 
   >
   > 
   >
   > 对Tensor数据打印时 形式是---Tensor(二维list)

   一些attribution

   > shape 表述tensor是几维数据 并且给出各个维度下的参数(不同对象转化tensor后各个维度下参数意义不同)
   >
   > 例如 对于数组就给出各个维度下的数组内个数 对于PIL转化tensor后对象则给出CHW参数等等

3. Torch创建Tensor的方式

   > torch.randn((3,3))#创建3x3的Tensor

4. TensorBoard的使用

   > TensorBoard是一个用于可视化和监控机器学习训练过程的工具 可以记录数据 实现数据的可视化

   引入库

   > from torch.utils.tensorboard import SummaryWriter
   >
   > 注意 这里的 tensorboard子模块实际上和torch是独立的 

   创建writer变量

   > writer =SummaryWriter(address, or "log_name")#使用后者会自动在工程文件夹内创建log_name名的的文件夹
   >
   > or
   >
   > with Summary_Writer(address or "log_name") as write:
   >
   > ​	operations....

   添加数据

   > writer.add_scalar("y=2x",y,x)#第一项是log_name文件夹下文件的log 后两项分别是对应的y x值

   添加图片

   > writer.add_image("log_name",tensor or ndarray,global_step,dataformats='HWC')
   >
   > 其中tensor or ndarray先使用Image.open(path) 返回图片对象 再使用np.array(Img)转化图片对象
   >
   > global_step是图片在tensorboard内的标签
   >
   > "HWC"是数据格式
   >
   > 转为tensor的格式就是默认格式 转为array的需要格外修改

   打开TensorBoard

   > 控制台输入tensorboard --logdir="address" or log_name --port=xxxx
   >
   > 这里格外注意一点 打不开路径时就使用相对路径 并且r在这里没有用 r在py语法中是忽略转义字符 但是在命令行中没有意义
   >
   > 并且只要打开后，对log文件夹内存在修改时，不可简单刷新界面查看修改
   >
   > 并且 对log文件写入行为时是叠加在以前的写入的

   tag和global_step

   > tag将数据分成不同组
   >
   > global_step将同组的数据进行编号 
   >
   > 比如说图片1 tag1 global_step=5 图片2 tag1 global_step=10 图片3 tag2 global_step=3
   >
   > 图片1和图片2就会分为一组 标号分别是5，10 图片三自成一组

5. transform

   > 用于转在图像处理或者数据预处理中对数据进行转换操作，可以将数据从一种形式转换成另一种形式
   >
   > 其内方法的时候一般是通过函数返回相应功能对象 使用\_\_call\_\_魔法函数使这个对象可以像函数一样实现对应的功能

   引入库

   > from  torchvision.transforms import transforms

   ToTensor方法返回对象

   > ToTensor can convert a "PIL Image" or "numpy.ndarray" to Tensor while uniformize the RPG values
   >
   > ```python
   > tensor_trans=transforms.ToTensor()
   > tensor_img=tensor_trans(img)
   > #注意 使用ToTensor获取的格式是C,H,W 而np.array()方法获得的是H，W，C
   > ```

   Resize方法返回对象

   > Resize can change the shape of the pictures(的确是修改尺寸 不裁剪 只伸缩)
   >
   > ```python
   > size_trans=transforms.Resize((256,256))
   > img=size_trans(img)
   > ```

   Normalize方法返回对象

   > 之前的方法输入值都是PIL图像或者numpy 这里需要tensor
   >
   > Normalize can change RPG values by formula $output=\frac{input-mean(均值)}{std(标准差)}$​
   >
   > ```python
   > img_tensor=transforms.Totensor()(img)
   > img_Normalized=transforms.Normalized(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
   > ```

   Compose方法返回整合对象

   > 可以向Compose输入上述方法返回对象，新返回的对象会按顺序操作
   >
   > ```python
   > transform_Composed=transforms.Compose([transform.Resiez(),transform.Totensor(),transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   > ```

   ImageFolder作用于数据集

   > ```python
   > transform_Composed=transforms.Compose([transform.Resiez(),transform.Totensor(),transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   > dataset=ImageFolder(root="",transform=transform_Composed)
   > ```

6. torchvision中数据集的使用

   引入库

   > import torchvision

   调用数据集

   > ```
   > torchvision.datasets.dataset_name(root="",train=,download=)
   > #root 下载的根地址
   > #train 使用数据集的训练集部分还是测试集部分
   > #download 是否下载
   > 
   > 注：下载数据集也可使用外部下载的方式 在程序目录下创建dataset文件夹(和root同名即可) 将下载的数据集直接复制到该文件夹即可(可能需要解压缩)
   > ```

   处理数据集

   > ```python
   > dataset_transform=transforms.Compose([transforms.ToTensor,
   >                                    ...
   >                                   ])
   > dataset=dataset_transform(dataset)
   > ```

7. 神经网络基本框架 nn.Module

   > 对神经网络的计算类 都需要继承这个大类 类似于Dataset
   >
   > 同继承Dataset 类内部必须定义\_\_init\_\_ \_\_getitem\_\_ \_\_len\_\_  此类内部也必须定义\_\_init\_\_  forward等
   >
   > ```python
   > class my_Module(nn.Module):
   > def __init__(self):
   >   super().__init__()#or super(my_Module,self).__init__()后者为python2写法#这里是必要的
   >   operation...
   >   self.conv1=nn.Conv2d(1,20,)
   > 
   > ```

8. 线性预测的实例代码

   > ```python
   > import torch
   > import torch.nn as nn#form torch import nn 前者写法更简单
   > import matplotlib.pyplot as plt
   > 1.引入参数
   > X=torch.tensor([,],[],...)
   > Y=torch.tensor(...)
   > 2.创建对应类
   > class LinearRegressionModel(nn.Module):
   > def __init__():
   > super().__init__()#调用父类初始化
   > self.linear=nn.Linear(2,1)#输入参数有两个 输出参数有一个 这里创建了一个线性全连接层 返回给类的linear属性 这个全连接层输入参数返回前向的数据(即__call__进行参数计算)
   > 	def forward(self,x):
   > return self.linear(x)
   > 3.实例化参数对象    
   > model = LinearRegressionModel()
   > 4.定义损失函数和优化器
   > criterion = nn.MSELoss()#使用MSE的Loss 返回类
   > optimizer = torch.optim.SGD(model.parameters(),lr=0.01)#Stochastic(随机的) Gradient Descent 
   > #对这一步的解析：
   > '''
   > 1.model.parameters()返回model内的可迭代对象 linear为自己定义的 其会自己寻找是否内部有变量是nn内部的可学习参数可迭代对象
   > 2.lr是学习率
   > 3.这一步将model和优化器绑定 后续优化器的操作可以直接改变参数值 奇怪在优化器这种不应该内置在nn.Module内部吗 还要外部绑定
   > '''
   > 5.训练模型
   > epoch = 1000
   > for epoch in range(epochs):
   >  #前向传播
   >  outputs = model(X)#实际通过__call__调用重写后的forward
   >  loss = criterion(outputs,Y)
   > 
   >  #反向传播和优化
   >  optimizer.zero_grad() #梯度清零 这一步的必要性在于下一步计算梯度实际上是将计算后的梯度累加到上一次的梯度(这和提高计算效率的底层有关)
   >  loss.backward()#反向计算梯度
   >  optimizer.step()#更新参数
   > 6.预测和可视化
   > predicted = model(X).detach().numpy()#预测结果
   > fig,ax=plt.subplots()
   > ax.scatter(X[:0,0].numpy(),Y.numpy(),label="Original Data")
   > ax.plot(X[:0,0].numpy(),pridicted,label="Fitted line")
   > ax.set_xlable("feature")
   > ax.set_ylable("y")
   > ax.legend()
   > plt.show(fig.number)
   > 
   > 
   > 
   > 
   > 
   > ```
   >
   > 由本实例:
   >
   > 1. 线性层的局限性，无论多少层最后都是线性的 无法拟合复杂曲线
   > 2. 训练到一定程度后，图像不变化 原因是已经训练到极限 找到了线性下最小的Loss
   > 3. 使用双层线性之后
   >
   > 猜想：是否可以使用大类加list实现多层---不是 直接在原始类定义多层 在forward前后调用 loss会通过计算图得到反向计算的各个层连接
   >
   > ```python
   > X = torch.tensor([[1.0, 8.0], [2.0, 3.0], [3.0, 1.0], [4.0, 10.0], [5.0, 11.0],
   >                   [6.0, 7.0], [7.0, 6.0], [8.0, 10.0], [9.0, 11.0], [10.0, 8.0]])
   > Y = torch.tensor([[1.0], [5.0], [3.0], [11.0], [15.0], [11.0], [10.0], [17.0], [19.0], [21.0]])
   > 
   > class LinearFittingModel(nn.Module):
   >     def __init__(self):
   >         super(LinearFittingModel,self).__init__()
   >         self.linear1 = nn.Linear(2,200)
   >         self.relu=nn.ReLU()
   >         self.linear2 = nn.Linear(200, 200)
   >         self.linear3 = nn.Linear(200, 200)
   >         self.linear4 = nn.Linear(200, 1)
   >         self.sigmoid=nn.Sigmoid()
   >     def forward(self,Input):
   >         x = self.linear1(Input)
   >         x = self.relu(x)
   >         x = self.linear2(x)
   >         x = self.relu(x)
   >         x = self.linear3(x)
   >         x = self.relu(x)
   >         x = self.linear4(x)
   >         # x = self.sigmoid(x)
   >         return x
   > ```
   >
   > 通过多层训练：
   >
   > 1. 最开始只用了两层 并且只给了20的大小 训练一直loss在1.8左右
   > 2. 后来用了4层 给了20大小 loss还是在1.8左右
   > 3. 给了200大小 loss训练到0左右
   >
   > 问题：使用sigmoid或者ReLU处理输出数据 Loss为定值--->输出为概率等才使用sigmoid作为输出的激活函数 但是为什么--->对于sigmoid 其只输出0-1 大于1的数据永远无法拟合 对于ReLU 其只输出非负数 负数无法拟合
   >
   > 扩充到batch计算：
   >
   > ```python
   > X = torch.tensor([[1.0, 8.0], [2.0, 3.0], [3.0, 1.0], [4.0, 10.0], [5.0, 11.0],
   >                   [6.0, 7.0], [7.0, 6.0], [8.0, 10.0], [9.0, 11.0], [10.0, 8.0]])
   > Y = torch.tensor([[-20.0], [5.0], [3.0], [11.0], [15.0], [11.0], [10.0], [17.0], [50.0], [100.0]])
   > dataset=TensorDataset(X, Y)#这里规定了dataloder迭代器返回几个元素
   > dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
   > for batch_inputs, batch_targets in dataloader:#使用dataloder来调用平行计算 以及进行batch的区分 不同的epoch是否shuffle等等
   >     # 前向传播
   >     outputs = model(batch_inputs)
   >     loss = criterion(outputs, batch_targets)
   > ```
   >
   > 一些可能的经验:
   >
   > 1.学习率可以通过先给小 看Loss变化是否单调以及实际收敛速度的快慢来决定学习率的值
   >
   > 2.输出直接返回nan 就是学习率太大了

9. ### The comparasion between ReLU,Softplus and sigmoid

   **现象 :**

   > ​	对于同一份回归实验数据 sigmoid可能无法完成拟合 relu可以完成但是训练久 softplus效果最优秀 
   >
   > ​	relu作为激活函数无法完成的拟合 输出作softplus处理后可以完成拟合 但是softsign作为激活函数本身可以收敛 输出加上softplus就无法收敛
   >
   > ​	学习率给大 收敛速度不一定变快 如果一直Loss跳变 学习率可以适量给小

   **解释**

   > **梯度更新和训练速度：**
   >
   > ​	ReLU只对正数有梯度 梯度大小固定1 梯度不会衰减 	
   >
   > ​	但是ReLU在负值区域的梯度为0 可能导致死神经元的问题 一些神经元无法更新(大意为反向传播值为0 对参数无贡献)
   >
   > ​	SoftPlus是ReLU的平滑版 梯度连续 输入负值仍然有梯度 有更高的精度 但是更难计算
   >
   > **实际的效果:**
   >
   > ​	使用ReLU Loss会跳变 在某一个数据点突然收敛 使用Softplus Loss比较平稳 但是速度确实慢一点点 
   >
   > ​	ReLU作用在输出Loss一般不收敛 而SoftPlus会对收敛有帮助

10. ### 归一化(Normalize)

    > **得到归一化数据**
    >
    > ```python
    > # 将所有数据堆叠在一起
    > train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    > 
    > data = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
    > 
    > # 计算每个通道的均值和标准差
    > mean = np.mean(data, axis=(0, 1, 2)) / 255.0
    > std = np.std(data, axis=(0, 1, 2)) / 255.0
    > 
    > ```
    >
    > **归一化操作**
    >
    > ```python
    > transform = transforms.Compose([
    > transforms.ToTensor(),
    > transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    > ])
    > #!!!这里ToTensor()进行归一化操作(将所有量除最大最小值偏移 这里totensor检测数据类型为uint8_t 所有数据除以255)
    > #Normalize进行标准化操作 减去平均值除以方差
    > 
    > or
    > 
    > from sklearn.preprocessing import StandardScaler
    > feature_scaler = StandardScaler()
    > feature_scaler.fit_transform(X_train)
    > ```
    >
    > **归一化好处**
    >
    > > ​	梯度下降算法依赖于特征的数值范围。如果特征的取值范围差异较大，梯度下降的路径会变得弯曲不平，这将导致收敛速度减慢甚至无法收敛。通过归一化，数据的尺度得到一致处理，梯度下降能够在更平滑的曲面上运行，从而更快地找到最优解。
    > >
    > > 
    > >
    > > ​	许多算法（例如K近邻、支持向量机、逻辑回归等）对输入特征的尺度非常敏感。如果不进行归一化，某些数值较大的特征会在计算距离或相似度时占据过多权重，导致模型无法有效学习其它特征的贡献。
    > >
    > > 
    >
    > **实践中遇到的问题**:
    >
    > 1. **训练效果好 实际效果不好---**
    >
    >    1. 自定义的数据集中 实际返回的数据是self.X=dataset.data 绕过了原始数据
    >
    >    2. transform为了节省内存 只有在使用到数据时才会将该数据转化为张量(实测将50000的CIFAR10数据进行一定transform操作后(在init内使用tensor.totensor(dataset))会占用49G内存) 
    >
    >       nn在设计时 之后在原dataset(自定义或者原始)的\_\_gettiem_\_内才会transform这个dataset的individual的全部数据(调用哪个处理哪个)
    >
    >    3. 测试时没有用到Loss 没有使用\___gettiem_\__ 就没有transform 本质self.X还是nparray
    >
    > 2. **怎么从整体调用数据?**：
    >
    >    ​	整体类调用数据有两种方式 第一中使用.直接调用其内变量 第二种 使用[]方法调用内部\_\_getitem\_\_的返回值 
    >
    >    ​	如果想使用transform 一定要使用后者！！！！
    >
    >    ****
    >
    > 3. **对数据调用的实例**
    >
    >    ```python
    >    print(train_data.data[100][11][11])#这里访问的是data变量[batch,H,W,C]
    >    print(train_data[100][0][0])#这里访问的是__getitem__返回的image变量[[100][0]这里访问到iamge image内部是[C,H,W]]
    >    print(train_data[100][0][0][11][11],train_data[100][0][1][11][11],train_data[100][0][2][11][11])#此数据为上数据的归一化
    >    ```
    >
    > 4. **为什么训练的时候调用了transform？？？**:
    >
    >    存疑
    >
    > 5. **将X在init的时候接受时就转化成张量** 
    >
    >    1. permute和view的使用
    >
    >       ​	permute用于将tensor的多维度数据更改存放顺序 比如说通道C在前很难调用一个pixel的三通道数据(调用三次,每次一个像素) 但是C放在最后就很好调用了
    >
    >       ​	view用于将tensor的多维度数据修改维度
    >
    >       ```python
    >       tensor = torch.randn(2, 3, 4)  # 形状为 [2, 3, 4] 的随机张量
    >       reshaped_tensor = tensor.view(-1, 4)  # 将形状调整为 [6, 4]
    >       print(reshaped_tensor.shape)  # 输出：torch.Size([6, 4])
    >       ```
    >
    >       ​	permute对应张量的shape 原本使用.permute(2,0,1)可以 是因为data取了index不用考虑batch了 而对整个数据集进行permute操作要考虑batch 也就是函数入口参数要有四个
    >
    >    2. 一次性处理太多数据 占用太大内存(实际的训练 底层是用到哪些数据再将哪些数据转为张量 这也就是为什么只有经过\_\_getitem\_\_才会进行transform的处理)
    >
    > 

11. ### 使用gpu加速

> **正常训练**
>
> > ```python
> > 1. 创建设备 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
> > 2. 创建模型并移动到设备上 model = CNNClassifier(my_train_data.class_num).to(device)
> > 3. 将数据移动到设备上 
> > 
> >    for X, Y in my_train_data_loader:
> >            X, Y = X.to(device), Y.to(device)  # 将数据移动到设备上
> > 4. 模型检验
> >    with torch.no_grad():
> >        for data, labels in my_test_data_loader:
> >            data, labels = data.to(device), labels.to(device)  # 将测试数据移动到设备上
> > ```
> >
> > 
>
> **模型加载**
>
> > ```python
> > 1. 创建模型并移动到设备上 loaded_model = CNNClassifier(class_num=10).to(device)
> > 2.加载模型并移动到设备上loaded_model.load_state_dict(torch.load(f'./model_parameter_set/model_parameter_20', map_location=device))
> > ```

12. ### 模型优化

> 1. **对数据的处理**
>
>    正则化
>
>    > Batch Norm 
>    >
>    > ```python
>    > nn.BatchNorm2d(num_features,...)#输入通道数 输出和输入通道数相同
>    > ```
>    >
>    > Vector Norm
>    >
>    > residual network
>
>    Batch数值
>
>    > 修改batch数值来修改训练效率 overfitting等等效果
>
> 2. **优化器的选择**
>
>    > SGD
>    >
>    > RMSprop
>    >
>    > Adam
>
> 3. **模型**
>
>    模型的选择
>
>    > FC
>    >
>    > CNN
>    >
>    > Transform
>    >
>    > ...
>
>    模型的参数大小
>
>    模型层数
>
> 4. **训练策略**
>
>    Early Stop
>
>    Dropout

13. ### 自定义Loss

    **实现的基础**

    > PyTorch的自动微分引擎**autograd**可以跟踪张量的运算历史,所以只要牵扯张量的运算,全部都是练一块的,信息都是可以访问的,所以核心是运算,而不是函数形式

    **example:**

    > CrossEntropy
    >
    > 一些背景提要 nn中CrossEntropy对运算做了简化 从原来的每个目标值和对应的softmax输出相乘取和 变成只看为1的label 两者本质相同
    >
    > <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241010203636472.png" alt="image-20241010203636472" style="zoom: 67%;" />
    >
    > ```python
    > def custom_Cross_Entropy(logits,target):
    >  softmax_output = F.softmax(logits,dim=1)#dim=1代表有两个维度 这里单个logit就是一个维度了(len=10) logits又是另外一个维度 同时 这里也介绍了F的一个用处 其内定义好很多基础函数
    >  log_softmax_output = -torch.log(softmax_output)#转化为log形式数据 返回张量对象
    >  batch_loss = log_softmax_output.gather(1,targte.unsqueeze(1)).squeeze(1)#squeeze用于修改向量维度unsqueeze表示产生序列1维度 squeeze表示消去序列1维度 底层操作其实就是将行向量变成列向量 gather第一个参数表示对数据的列进行操作 本身代表通过给定给定数据序列 来选取每一列中的第几个序列元素
    > loss = batch_loss.mean()#对列向量求均值   
    > return loss
    > ```

14. ### RL

    **Q-learning实例**

    > **环境和行为的搭建**
    >
    > ```python
    > import numpy as np
    > import random
    > 
    > # 定义迷宫环境 (0: 可走的路, 1: 墙壁, 2: 终点)
    > maze = np.array([
    >  [0, 0, 1, 0, 0],
    >  [1, 0, 1, 0, 1],
    >  [0, 0, 0, 0, 0],
    >  [0, 0, 1, 1, 0],
    >  [0, 0, 0, 0, 2]
    > ])
    > 
    > # 状态和动作定义
    > n_rows, n_cols = maze.shape
    > actions = ['up', 'down', 'left', 'right']
    > n_actions = len(actions)
    > 
    > # Q表：记录每个状态下的动作值
    > Q_table = np.zeros((n_rows, n_cols, n_actions))#np.zeros输入的是元组
    > 
    > # 超参数
    > alpha = 0.1  # 学习率
    > gamma = 0.9  # 折扣因子
    > epsilon = 0.1  # 探索率
    > 
    > 
    > # 奖励函数
    > def get_reward(state):
    >  row, col = state
    >  if maze[row, col] == 2:  # 终点
    >      return 100
    >  elif maze[row, col] == 1:  # 墙壁
    >      return -0.2
    >  else:  # 每移动一步扣0.1
    >      return -0.1
    > 
    > 
    > # 判断是否到达终点
    > def is_terminal(state):
    >  return maze[state[0], state[1]] == 2
    > 
    > 
    > # 定义智能体的移动规则
    > def take_action(state, action):
    >  row, col = state
    >  if action == 'up' and row > 0:
    >      row -= 1
    >  elif action == 'down' and row < n_rows - 1:
    >      row += 1
    >  elif action == 'left' and col > 0:
    >      col -= 1
    >  elif action == 'right' and col < n_cols - 1:
    >      col += 1
    >  return (row, col)
    > 
    > 
    > # ε-贪婪策略选择动作
    > def choose_action(state, epsilon):
    >  if random.uniform(0, 1) < epsilon:
    >      return random.randint(0, n_actions - 1)  # 随机选择动作
    >  else:
    >      row, col = state
    >      return np.argmax(Q_table[row, col])  # 选择Q值最大的动作
    > ```
    >
    > **Q-learning算法**
    >
    > ```python
    > n_episodes = 5000  # 训练的回合数
    > for episode in range(n_episodes):
    >  state = (0, 0)  # 初始状态（左上角）
    > 
    >  while not is_terminal(state):#这就是应该epoch 是否结束游戏
    >      # 根据ε-贪婪策略选择动作
    >      action_idx = choose_action(state, epsilon)
    >      action = actions[action_idx]
    > 
    >      # 执行动作，获取下一个状态
    >      next_state = take_action(state, action)
    >      reward = get_reward(next_state)
    > 
    >      # Q学习公式更新Q值
    >      row, col = state
    >      next_row, next_col = next_state
    >      Q_table[row, col, action_idx] += alpha * (
    >              reward + gamma * np.max(Q_table[next_row, next_col]) - Q_table[row, col, action_idx]#这里是为了考虑未来的所有r(max Q本身就可以当作一直选择最优解r的递归)
    >      )
    > 
    >      # 更新状态
    >      state = next_state
    > ```
    >
    > 对上算法的解释
    >
    > 1. Q-learning通过状态的维度(多少的状态多少个维度)和action的维度(所有action在一个维度内)搭建 
    >
    > 2. Q-learning算法的核心在于由state得到action的方式 以及使用Q代替G(Policy Gradient内使用)来评估更新参数
    >
    > 3. Q-learning由state得到action的方式是通过Q表 假设存在一个初始随机的Q表(or zeros)其有features+1维度 调用时Q[state1,state2...]返回的就是actions 通过actions内各个对应值的大小选取最大值的action 以此得到本次的最优行为(此处在训练时存在随机 也即每一次选择都可能会选择随机动作而非最优解)
    >
    > 4. Q-learning算法使用Q代替G(G为Policy Gradient内使用 同时G Q的区别就是Q的计算还考虑了本次状态到下次)来评估更新参数的方式如下图所示
    >
    >    这里存在几层逻辑 
    >
    >    1.Q表只更新当前状态当前维度那一个值的Q
    >
    >    2.r(s,a)是本次的收益 很好理解而后的<img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241018154146417.png" alt="image-20241018154146417" style="zoom: 67%;" />本身实际意义是 通过上	一次留下的Q表和本次的状态得到本次的action 但是这个action使环境变成什么样的s无法确定(比如说给定选定方向但是仍有一定概率偏航) 此时考虑a作用到当前环境得到的新state中对当前Q表取最大值的action代表的Q(Q本身就是reward的一种形式) 再对其做总和 就代表当前的a的所有收益
    >
    >    ​	对上做总结 Q由本次action的收益和本次action后永远选取最优解下的所有收益和 并且考虑衰减 第二项本身是一种地柜
    >
    > 
    >
    >    <img src="C:/Users/Lenovo/Desktop/Typora/pictures/image-20241018150126816.png" alt="image-20241018150126816" style="zoom: 80%;" />
    >
    > 5. 计算逻辑是 从上一次尝试计算出结果后开始(开始就有上一步留下来的状态和最新更新的Q表) 计算本次的动作 得到下一次的预期Q‘和当前的r(不理解就不理解在a r都是本次 使用的Q’是下次 注意这里辨别Q*和Q’ 前者是固定一个对象 只不过值一直在修改 而Q’应该理解成)

15. ### 训练中出现loss=nan

    > 1. 降低学习率
    > 2. 加入梯度限制
    >
    > ```
    > torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)#整体缩放梯度使绝对值最大值未1(整体缩放 小值有影响)
    > torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
    > #强制最大值超过1时 该值等于1(不整体缩放 小值无影响)
    > ```

16. ### 权重初始化

    > ```
    > for param in model.parameters():
    >  if param.dim() > 1:  # 只对多维参数（如权重矩阵）应用 Xavier 初始化
    >      torch.nn.init.xavier_uniform_(param)
    > ```

17. ### 模型在训练集上效果差问题

    > 1. 学习率不够大导致没学习下来
    > 2. 参数本身小 导致使用Mse作为标准时小Loss也代表大偏差
    > 3. loss一直小幅度上升---修改学习率可以解决
    >
    > 数模比赛遇到的问题---数据没有归一化 数据本身小Loss不能正确判断 学习时学习率过小

18. ### RNN And LSTM

    > ```python
    > class LSTMModel(nn.Module):
    > def __init__(self, input_size, hidden_size, output_size, num_layers=2):
    >   super(LSTMModel, self).__init__()
    >   self.hidden_size = hidden_size
    >   self.num_layers = num_layers
    >   # 定义多层LSTM
    >   self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    >   # 定义全连接层
    >   self.fc = nn.Linear(hidden_size, output_size)
    > 
    > def forward(self, x):
    >   # 前向传播LSTM
    >   x = x.unsqueeze(1)  # 添加序列长度维度，形状变为 (batch_size, seq_len=1, input_size)
    >   #!!!!这里实际上强行引入了batch_size信息 因为传入的是一个batch 所以这个维度batch是1
    >   #h0的顺序是num_layers×num_directions,batch_size,hidden_size
    >   lstm_out, (hn, cn) = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size)
    >   # 取最后一个时间步的输出
    >   y = self.fc(lstm_out[:, -1, :])  # 正确索引最后一个时间步的输出
    >   return y#这里y的维度是sample_num 1(计算loss那一步应该会对此处理)
    > 
    > # 初始化模型、损失函数和优化器
    > input_size = len(features)  # 输入特征数量
    > num_epochs=301
    > hidden_size = 64  # LSTM的隐藏层大小
    > output_size = 1  # 输出层的大小（例如预测一个数值）
    > num_layers = 2  # LSTM的层数
    > loss_list=[]
    > model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
    > criterion = nn.MSELoss()
    > optimizer = torch.optim.Adam(model.parameters(), lr=0.000005, betas=(0.9, 0.999), eps=1e-08,
    >                  weight_decay=0)  # 如果没有二阶beta参数 Adam和RMSprop可以认为同效果
    > # 训练循环
    > for epoch in range(num_epochs):
    > model.train()
    > epoch_loss = 0
    > for batch_x, batch_y in dataloader_train:
    >   batch_x = batch_x.to(device)
    >   batch_y = batch_y.to(device)
    >   # 前向传播
    >   outputs = model(batch_x)
    >   loss = criterion(outputs, batch_y)
    >   # 反向传播和优化
    >   optimizer.zero_grad()
    >   loss.backward()
    >   optimizer.step()
    >   epoch_loss += loss.item()
    > loss_list.append([epoch,loss.item()])
    > avg_loss = epoch_loss / len(dataloader_train)
    > if epoch != 0 and (epoch) % 10 == 0:
    >   print(f'Epoch [{epoch}/{num_epochs-1}], Loss: {avg_loss:.4f}')
    >   np.save('./save_data/save_list/save_list.npy', np.array(loss_list))
    >   torch.save(model.state_dict(), f'save_data/save_model_parameter/model_parameter_{epoch}')
    > ```

19. ### 数据维度的理解

    > definition:本质上就是从内到外数组有多少层括号
    >
    > main idea:
    >
    > 1. 对于pytorch 有时数据维度处理会很死板 比如LSTM可以处理[[[5]]]而不能处理[5]
    >
    > 2. 使用unsequenze可以对张量数据在指定维度前插入维度
    >
    >    如果feature只有1个 那么此时转化为张量时甚至连feature维度都没有
    >
    >    使用sequenze可以对张量数据消除无用维度
