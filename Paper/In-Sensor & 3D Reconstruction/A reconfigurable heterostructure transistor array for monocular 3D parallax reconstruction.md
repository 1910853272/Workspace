# 用于单目三维视差重建的可重构异质结构晶体管阵列 半所-王丽丽

![iShot_2025-04-17_21.51.43](https://raw.githubusercontent.com/1910853272/image/master/img/202504172151137.png)

现有光场检测方法要么依赖专门的光源以激发光–物质相互作用（例如利用光学共振），要么需要多孔或多摄像头来捕获反射场，因而集成到 CMOS 工艺中往往既复杂又昂贵，而且多相机系统的校准工作也极为繁琐。

已有一些基于结构光或双目成像的传感器可用于深度场测量，单目条件下实现真正的三维信息重建，仍然面临很大挑战。

本文提出了一种可重构极性转换异质结晶体管阵列 PCHT，可通过电场调制在 n 型、p 型和双极三种工作模式之间切换，从而既能调制输入信号的幅度，也能调制其相位。通过连续二维图像序列中像素的视差变化来推算三维结构，用于单目 3D 信息重建。

# 可重构 PCHT 架构

![iShot_2025-04-17_22.33.14](https://raw.githubusercontent.com/1910853272/image/master/img/202504172233067.png)

a.可重构PCHT阵列与芯片的示意图

b.异质结拉曼光谱映射

c.3D视差重建原理。

d.输入电压与电输出变化。灰色线条代表输入波形，橙色数据代表在幅度调制模式下的输出波形，绿色数据代表在相位调制模式下的输出波形

e.可重构PCHT的极性转换。虚线和箭头表示输出电流I_ds的动态范围变化

f.光电输出随光照变化

## 静态三维视差重建不同空间平面的数学原理

### 平面表面及关键点间的线性插值

![iShot_2025-04-18_13.28.13](https://raw.githubusercontent.com/1910853272/image/master/img/202504181328093.png)

每个像素的坐标和强度被记录为空间向量 (x,y,I)，匹配对的平面位移通过视差原理视为每个关键点的z 坐标。



**强度和坐标的拟合平面：**$0=a⋅x+b⋅y+c⋅z+d$  其中，a、b、c、d 是空间参数。

**空间参数可以通过以下方程确定：**

$a=(y_2−y_1)⋅(z_3−z_1)−(y_3−y_1)⋅(z_2−z_1)$

$b=(x_3−x_1)⋅(z_2−z_1)−(x_2−x_1)⋅(z_3−z_1)$

$c=(x_2−x_1)⋅(y_3−y_1)−(x_3−x_1)⋅(y_2−y_1)$

$d=−(a⋅x+b⋅y+c⋅z)$

**其他点的坐标 (x, y) 可以通过以下线性位置方程进行插值：**

$y=k⋅x+b$ 

$\begin{vmatrix} \mathrm{k} \\ \mathrm{b} \end{vmatrix}=\mathrm{inv}\left( \begin{vmatrix} \mathrm{x_i} & 1 \\ \mathrm{x_j} & 1 \end{vmatrix}\right)\cdot \begin{vmatrix} \mathrm{y_i} \\ \mathrm{y_j} \end{vmatrix}$



对于**向量 $P_1P_2$ 和 $P_1P_4$上的两个坐标点 $L (x_l, y_l)$ 和 $R (x_r, y_r)$**可通过下列线性方程得到：

$y_{l} =\mathrm{inv}\left( \begin{vmatrix} \mathrm{x_1} & 1 \\ \mathrm{x_2} & 1 \end{vmatrix}\right)\cdot \begin{vmatrix} \mathrm{y_1} \\ \mathrm{y_2} \end{vmatrix}\cdot\begin{vmatrix} \mathrm{x_l} \\ \mathrm{1} \end{vmatrix} $

$y_{r} =\mathrm{inv}\left( \begin{vmatrix} \mathrm{x_1} & 1 \\ \mathrm{x_4} & 1 \end{vmatrix}\right)\cdot \begin{vmatrix} \mathrm{y_1} \\ \mathrm{y_4} \end{vmatrix}\cdot\begin{vmatrix} \mathrm{x_r} \\ \mathrm{1} \end{vmatrix} $

**向量$LR$上的每个点坐标$(x,y)$**可由下列方程得到：

$\mathrm{z=~-\frac{a~\cdot~x~+~b~\cdot inv\left( \begin{vmatrix} x_l & 1 \\ x_r & 1 \end{vmatrix}\right)\cdot~ \begin{vmatrix} y_l \\ y_r \end{vmatrix}\cdot \begin{vmatrix} x \\ 1 \end{vmatrix}~+d}{c}}$

### 曲面及关键点间的多项式插值

![Å](https://raw.githubusercontent.com/1910853272/image/master/img/202504181328678.png)

同理将平面方程推广到曲面$y = a · x^2 + b · x + c$，圆形和椭圆形$(x−x_0)^2+(y−y_0)^2+(z−z_0)^2=r^2$

## 动态三维重建时变空间尺寸的数学原理

<img src="https://raw.githubusercontent.com/1910853272/image/master/img/202504181423320.png" alt="iShot_2025-04-18_14.23.16" style="zoom:50%;" />

### 特定平移运动

相机的焦距为 f，物体与成像平面之间的距离为z，PCHT的位置设置为坐标系统的原点。

**物体与成像平面平行的一边$w_1$在成像平面上的投影尺寸$L_1$：**

$L_1=\frac{x_1+w_1}{z+f}\cdot f-\frac{x_1}{z+f}\cdot f=\frac{f}{z+f}\cdot w_1$

**物体与成像平面垂直的边$w_2$的横向尺寸$R_1$：**

$R_1=\frac{x_1}{z+f}\cdot f-\frac{x_1}{z+f+w_2}\cdot f=\frac{f}{(z+f)(z+f+w_2)}\cdot x_1\cdot w_2$

### 旋转平移运动

物体的每个边都有其与成像平面的夹角。

**物体的边长$w_i$在成像平面上的投影尺寸$L_i$：**

$\begin{aligned}L_{i}&=\frac{x_{i}+w_{i}\cos(\theta_{i})}{z+f}\cdot f-\frac{x_{i}}{z+f+\Delta z}\cdot f\\&=\left(\frac{f}{z+f}-\frac{f}{z+f+w_i\sin(\theta_i)}\right)\cdot x_1+\frac{w_i\cos(\theta_i)}{z+f}\cdot f\end{aligned}$

当$θ_i=0$或$θ_i=90$时，方程可简化为特定平移运动中的$L_1$和$R_1$公式。

### 垂直于相机平面的运动

![iShot_2025-04-18_14.53.01](https://raw.githubusercontent.com/1910853272/image/master/img/202504181455208.png)

L 是世界坐标中的位置

**深度场与投影位置关系：**$\frac{f}{f+z_i}=\frac{x_i}{L_i}$

**垂直运动为每个特征点带来了相同的深度增量$\Delta z$：**$\frac{f}{f+z_i+\Delta z}=\frac{x_i^{\prime}}{L_i}$

**深度场与其平面投影位置和像素坐标原点之间的距离呈负相关：**$\frac{x_i}{x_i^{\prime}}=1+\frac{\Delta z}{f+z_i}$

**通过对平面投影的两对特征点进行插值重建深度场：**$\frac{f+z_j}{f+z_i}=\frac{\frac{x_i}{x_i^{\prime}}-1}{\frac{x_j}{x_j^{\prime}}-1}$

# PCHT 不同模式光电性能

![iShot_2025-04-18_10.10.48](https://raw.githubusercontent.com/1910853272/image/master/img/202504181011055.png)

a.PCHT器件的光学显微镜图像

b.动态成像模式下时间依赖存储的操作机制模型。该模式允许器件根据时间变化存储图像数据。

c.恒感知模式下静态成像的操作机制模型。p型半导体的电流流动被调节为恒定状态

d.双极模式操作机制模型。同时调节两种载流子的流动，以实现更强的光电响应

e.可重构PCHT的等效功能电路。

f.时间依赖性存储模式下的光电响应

g.时间依赖性存储模式下对不同波长的不同脉冲数的光学响应

h.不同光激发强度的存储保持

i.恒感知模式下的光电流输出响应

j.恒感知模式下对不同波长的不同脉冲数的光学响应

k.光电流响应与静态与动态模式下的光电流比较

# 3D视差重建方法

![iShot_2025-04-18_10.38.25](https://raw.githubusercontent.com/1910853272/image/master/img/202504181042255.png)

a、b、c.单片集成的PCHT阵列在低分辨率和高分辨率下的单像素光学图像

d.PCHT阵列硬件架构示意

e.3D视差重建算法。算法灵感来源于尺度不变特征转换SIFT和Harris角点检测算法。

光流：空间运动物体在观察成像平面上的像素运动的瞬时速度

- PCHT阵列捕捉不同位置的运动特征。
- 计算方差和梯度来构建光流梯度OFG。
- 提取的OFG被编码成描述其特征的特征字符characterons。
- 通过向量拟合特征字符并匹配高欧几里得相关性的特征对。
- 匹配的特征对只有平面坐标，视差分析为这些匹配的特征对提供深度场坐标。完整的空间信息通过基于视差的空间插值重建。

f.传统CMOS成像与PCHT阵列成像的概念性差异。传统CMOS技术仅记录强度并通过多帧方法重建视差，PCHT阵列直接在传感器级别记录强度和时空视差。

g.在三个不同时间点（t0、t1、t2）对运动中的立方体进行动态成像

h.光流梯度OFG向量的分布

i.与特征关键点相关的字符超参数结构

j.特征匹配的关键点分布与相关性

k.通过2D光强重建物体的表面形状。R 是曲率半径，X是图像平面上的像素列，Y是像素行。

l.不同配置下的3D重建

## 相机标定

通过一系列的三维点和它对应的二维图像点进行数学变换，求出相机的内参数和外参数。内参与相机自身的焦距等有关，外参与相机的平移和旋转有关。

![iShot_2025-04-18_16.46.14](https://raw.githubusercontent.com/1910853272/image/master/img/202504181647909.png)

a.使用了带条纹的立方体作为标定设备，通过测量世界坐标系和像素坐标系中的10组对应点来标定投影矩阵。

b-e.PCHT时域配置在不同时刻记录一张图像中物体的运动轨迹。图像中明亮的对比度会与较晚的曝光相关，而空间配置的大小则决定了物体与相机的距离。

**世界坐标系$P_w$与像素坐标系p之间的关系可以通过投影矩阵来描述：**

$p=\begin{bmatrix}u_0\\v_0\end{bmatrix}=MP_w=K\begin{bmatrix}R&T\end{bmatrix}=K\begin{bmatrix}\overrightarrow{r}_1&\overrightarrow{t}_x\\\overrightarrow{r}_2&\overrightarrow{t}_y\\\overrightarrow{r}_3&\overrightarrow{t}_z\end{bmatrix}P_w$

K 是相机的内参矩阵，R是旋转矩阵，T是平移矩阵。

**投影矩阵 M可以通过奇异值分解（SVD）来计算**：

$M=K[RT]=\rho\begin{bmatrix}\overrightarrow{a_1}\\\overrightarrow{a_2},\vec{b}\\\overrightarrow{a_3}\end{bmatrix}=\begin{bmatrix}\alpha\cdot\overrightarrow{r_1}-\alpha\cdot\cot\theta\cdot\overrightarrow{r_2}+u_0\cdot\overrightarrow{r_3}\\\frac{\beta}{sin\theta}\overrightarrow{r_2}+v_0\overrightarrow{r_3}&\vec{b}\\\vec{r_3}\end{bmatrix}$

**旋转矩阵$r_1,r_2,r_3$相互独立，内参矩阵 K 可以通过以下方程来计算：**

$u_{0} = \frac{\overrightarrow{a_{1}} \cdot \overrightarrow{a_{3}}}{|\overrightarrow{a_{3}}|^{2}}$

$v_{0} = \frac{\overrightarrow{a_{2}} \cdot \overrightarrow{a_{3}}}{|\overrightarrow{a_{3}}|^{2}}$

$\alpha = \frac{|\overrightarrow{a_{1}} \times \overrightarrow{a_{3}}| \cdot \sin \theta}{|\overrightarrow{a_{3}}|^{2}}$

$\beta = \frac{|\overrightarrow{a_{2}} \times \overrightarrow{a_{3}}| \cdot \sin \theta}{|\overrightarrow{a_{3}}|^{2}}$

$\cos \theta = -\frac{(\overrightarrow{a_{1}} \times \overrightarrow{a_{3}}) \cdot (\overrightarrow{a_{2}} \times \overrightarrow{a_{3}})}{|\overrightarrow{a_{1}} \times \overrightarrow{a_{3}}| \cdot |\overrightarrow{a_{2}} \times \overrightarrow{a_{3}}|}$

$K=\begin{bmatrix}\alpha&-\alpha cot\theta&u_0\\0&\frac{\beta}{sin\theta}&v_0\\0&0&1\end{bmatrix}=\begin{bmatrix}1.9162&5.3802&25.2568\\0&9.9363&3.3921\\0&0&1\end{bmatrix}$

# 3D 视差重建演示

![iShot_2025-04-18_13.18.10](https://raw.githubusercontent.com/1910853272/image/master/img/202504181318786.png)

a.不同视图下的物体组件重建

b.2D深度场映射。暖色调表示物体离PCHT阵列较近

c.视差重建中使用的欧几里得距离和角度的分布统计

d.匹配关键点的相关空间

e.坐标变换方法以及坐标变换后的关键点分布。

f.光电流强度与多视角合成和视差重建获取物体的深度信息。

g.2D强度对比成像与3D视差成像的比较

g.正常眼球（上）和近视眼球（下）的表面重建。ρ表示重建眼球的曲率半径，展示了通过视差重建获取的眼球表面结构，比较了正常眼和近视眼在形态上的差异。



