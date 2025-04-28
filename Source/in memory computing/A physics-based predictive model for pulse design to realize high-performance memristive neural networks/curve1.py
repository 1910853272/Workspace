import numpy as np
import math


#pulse(same E,same t0)
#thick=2.8#厚度，在之后曲线拟合中会用到
E=2.2
Gon=1/600
Goff=1/180000
x_start=50E-9#脉冲宽度
Ptot=256
Y=[]
S=[]
N=[]
#计算G的值
def GLTP(x):
    S=0.5-(1/3.14159)*np.arctan((math.log10(x/tmean))/w)#LTP曲线#这是S向下的区域面积
    G = (1 - S) * Gon + S * Goff  #Goff
    return G,S
def GLTD(x):
    S=0.5+(1/3.14159)*np.arctan((math.log10(x/tmean))/w)#x是自变量，LTD曲线
    G = (1 - S) * Gon + S * Goff  #Gon
    return G,S

#根据当前G值判断t，在LTP换LTD里面会用到
def T(G,w,tmean):
    s=(G-Gon)/(Goff-Gon)
    x=10**(w*np.tan(3.14159*(s-0.5))+math.log10(tmean)) # LTD曲线的反转
    return x

#calculate tmean and w
tmean=10**(7.01069*(2.8/E)-14.0302566)
w=0.4577*((2.8**2)/(E**2))-0.23397

#脉冲主函数
#相同电场不同宽度，要增加一个初始值（第0个脉冲的时候电导值为0）
#电导
# IDX=0  # IDX用于检验循环中间是否有问题
#LTP部分
Y.append(0)
for i in range(Ptot):
    #这个i要单独写一写,写脉冲宽度上去
    if i>=1:
        x=x_start*i
        y,s = GLTP(x)
        Y.append(y)
        S.append(s)
        N.append(i)  # 计算脉冲数
        i = i + 1
        print(y)
#LTD部分
ltp_end=y #从LTP最后一个脉冲的y
t=T(ltp_end,w,tmean)
for i in range(Ptot):
    x = x_start * i + t
    y,s = GLTD(x)
    Y.append(y)
    S.append(s)
    N.append(i)  # 计算脉冲数
    i = i + 1
    print(y)