import numpy as np
import math
"""修改LTP和LTD需要修改三个地方：函数：G_start,conductance,T都要改，然后图片标题的LTPLTD要改"""

#pulse(different E,same t0)
pulse=256
thick=2.8#厚度，在之后曲线拟合中会用到
Gon=1/600
Goff=1/180000
E_start=1.4
E_end=2.4

width=(E_end-E_start)/pulse
n=0
t_change=50E-9 #脉冲宽度相同，每个脉冲之间的宽度
N=[]
Y=[]
S=[]
end=[]
start=[]

# 计算初始电导的值，y是要把E_start这个曲线的脉宽的第一个G算出来
def GLTP_start(E,x):
    tmean = 10 ** (7.01069 * (2.8 / E) - 14.0302566)
    w = 0.4577 * ((2.8 ** 2) / (E ** 2)) - 0.23397
    S=0.5-(1/3.14159)*np.arctan((math.log10(x/tmean))/w)#LTP曲线,这是S向下的区域面积
    G = (1 - S) * Gon + S * Goff  #Goff
    return G

def GLTD_start(E,x):
    tmean = 10 ** (7.01069 * (2.8 / E) - 14.0302566)
    w = 0.4577 * ((2.8 ** 2) / (E ** 2)) - 0.23397
    S=0.5+(1/3.14159)*np.arctan((math.log10(x/tmean))/w)#x是自变量，LTD曲线
    G = (1 - S) * Gon + S * Goff  #Gon
    return G

#计算G的值
def GLTP(x,tmean,w):
    S=0.5-(1/3.14159)*np.arctan((math.log10(x/tmean))/w)#LTP曲线
    G = (1 - S) * Gon + S * Goff
    return S,G

def GLTD(x,tmean,w):
    S=0.5+(1/3.14159)*np.arctan((math.log10(x/tmean))/w)#x是自变量，LTD曲线
    G = (1 - S) * Gon + S * Goff
    return S,G

#由G值判断当前t
def LTP_T(G,w,tmean):
    s=(G-Gon)/(Goff-Gon)
    x=10**(w*np.tan(3.14159*(0.5-s))+math.log10(tmean)) # LTP
    return x

def LTD_T(G,w,tmean):
    s=(G-Gon)/(Goff-Gon)
    x=10**(w*np.tan(3.14159*(s-0.5))+math.log10(tmean)) # LTD曲线的反转
    return x

#设计脉冲电场强度，calculate tmean and w
def parameter(E):
    tmean=10**(7.01069*(2.8/E)-14.0302566)
    w=0.4577*(2.8**2/(E**2))-0.23397
    return tmean,w

#脉冲
#每两个脉冲上升一丢丢，宽度不变
#脉冲第一段
IDX=0
#LTP曲线
E=np.arange(E_start,E_end,width)#E逐渐增加,但也可以改成E逐渐减小
y=GLTP_start(E_start,t_change)
for i in E:
    t, w = parameter(i)  # 计算参数
    #计算上一个的结束时间
    T_end=LTP_T(y,w,t)
    T_start=T_end+t_change # t_change是每次间隔的距离
    #计算每个脉冲的y
    n=n+1
    s,y=GLTP(T_start,t,w)#这里的T_start是起始初始的t
    N.append(n)  # 计算脉冲数
    Y.append(y)  # 计算电导值序列
    S.append(s)
    end.append(T_end)
    start.append(T_start)
    print(y)
flag=0
#LTD曲线
E=np.arange(E_start,E_end,width)#E逐渐增加,但也可以改成E逐渐减小
for i in E:
    t, w = parameter(i)  # 计算参数
    #计算上一个的结束时间
    T_end=LTD_T(y,w,t)
    T_start=T_end+t_change # t_change是每次间隔的距离
    #计算每个脉冲的y
    n=n+1
    s,y=GLTD(T_start,t,w)#这里的T_start是起始初始的t
    N.append(n)  # 计算脉冲数
    Y.append(y)  # 计算电导值序列
    S.append(s)
    end.append(T_end)
    start.append(T_start)
    print(y)
    #print(s)