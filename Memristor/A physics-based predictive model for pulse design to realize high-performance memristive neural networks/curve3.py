import numpy as np
import math

#pulse(初始宽度为50E-9,每个脉冲增加1E-2宽)
pulse=256
thick=2.8#厚度，在之后曲线拟合中会用到
E=2.2
Gon=1/600
Goff=1/180000
wid_start=5E-10
wid_end=1E-7
wid_change=(wid_end-wid_start)/pulse
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

def T(G,w,tmean):
    s=(G-Gon)/(Goff-Gon)
    x=10**(w*np.tan(3.14159*(s-0.5))+math.log10(tmean)) # LTD曲线的反转
    #x = 10 ** (w * np.tan(3.14159 * (0.5 - s)) + math.log10(tmean))  # LTP
    return x

#calculate tmean and w
tmean=10**(7.01069*(2.8/E)-14.0302566)
w=0.4577*((2.8**2)/(E**2))-0.23397

width=np.arange(wid_start,wid_end,wid_change)#E逐渐增加,但也可以改成E逐渐减小

#脉冲主函数
#相同电场不同宽度，要增加一个初始值（第0个脉冲的时候电导值为0）
#应该修改这个占空比，
#电导LTP
wid=0
WIDTH=np.arange(wid_start,wid_end,wid_change)
for i in WIDTH:
    #这个i要单独写一写,写脉冲宽度上去
    wid=i+wid
    y,s = GLTP(wid)
    Y.append(y)
    S.append(s) #增加固定脉冲宽度
    # print(y)

#LTD
ltp_end=y #从LTP最后一个脉冲的y
t=T(ltp_end,w,tmean)
wid=t #初始脉冲宽度
for i in WIDTH:
    y, s = GLTD(wid)
    Y.append(y)
    S.append(s)
    N.append(i)  #
    wid=wid+i
    # print(y)
