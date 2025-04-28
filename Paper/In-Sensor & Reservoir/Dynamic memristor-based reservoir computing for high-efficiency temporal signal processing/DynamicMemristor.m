<<<<<<< HEAD
% 函数 DynamicMemristor 用于更新忆阻器的电导𝐺并计算输出电流𝐼
function [I,G]=DynamicMemristor(V,G,para)
% 衰减+恢复项+动态更新项
G=para.r*G+(1-para.r)*para.G0+updata(V,para.alpha).*(binaryFunc(V)-G);
% 电流输出公式
I=G.*(para.Kp*NL(max(V,0))+para.Kn*NL(min(V,0)));
end

function y=binaryFunc(x)
id=x>0;
y(id,1)=1;
y(~id,1)=0;
end

function y=updata(x,a)
y=a*abs(x)./(a*abs(x)+1);
end

function y=NL(x)
y=x.^3;
end
=======
% 函数 DynamicMemristor 用于更新忆阻器的电导𝐺并计算输出电流𝐼
function [I,G]=DynamicMemristor(V,G,para)
% 衰减+恢复项+动态更新项
G=para.r*G+(1-para.r)*para.G0+updata(V,para.alpha).*(binaryFunc(V)-G);
% 电流输出公式
I=G.*(para.Kp*NL(max(V,0))+para.Kn*NL(min(V,0)));
end

function y=binaryFunc(x)
id=x>0;
y(id,1)=1;
y(~id,1)=0;
end

function y=updata(x,a)
y=a*abs(x)./(a*abs(x)+1);
end

function y=NL(x)
y=x.^3;
end
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
