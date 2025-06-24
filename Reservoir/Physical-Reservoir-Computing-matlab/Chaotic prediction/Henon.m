<<<<<<< HEAD
function [x,y] = Henon(datasize)
x(1,1)=0;
y(1,1)=0;
a=1.4;
b=0.3;
for i=1:datasize+1
    x(1,1+i)=1+y(1,i)-a*x(1,i)^2;
    y(1,1+i)=b*x(1,i);
end
x=x(1,3:end);
y=y(1,3:end);
=======
function [x,y] = Henon(datasize)
x(1,1)=0;
y(1,1)=0;
a=1.4;
b=0.3;
for i=1:datasize+1
    x(1,1+i)=1+y(1,i)-a*x(1,i)^2;
    y(1,1+i)=b*x(1,i);
end
x=x(1,3:end);
y=y(1,3:end);
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
end