<<<<<<< HEAD
function output=device_linear(input)
[m,n]=size(input);
out=zeros(m,1);
for i=1:m
    for j=1:n
        out(i,j+1)=out(i,j)+input(i,j);
    end
end
output=out(:,2:end);
=======
function output=device_linear(input)
[m,n]=size(input);
out=zeros(m,1);
for i=1:m
    for j=1:n
        out(i,j+1)=out(i,j)+input(i,j);
    end
end
output=out(:,2:end);
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
end