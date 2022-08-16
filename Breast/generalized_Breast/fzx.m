function y=fzx(a,b1,b2,x)
y=1./(1+((x-a)/b1).^(2*b2));
%x=-1:0.01:10;plot(x,fzx(4,4,6,x)),hold on
end