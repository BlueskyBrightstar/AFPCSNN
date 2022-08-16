function y=fsja(a,b,x)
y=(-1/b).*(a-b<=x & x<a)+(1/b).*(a<=x & x<a+b)+0.*(x<a-b | x>=a+b);
end