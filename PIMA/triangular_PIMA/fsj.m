function y=fsj(a,b,x)
y=((x-a+b)/b).*(a-b<=x & x<a)+((a+b-x)/b).*(a<=x & x<a+b)+0.*(x<a-b | x>=a+b);
end