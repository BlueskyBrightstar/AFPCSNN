function y=fsjb(a,b,x)
y=((a-x)/(b^2)).*(a-b<=x & x<a)+((x-a)/(b^2)).*(a<=x & x<a+b)+0.*(x<a-b | x>=a+b);
end