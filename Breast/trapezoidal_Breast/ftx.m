function y=ftx(a,b1,b2,x)
y=((x-a+b2)/(b2-b1)).*(a-b2<=x & x<a-b1)+((a+b2-x)/(b2-b1)).*(a+b1<=x & x<a+b2)+0.*(x<a-b2 | x>=a+b2)+1.*(a-b1<=x & x<a+b1);
end