function y=ftxb1(a,b1,b2,x)
y=((x-a+b2)/((b2-b1)^2)).*(a-b2<=x & x<a-b1)+((a+b2-x)/((b2-b1)^2)).*(a+b1<=x & x<a+b2)+0.*(x<a-b2 | x>=a+b2| a-b1<=x & x<a+b1);
end