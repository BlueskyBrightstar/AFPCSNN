function y=fzxa(a,b1,b2,x)
y=(2*b2*(((x-a)/b1).^(2*b2-1)))./(b1*((1+((x-a)/b1).^(2*b2))^2));
end