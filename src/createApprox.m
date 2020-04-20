function [poly, error] = createApprox(answer,deg,ifplot,Iconfid,ifconfidence)
%CREATEAPPROX create a rational or polynomial approximation for the
%hyperbolic tangent.
%
% 'Polynomial' = taylor approximation
% 'Chebyshev' = Chebyshev approximation
% 'Rational' = Lambert's serie
%
% deg is the degree for the expression. If you select 'Chebyshev' you need 
% to insert a vector of two elements
%
% plot is a boolean variable to plot the approximation in comparison with 
% tansig(x).
%
z = sym('z');
switch answer
    case 'Polynomial'
        f(z) = tanh(0)*z^0;
        d(z) = tanh(z);
        for k=1:deg
            d(z) = diff(d(z));
            f(z) = f(z)+d(0)*z^k;
        end
        poly = simplify(f);
    case 'Chebyshev'
        p = deg(1);
        q = deg(2);
        f(z) =  nchoosek(q,0)*(1-z)^0;
        for i=1:p
            f(z) = f(z) + nchoosek(q+i,i)*((1-z)/2)^i;
        end
        f(z) = ((1+z)/2)^(q+1)*f(z);
        f(z) = f(z/3.7)*2-1;
        poly = simplify(f); 
    case 'Rational'
        const = 3:2:deg*2;
        f(z) = z^2/const(end);
        for k=length(const)-1:-1:1
            f(z) = z^2/(const(k)+f(z));
        end
        f(z) = z/(1+f(z));

        poly = simplify(f);
end

if ifplot
    xx = linspace(Iconfid(1),Iconfid(2),1000);
    figure
    plot(xx,eval(poly(xx)),'LineWidth',1)
    hold on
    plot(xx,tansig(xx),'blue','LineWidth',2)
    legend('approx','tanh')
end

if ifconfidence
    f1 = matlabFunction(poly);
    g =@(z) -abs(f1(z)-tansig(z))
    x1 = fminbnd(g,Iconfid(1),Iconfid(2));
    x2 = fminbnd(@(z) tansig(z)-1,Iconfid(2),100);
    x3 = fminbnd(@(z) -1-tansig(z),-100,Iconfid(1));
    error1 = double(g(x1));
    error2 = tansig(x2)-1;
    error3 = -1-tansig(x3);
    
    error = [error1 error2 error3];
else
    error = -1;
end
end

