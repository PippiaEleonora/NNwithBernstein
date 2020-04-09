function poly = createApprox(answer,deg,ifplot)
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
    xx = linspace(-5, 5,100);
    figure
    plot(xx,eval(poly(xx)),'LineWidth',1)
    hold on
    plot(xx,tansig(xx),'blue','LineWidth',2)
    legend('approx','tanh')
end
end

