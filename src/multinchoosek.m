function k = multinchoosek(alpha,beta)
%  MULTINCHOOSEK - Binomial coefficient for multi-index.
%
% multinchoosek(alpha,beta) computes product(alpha(i)!/beta(i)!) where 
%   alpha and beta are two multi-index of the same length. 
%
% multinchoosek(alpha,beta) if alpha is a constant it computes 
%   alpha!/product(beta(i)!) where |beta| = sum(beta(i)) = alpha or it
%   computes  alpha!/product(beta(i)!)*(alpha-|beta|) when |beta|<alpha.
%
if length(alpha)==1
    k = factorial(alpha);
    for i=1:length(beta)
        k=k/factorial(beta(i));
    end
    if sum(beta)<alpha
        k=k/factorial(alpha-sum(beta));
    end
else
    assert(length(alpha)==length(beta))
    k=1;
    for i=1:length(alpha)
        k=k*nchoosek(alpha(i),beta(i));
    end
end

end

