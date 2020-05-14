function coeff = multiCoeff(A,b,C,type)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
deg = length(C)-1;
N = length(A);

if N>1
    coeff = zeros(deg+1,(deg+1)^(N-1));
    for d=0:deg
        for alpha=1:(d+1)^N
            ID = column2id(alpha,d*ones(1,N),type);
            if sum(ID)<=d
                idx = id2column(ID(2:end),deg*ones(1,N-1),type);
                coeff(ID(1)+1,idx) = coeff(ID(1)+1,idx) + C(d+1)*multinchoosek(d,ID)*prod(A.^ID)*b^(d-sum(ID));
            end
        end
    end
else
    coeff = 0.*C;
    for d=0:deg
        for alpha=d:deg
            coeff(d+1) = coeff(d+1) + C(alpha+1)*nchoosek(alpha,d)*A^d*b^(alpha-d);
        end
    end
end
end

