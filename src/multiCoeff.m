function coeff = multiCoeff(A,b,C)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
deg = length(C)-1;
N = length(A);

if N>1
    coeff = zeros(deg+1,(deg+1)^(N-1));
    for d=0:deg
        for alpha=1:(d+1)^N
            ID = column2id(alpha,d*ones(1,N),'first');
            if sum(ID)<=d
                idx = id2column(ID(2:end),deg*ones(1,N-1),'first');
                coeff(ID(1)+1,idx) = coeff(ID(1)+1,idx) + C(d+1)*multinchoosek(d,ID)*prod(A.^ID)*b^(d-sum(ID));
            end
        end
    end
end
end

