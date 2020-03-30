function [bcoeff,v] = GeneralizedBernsteinCoeff_nD(coeff,deg,ID)
%GeneralizedBernsteinCoeff_nD - compute bernstein coefficients for a 
%multivariate polynomial funcion over the standard Simplex.
%
%bcoeff = GeneralizedBernsteinCoeff_nD(coeff,deg,ID) return a row vector 
%   with the Bernstein coefficient of a polynomial function defined over 
%   the standard simplex with the matrix of coefficients ordered as 
%   follows
%c_0,00...0 c_0,10...0 ... c_0,d(2)0...0 c_0,01...0 ...c_0,d(2)...d(n)
%c_1,00...0 c_1,10...0 ... c_1,d(2)0...0 c_1,01...0 ...c_1,d(2)...d(n)
% ...                                                   ...
%c_(1),00...0                    ...                c_d(1),d(2)...d(n)               
%
%   - DEG is one for each variable (n)
%
%   - ID is a matrix prod(deg+1)X(n-1) and it contains all the second indices 
%of the matrix coeff. You can leave it empty and the function will create
%it.
%
%
%[bcoeff,v] = BernsteinCoeff_1D(coeff,deg,domain,type) return not only the 
%   Bernstein coefficients but also the corresponding input variables, so
%   the control points. It holds that p(x) C= convhull(v,bcoeff)
%
maxdeg = sum(deg);
n = length(deg);

P = eye(maxdeg+1);
for i=1:maxdeg
    K = diag(ones(1,maxdeg+1)) + diag([zeros(1,i-1) ones(1,maxdeg+1-i)],-1);
    P = K*P;
end    

if isempty(ID)
    ID = zeros(prod(deg(2:end)+1),n-1);
    for j=2:prod(deg(2:end)+1)
        ID(j,:) = ID(j-1,:)+[1,zeros(1,n-2)];
        for k=1:n-2
            if ID(j,k)>deg(k+1)
                ID(j,k)=0;
                ID(j,k+1) = ID(j,k+1)+1;
            end
        end
    end
end

ID_new = zeros((maxdeg+1)^(n-1),n-1);
for j=2:(maxdeg+1)^(n-1)
    ID_new(j,:) = ID_new(j-1,:)+[1,zeros(1,n-2)];
    for k=1:n-2
        if ID_new(j,k)>maxdeg
            ID_new(j,k)=0;
            ID_new(j,k+1) = ID_new(j,k+1)+1;
        end
    end
end

C = zeros(maxdeg+1,(maxdeg+1)^(n-1));
for i=1:deg(1)+1
    for j=1:prod(deg(2:end)+1)
        current = j;
        while ~prod(ID(j,:)==ID_new(current,:))
            current = current + 1;
        end
        C(i,current) = coeff(i,j)/multinchoosek(maxdeg,[i-1 ID(j,:) maxdeg-i+1-sum(ID(j,:))]);
    end
end

A = P*C;
for k=2:n
    A_tilde = reshape(A,[(maxdeg+1)^(n-1),maxdeg+1])';
    A = P*A_tilde;
end
bcoeff_matrix = reshape(A,[(maxdeg+1)^(n-1),maxdeg+1])';

current = 1;
for i=1:maxdeg+1
    for j=1:(maxdeg+1)^(n-1)
        if i+sum(ID_new(j,:))-1<=maxdeg
            bcoeff(current) = bcoeff_matrix(i,j);
            current = current + 1;
        end
    end
end

v = cell(1,n);
current = 1;
for i=1:maxdeg+1
    for j=1:(maxdeg+1)^(n-1)
        if i+sum(ID_new(j,:))-1<=maxdeg
            v{1}(current) = (i-1)/maxdeg;
            for k=2:n
                v{k}(current) = (ID_new(j,k-1))/maxdeg;
            end
            current = current + 1;
        end
    end
end
end

