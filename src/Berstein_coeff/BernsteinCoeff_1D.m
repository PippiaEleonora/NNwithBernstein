function [bcoeff,v] = BernsteinCoeff_1D(coeff,deg,domain,type)
%BernsteinCoeff_1D - compute bernstein coefficients for single variable
%polynomial function.
%
%bcoeff = BernsteinCoeff_1D(coeff,deg,domain,type) return column vector 
%   with the bernstein coefficient of a polynomial function defined in 
%   an interval [domain(1),domain(2)] with the following expression:
%   p(x)=coeff(1)+coeff(2)*x+coeff(3)*x^2+...coeff(deg+1)*x^deg
%   - COEFF can be both row or column vector
%   - DEG is the length of COEFF
%   - DOMAIN row vector domain(1),domain(2)]
%   - TYPE is a string vector 
%
%
%[bcoeff,v] = BernsteinCoeff_1D(coeff,deg,domain,type) return not only the 
%   Bernstein coefficients but also the corresponding input variables, so
%   the control points. It holds that p(x) C = convhull(v,bcoeff)
%
%
%   You can choose between two implementations, type='Ray_Nataraj' and
%   type='Garloff'
%
if size(coeff,1)>1
    coeff = coeff';
end
switch type
    case 'Ray_Nataraj'
        Binomial = zeros(deg+1);
        for i=0:deg
            for j=0:i
                Binomial(i+1,j+1) = nchoosek(i,j);
            end
        end
        U_1 = zeros(deg+1);
        V_1 = zeros(deg+1);
        W_1 = zeros(deg+1);
        U_1(deg+1,:) = ones(1,deg+1);
        U_1(:,1) = ones(deg+1,1);
        V_1(1,1) = 1;
        W_1(1,1) = 1;
        
        LB = ones(1,deg+1);
        for i=2:deg+1
            W_1(i,i) = 1;
            LB(i) = LB(i-1).*domain(1);
            V_1(i,i) = V_1(i-1,i-1)*(domain(2)-domain(1));
            for j=2:i
                U_1(i,j) = Binomial(i,i-j+1)./Binomial(deg+1,j);
            end
        end

        for i=1:deg+1
            for j=i+1:deg+1
                W_1(i,j) = Binomial(j,i).*LB(j-i+1);
            end
        end

        if size(coeff,1)==1
            coeff = coeff';
        end
        bcoeff = U_1*V_1*W_1*coeff;
        
    case 'Garloff'
        K = cell(1,deg);
        P = eye(deg+1);
        for i=1:deg
            K{i} = diag(ones(1,deg+1)) + diag([zeros(1,i-1) ones(1,deg+1-i)],-1);
            P = K{i}*P;
        end   
        
        if domain(1)==0 
            if domain(2)==1
                coeff_new = coeff;
            else
                vector = zeros(1,deg+1);
                for i=0:deg
                    vector(i+1) = domain(2)^i;
                end
                coeff_new = (diag(vector)*coeff')';
            end
        else
            c = (domain(2)-domain(1))/domain(1);
            vector1 = zeros(1,deg+1);
            vector2 = zeros(1,deg+1);
            for i=0:deg
                vector1(i+1) = c^i;
                vector2(i+1) = domain(1)^i;
            end
            coeff_new = ((diag(vector1)*P'*diag(vector2))*coeff')';
        end
        
        Lambda = zeros(deg+1,1);
        for i=1:deg+1
            Lambda(i) = coeff_new(i)/P(deg+1,i);
        end
        bcoeff = P*Lambda;
end
v = linspace(0,1,deg+1).*(domain(2)-domain(1))+domain(1);
end

