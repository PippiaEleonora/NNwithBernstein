function [bcoeff,v] = BernsteinCoeff_nD(coeff,deg,domain,type)
%BernsteinCoeff_nD - compute bernstein coefficients for multi variable
%polynomial function.
%
%bcoeff = BernsteinCoeff_nD(coeff,deg,domain,type) return column vector 
%   with the bernstein coefficient of a polynomial function defined in 
%   an box [domain(i,1),domain(i,2)] with the following expression:
%c_0,0...00 c_0,0...01 ... c_0,0...0d(n) c_0,0...10 ...c_0,d(2)...d(n)
%c_1,0...00 c_1,0...01 ... c_1,0...0d(n) c_1,0...10 ...c_1,d(2)...d(n)
% ...                                                   ...
%c_(1),0...00                    ...                c_d(1),d(2)...d(n) 

%   - DEG is one for each variable (n)
%   - DOMAIN is a matrix nX2
%   - TYPE is a string vector 
%
%
%[bcoeff,v] = BernsteinCoeff_nD(coeff,deg,domain,type) return not only the 
%   Bernstein coefficients but also the corresponding input variables, so
%   the control points. It holds that p(x) C = convhull(v,bcoeff)
%
%
%   You can choose between two implementations, type='Ray_Nataraj' and
%   type='Garloff'
%
maxdeg = max(deg);
n = length(deg);

switch type
    case 'Ray_Nataraj'
        Binomial = zeros(maxdeg+1);
        U_1 = cell(1,n);
        V_1 = cell(1,n);
        W_1 = cell(1,n);
        LB = cell(1,n);

        for i=1:n
            U_1{i} = zeros(deg(i)+1);
            V_1{i} = zeros(deg(i)+1);
            W_1{i} = zeros(deg(i)+1);

            U_1{i}(deg(i)+1,:) = ones(1,deg(i)+1);
            U_1{i}(:,1) = ones(deg(i)+1,1);
            V_1{i}(1,1) = 1;

            LB{i} = ones(1,deg(i)+1);
        end

        for i=0:maxdeg
            for k=1:n
                if i<=deg(k)
                    W_1{k}(i+1,i+1) = 1;
                end
            end
            for j=0:i
                Binomial(i+1,j+1) = nchoosek(i,j);
            end
        end

        for i=2:maxdeg+1
            for k=1:n
                if i<=deg(k)+1
                    LB{k}(i) = LB{k}(i-1).*domain(k,1);
                    V_1{k}(i,i) = V_1{k}(i-1,i-1)*(domain(k,2)-domain(k,1));
                    for j=2:i
                        U_1{k}(i,j) = Binomial(i,i-j+1)./Binomial(deg(k)+1,j);
                    end
                end
            end
        end
        for i=1:maxdeg+1
            for k=1:n
                if i<=deg(k)+1
                    for j=i+1:deg(k)+1
                        W_1{k}(i,j) = Binomial(j,i).*LB{k}(j-i+1);
                    end
                end
            end
        end

        A = U_1{1}*V_1{1}*W_1{1}*coeff;
        for k=2:n
            A_tilde = reshape(A,[prod(deg+1)/(deg(k)+1),deg(k)+1])';
            A = U_1{k}*V_1{k}*W_1{k}*A_tilde;
        end
        bcoeff = reshape(A,[prod(deg+1)/(deg(1)+1),deg(1)+1])';

    case 'Garloff'
        K = cell(n,maxdeg);
        P = cell(1,n);
        for k=1:n
            P{k} = eye(deg(k)+1);
        end
        for i=1:maxdeg
            for k=1:n
                if i<=deg(k)
                    K{k,i} = diag(ones(1,deg(k)+1)) + diag([zeros(1,i-1) ones(1,deg(k)+1-i)],-1);
                    P{k} = K{k,i}*P{k};
                end
            end
        end    
        
        update = 0;
        for k=1:n
            if domain(k,1)~=0 || domain(k,2)~=1
                update = 1;
            end
        end
        
        if update
            Q = cell(1,n);
            for k=1:n
                if domain(k,1)==0 
                    vector = zeros(1,deg(k)+1);
                    for i=0:deg(k)
                        vector(i+1) = domain(k,2)^i;
                    end
                    Q{k} = diag(vector);
                else
                    c = (domain(k,2)-domain(k,1))/domain(k,1);
                    vector1 = zeros(1,deg(k)+1);
                    vector2 = zeros(1,deg(k)+1);
                    for i=0:deg(k)
                        vector1(i+1) = c^i;
                        vector2(i+1) = domain(k,1)^i;
                    end
                    Q{k} = diag(vector1)*P{k}'*diag(vector2);
                end
            end
            Q_star = Q{1}*coeff;
            for k=2:n
                Q_tilde = reshape(Q_star,[prod(deg+1)/(deg(k)+1),deg(k)+1])';
                Q_star = Q{k}*Q_tilde;
            end
            coeff_new = reshape(Q_star,[prod(deg+1)/(deg(1)+1),deg(1)+1])';
        else
            coeff_new = coeff;
        end
            
        
        Lambda = zeros(deg(1)+1,prod(deg(2:end)+1));
        ID = zeros(prod(deg(2:end)+1),n-1);
        for j=2:prod(deg(2:end)+1)
            ID(j,:) = ID(j-1,:)+[zeros(1,n-2),1];
            for k=n-1:-1:2
                if ID(j,k)>deg(k+1)
                    ID(j,k)=0;
                    ID(j,k-1) = ID(j,k-1)+1;
                end
            end
        end
        for i=1:deg(1)+1
            for j=1:prod(deg(2:end)+1)
                denominator = P{1}(deg(1)+1,i);
                for k=1:n-1
                    denominator = denominator*P{k+1}(deg(k+1)+1,ID(j,k)+1);
                end
                Lambda(i,j) = coeff_new(i,j)/denominator;
            end
        end
        
        A = P{1}*Lambda;
        for k=2:n
            A_tilde = reshape(A,[prod(deg+1)/(deg(k)+1),deg(k)+1])';
            A = P{k}*A_tilde;
        end
        bcoeff_matrix = reshape(A,[prod(deg+1)/(deg(1)+1),deg(1)+1])';
        bcoeff = reshape(bcoeff_matrix,1,[]);
end
v = cell(1,n);
for k=1:n
    v{k} = zeros(1,deg(k)+1);
    for j=1:deg(k)+1
        v{k}(j) = (j-1)/deg(k).*(domain(k,2)-domain(k,1))+domain(k,1);
    end
end
end

