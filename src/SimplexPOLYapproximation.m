function [polytope,Ptotal] = SimplexPOLYapproximation(polynomial,poly,W,bias,n_layer,n_neurons,x,Domain)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
polytope_curr = Domain;
z = sym2cell(x);
Ptotal = cell(1,n_layer+1);
coeffP = fliplr(double(coeffs(poly(x(1)),x(1),'All')));
for l=1:n_layer
    [A,b,Aeq,beq] = vert2lcon(polytope_curr);
    singlePoint = 0;
    degenerate = 0;
    
    if isempty(Aeq)
        [M,N] = size(polytope_curr);
        assert(M>=N+1)
        polytope_curr = lcon2vert(A,b);
        Ptotal{l} = polytope_curr;
        [PT,n_triangle,polytope_new] = My_Triangulation(polytope_curr);
    elseif isempty(A) %single point
        singlePoint = 1;
    else %is degenerate
        [P_new,Projection] = degeneration('vert',polytope_curr);
        degenerate = 1;
        Ptotal{l} = polytope_curr;
        [m1,N] = size(Aeq);
        assert(m1<N)
        Aeq_1 = Aeq(:,1:m1);
        Aeq_2 = Aeq(:,m1+1:end);
        
        A_1 = A(:,1:m1);
        A_2 = A(:,m1+1:end);
        
        assert(det(Aeq_1)~=0) % DEAL WITH THIS CASE!
        
        invAeq_1 = Aeq_1^-1;
        assert(det(Aeq_1)~=0)
        A_new = A_2-(A_1*invAeq_1*Aeq_2);
        b_new =b-(A_1*invAeq_1*beq);
        
        V=lcon2vert(A_new,b_new);
        [PT,n_triangle,polytope_new] = My_Triangulation(V);
        
        proj_sym(x) = invAeq_1*(beq - Aeq_2*transpose(x(1,1:N-m1)));
    end
    %% 
    if singlePoint
        Point = lcon2vert(A,b,Aeq,beq);
        Pcell = num2cell(Point);
        polytope_curr = zeros(1,n_neurons(l));
        for t=1:n_neurons(l)
            polytope_curr(t) = double(poly(W{l,1}(t,:)*Point + bias{l,1}(t)));%polynomial{l,t}(Pcell{:});
        end
        Ptotal{l} = polytope_curr;
    else
%         v_tri = [];
        bcoeff_tri = [];
        for i=1:n_triangle
            vertex = polytope_new(PT(i,:),:)';
            V = vertex(:,2:end)-vertex(:,1);
            b = vertex(:,1);
            p = cell(1,length(b));
            for idx=1:length(b)
                p{1,idx} = x(1:size(vertex,1))*(V(idx,:)') + b(idx);
            end
            if degenerate
                
                proj_cell = cell(1,m1);
                matrix = invAeq_1*Aeq_2;
                for idx=1:m1
                    proj_cell{1,idx} = invAeq_1(idx,:)*beq - (x(1:size(vertex,1))*(V')+b')*matrix(idx,:)';
                end
            else
                proj_cell = {};
            end
            % 'deg_all' is the degree of each polynomial and 'deg' is dhe
            % maximum degree for each variable in each polynomial
            deg_all = cell(n_neurons(l),1);
            C = cell(n_neurons(l),1);
            for t=1:n_neurons(l)
                if degenerate
                    Matrix = W{l,1}(t,:)*[Projection.A; eye(m)]*V;
                    vector = W{l,1}(t,:)*[Projection.A; eye(m)]*b + W{l,1}(t,:)*[Projection.b; zeros(m,1)]+bias{l,1}(t);
                else
                    Matrix = W{l,1}(t,:)*V;
                    vector = W{l,1}(t,:)*b+bias{l,1}(t);
                end
                
                if l==n_layer
                    C{t} = coeffs(polynomial{l,t}(proj_cell{:},p{:}),'All');
                else
                    C{t} = multiCoeff(Matrix, vector,coeffP);
                end
                if size(C{t},1)>1
                    deg_all{t} = size(C{t})-1;
                else
                    deg_all{t} = length(C{t})-1;
                end
            end
            deg = deg_all{1};
            for t=2:n_neurons(l)
                for d=1:size(polytope_new,2)
                    if deg_all{t}(d)>deg(d)
                        deg(d) = deg_all{t}(d);
                    end
                end
            end
            bcoeff_neu = [];
            for t=1:n_neurons(l)
                if isempty(C{t})
                    bcoeff = 0;
                else
                    if N<3
                        coeff_reverse = double(C{t});
                    else
                        coeff_reverse = double(C{t});
                    end

                    coeff = zeros(deg(1)+1,prod(deg(2:end)+1));
                    n = length(deg);
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
                    for k=0:deg(1)
                        for j=1:prod(deg(2:end)+1)
                            index = num2cell(deg_all{t}(2:end)+1-ID(j,:));
                            if isempty(index)
                                if k<=deg_all{t}(1)
                                    coeff(k+1) = coeff_reverse(deg_all{t}(1)+1-k);
                                end
                            else
                                if (prod(ID(j,:)<=deg_all{t}(2:end))) && k<=deg_all{t}(1)
                                    coeff(k+1,j) = coeff_reverse(deg_all{t}(1)+1-k,index{:});
                                end
                                
                            end
                        end
                    end
                    if size(coeff,2)<2
                        bcoeff = BernsteinCoeff_1D(coeff,deg,[0 1],'Garloff');
                    else
                        bcoeff = GeneralizedBernsteinCoeff_nD(coeff,deg,ID);
                        bcoeff = bcoeff';
                    end
                end
                bcoeff = max(min(bcoeff,1),-1);
                bcoeff_neu = [bcoeff_neu bcoeff];

            end
            % devo mantenere la struttura di tutte le variabili
            % devo gestire vincoli vuoti
            % in generale devo ricondurmi a casi precedenti (1D->intervalli) 
    %         [A,b] = vert2lcon([v_new, bcoeff_neu]);
    %         A_lay = [A_lay; A];
    %         b_lay = [b_lay; b];
    %         v_tri = [v_tri; v_new];
            bcoeff_tri = [bcoeff_tri; bcoeff_neu];
        end
        polytope_curr = unique(bcoeff_tri, 'row');
        
    end
   
end
polytope = polytope_curr;
Ptotal{l+1} = polytope_curr;
end

