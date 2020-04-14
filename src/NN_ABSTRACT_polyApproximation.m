function [polytope] = NN_ABSTRACT_polyApproximation(poly,precision,Iconfident,W,bias,n_layer,n_neurons,Domain)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
poly_curr = Domain;
%
% poly_curr is a structure define with 2 elements: poly_curr.A, poly_curr.b
% 
%   each point x in poly_curr satisfy
%       poly_curr.A*x <= poly_curr.b
%       poly_curr.Aeq*x = poly_curr.beq
%   
%
for l=1:n_layer
    N = size(poly_curr.A,2);
    poly_new.A = [];
    poly_new.b = [];
    poly_new.Aeq = [];
    poly_new.beq = [];
    for t=1:n_neurons(l)
        if N==1
            assert(size(poly_curr.A,1)==N+1)
            assert(sign(poly_curr.A(1)*poly_curr.A(2))<0)
            
            interval = poly_curr.b./poly_curr.A;
            interval_new = W{l,1}(t,:)*interval + bias{l,1}(t);    
            if interval_new(1)>interval_new(2)
                interval_new = [interval_new(2) interval_new(1)];
            end
            
            Pminus = [interval_new(1) min(Iconfident(1),interval_new(2))];
            Pplus = [max(Iconfident(2),interval_new(1)) interval_new(2)];
            P = [max(Iconfident(1),interval_new(1)) min(Iconfident(2),interval_new(2))];
            
            poly_temp = [];
            if Pminus(2)>=Pminus(1)
                I_split = (Pminus - bias{l,1}(t))/W{l,1}(t,:);
                poly_temp = [poly_temp; I_split' [-1; -1]];
            end
            if Pplus(2)>=Pplus(1)
                I_split = (Pplus - bias{l,1}(t))/W{l,1}(t,:);
                poly_temp = [poly_temp; I_split' [1; 1]];
            end
            if P(2)>=P(1)
                coeffP = coefficients(poly, precision(l,t), [], []); % TBI
                [bcoeff, v] = BernsteinCoeff_1D(coeffP,length(coeffP)-1,P,'Garloff');
                I_split = (v - bias{l,1}(t))/W{l,1}(t,:);
                poly_temp = [poly_temp; I_split' bcoeff];
            end
        else
            poly_temp = [];
            
            halfspacePlus.A =  -W{l,1}(t,:);
            halfspacePlus.b = bias{l,1}(t)-Iconfident(2);
            Pplus = lcon2vert([poly_curr.A; halfspacePlus.A], [poly_curr.b; halfspacePlus.b]); 
            if ~isempty(Pplus)
                poly_temp = [poly_temp; Pplus ones(size(Pplus,1),1)];
            end
            
            halfspaceMinus.A =  W{l,1}(t,:);
            halfspaceMinus.b = -bias{l,1}(t)+Iconfident(1);
            Pminus = lcon2vert([poly_curr.A; halfspaceMinus.A],[poly_curr.b; halfspaceMinus.b]);
            if ~isempty(Pminus)
                poly_temp = [poly_temp; Pminus -ones(size(Pminus,1),1)];
            end
            
            verticalspace.A = [W{l,1}(t,:); -W{l,1}(t,:)];
            verticalspace.b = [-bias{l,1}(t)+Iconfident(2); bias{l,1}(t)-Iconfident(1)];
            P = lcon2vert([poly_curr.A; verticalspace.A], [poly_curr.b; verticalspace.b]);
            
            if ~isempty(P)
                [P_new,Projection] = degeneration('vert',P); % 1 CASE TBI
                if size(P_new,1)==1 % single point
                    point = [Projection.A*P_new+Projection.b , P_new];
                    poly_temp = [point, evaluation(poly, precision(l,t), W{l,1}(t,:)*point'+ bias{l,1}(t))]; %TBI
                else
                    [indexTR,n_triangle,polytope_new] = My_Triangulation(P_new); 
                    for k=1:n_triangle
                        vertex = polytope_new(indexTR(k,:),:)';
                        V = vertex(:,2:end)-vertex(:,1);
                        a = vertex(:,1);
                        if size(P_new,2)<N % degenerate case
                            m = size(P_new,2);
                            assert(size(P_new,2)==size(Projection.A,2))
                            
                            Matrix = W{l,1}(t,:)*[Projection.A; eye(m)]*V;
                            vector = W{l,1}(t,:)*[Projection.A; eye(m)]*a + W{l,1}(t,:)*[Projection.b; zeros(m,1)]+bias{l,1}(t);

                            coeff = coefficients(poly, precision(l,t), Matrix, vector); % TBI
                            deg = (size(coeff,1)-1)*ones(1,m);
                            if m==1
                                [bcoeff,v] = BernsteinCoeff_1D(coeff,deg,[0 1],'Garloff');
                            else
                                [bcoeff,v_cell] = GeneralizedBernsteinCoeff_nD(coeff,deg,[]);
                                bcoeff = bcoeff';
                                v = [];
                                for idx=1:m
                                    v =[v v_cell{idx}'];  
                                end
                            end
                            v = (V*v'+a)';
                            v = [(Projection.A*v'+Projection.b)' v];
                        else
                            Matrix = W{l,1}(t,:)*V;
                            vector = W{l,1}(t,:)*a+bias{l,1}(t);
                            coeff = coefficients(poly, precision(l,t), Matrix, vector); % TBI
                            deg = (size(coeff,1)-1)*ones(1,N);
                            [bcoeff,v_cell] = GeneralizedBernsteinCoeff_nD(coeff,deg,[]);
                            bcoeff = bcoeff';
                            v = [];
                            for idx=1:N
                                v =[v v_cell{idx}'];  
                            end
                            v = (V*v'+a)';
                        end
                        poly_temp = [poly_temp; v bcoeff];
                    end
                end
            end
        end
        [A,b,Aeq,beq] = vert2lcon(poly_temp);
        if ~isempty(A)
            n1 = size(A,1);
            A = [A(:,1:N) zeros(n1,t-1) A(:,N+1) zeros(n1,n_neurons(l)-t)];
        end
        if ~isempty(Aeq)
            n1 = size(Aeq,1);
            Aeq = [Aeq(:,1:N) zeros(n1,t-1) Aeq(:,N+1) zeros(n1,n_neurons(l)-t)];
        end
        poly_new.A = [poly_new.A;A];
        poly_new.b = [poly_new.b;b];
        poly_new.Aeq = [poly_new.Aeq;Aeq];
        poly_new.beq = [poly_new.beq;beq];
    end
    V_new = lcon2vert(poly_new.A,poly_new.b,poly_new.Aeq,poly_new.beq);
    V_new = V_new(:,N+1:end);
    if size(V_new,2)<2
        poly_curr.A = [1 -1];
        poly_curr.b = [max(V_new), min(V_new)];
        poly_curr.Aeq = [];
        poly_curr.beq = [];
    else
        [poly_curr.A,poly_curr.b,poly_curr.Aeq,poly_curr.beq] = vert2lcon(V_new);
    end
end

N = size(poly_curr.A,2);
l = n_layer+1;
if N==1
    asset(size(poly_curr.A,1)==N+1)
    assert(sign(poly_curr.A(1)*poly_curr.A(2))<0)

    interval = poly_curr.b./poly_curr.A;    
    if interval(1)>interval(2)
        interval = [interval(2) interval(1)];
    end
    poly_temp = [interval(1) (W{l,1}*interval(1) + bias{l,1}(t))';...
                interval(1) (W{l,1}*interval(1) + bias{l,1}(t))'];
else
    poly_temp = [];
    P = lcon2vert(poly_curr.A, poly_curr.b, poly_curr.Aeq, poly_curr.beq);
    if ~isempty(P)
        [P_new,Project] = degeneration('vert',P); % 1 CASE TBI
        if size(P_new,1)==1 % single point
            if ~isempty(Project.A)
                point = [Project.A*P_new+Project.b , P_new];
            else
                point = P_new;
            end
            poly_temp = [point*ones(n_neurons(l),1), W{l,1}*point'+ bias{l,1}];
        else
            [indexTR,n_triangle,polytope_new] = My_Triangulation(P_new); 
            for k=1:n_triangle
                vertex = polytope_new(indexTR(k,:),:)';
                V = vertex(:,2:end)-vertex(:,1);
                a = vertex(:,1);
                m = size(vertex,2);
                W_new = W{l,1}*V;
                bias_new = W{l,1}*a + bias{l,1};

                bcoeff = zeros(n_neurons(l), m);
                bcoeff(:,1) = bias_new;
                for idx = 2:m
                    bcoeff(:,idx) = W_new*[zeros(idx-2,1); 1; zeros(m-idx,1)] + bias_new;
                end
                bcoeff = bcoeff';
                if ~isempty(Project.A)
                    project_vertex = Project.A*vertex'+Project.b;
                    v = [project_vertex vertex'];
                else
                    v = vertex';
                end
                poly_temp = [poly_temp;v bcoeff];
            end
        end
    end
end
V_new = poly_temp(:,N+1:end);
if size(V_new,2)<2
    poly_curr.A = [1 -1];
    poly_curr.b = [max(V_new), min(V_new)];
    poly_curr.Aeq = [];
    poly_curr.beq = [];
else
    [poly_curr.A,poly_curr.b,poly_curr.Aeq,poly_curr.beq] = vert2lcon(V_new);
end
polytope = poly_curr;
end

