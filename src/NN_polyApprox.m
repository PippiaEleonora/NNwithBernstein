function [polytope] = NN_polyApprox(poly,precision,Iconfident,W,bias,n_layer,n_neurons,Domain,x)
%NN_ABSTRACT_POLYAPPROXIMATION  - a convex approximation of a neural
%network.
%
% polytope = NN_ABSTRACT_polyApproximation(poly,precision,Iconfident,W,
% bias,n_layer,n_neurons,Domain) return a polytope structure that
% overapproximate the output of a neural network.
% 
% polytope is a structure defined with 4 elements: polytope.A, polytope.b,
% polytope.Aeq and polytope.beq
%   each output of the nerual network z = NN(x) satisfy
%           polytope.A*z  <= polytope.b
%           polytope.Aeq*z == polytope.beq
%
%       Input description:
%       - poly: is the polynomial or rational approximation of the 
%               activation function 
%       - precision: is a matrix with the degree of poly for each neuron in
%                    each layer
%       - Iconfident: if the poly function is a piecewise function, we 
%                 consider the poly expression inside Iconfident and outside 
%                 we consider constantly -1 or 1.
%       - W: is a cell structure (mX1) with the weight matrix for each layer
%       - bias: is a cell structure (mX1) with the bias vector for each layer
%       - n_layer: is the number of layers (hidden layers + output layer)
%       - n_neurons: is a vector with the number of neurons for each layer
%                   (again hidden layers + output layer)
%       - x: is a single simbolic variable 
%       - Domain: is a structure with the equalities and inequalities that
%                 define the convex domain
%

poly_curr = Domain;

if isempty(precision)
    coeffP = fliplr(double(coeffs(poly(x(1)),x(1),'All')));
end

for l=1:n_layer
    N = poly_curr.Dim; %N is the number of input variables
    
    poly_new.A = [];
    poly_new.b = [];
    poly_new.Aeq = [];
    poly_new.beq = [];
    for t=1:n_neurons(l)
        if N==1 %we have a single variable, so the domain is an interval
            interval = poly_curr.V;
            interval_new = W{l,1}(t,:)*interval + bias{l,1}(t);    
            if interval_new(1)>interval_new(2)
                interval_new = [interval_new(2) interval_new(1)];
            end
            
            % we split the interval considering the interval of confidence
            % of the polynomial function (Iconfident)
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
%                 coeffP = coefficients(poly, precision(l,t), [], []); % TBI, we need an efficient function to extract the coefficients
                [bcoeff, v] = BernsteinCoeff_1D(coeffP,length(coeffP)-1,P,'Garloff');
                I_split = (v - bias{l,1}(t))/W{l,1}(t,:);
                poly_temp = [poly_temp; I_split' bcoeff]; 
            end
        else
            poly_temp = [];  %inside poly_temp we collect all the vertices 
            % of the polytope that approximate the input-output function
            
            % We need to split according to the interval of confidence for
            % the polynomial function
            A = [poly_curr.A; -W{l,1}(t,:)];
            b = [poly_curr.b; bias{l,1}(t)-Iconfident(2)];
            Pplus = Polyhedron(A,b);
            if ~Pplus.isEmptySet
                poly_temp = [poly_temp; Pplus.V ones(size(Pplus.V,1),1)];
            end
            
            A =  [poly_curr.A; W{l,1}(t,:)];
            b =  [poly_curr.b; -bias{l,1}(t)+Iconfident(1)];
            Pminus = Polyhedron(A,b);
            if ~Pminus.isEmptySet
                poly_temp = [poly_temp; Pminus.V -ones(size(Pminus.V,1),1)];
            end
            
            A = [poly_curr.A; W{l,1}(t,:); -W{l,1}(t,:)];
            b = [poly_curr.b;-bias{l,1}(t)+Iconfident(2); bias{l,1}(t)-Iconfident(1)];
            P = Polyhedron(A,b);
            % We compute the Berstein coefficients just for the vertices of
            % P, in the other cases the function is constantly -1 o 1
            if ~P.isEmptySet
                if isempty(P.Ae)
                    P_new = P.V;
                    Projection = [];
                else
                    [P_new,Projection] = degeneration('vert',P.V); % degeneration so is not full dimension, refering 
                end                                         % to overleaf (Answers 1 for the degenerate case) (1 CASE TBI)
                if size(P_new,1)==1 % single point
                    point = [Projection.A*P_new+Projection.b , P_new];
                    poly_temp = [point, evaluation(poly, precision(l,t), W{l,1}(t,:)*point'+ bias{l,1}(t))]; %TBI
                else
                    % Function that triangulates 
                    T = triangulate(Polyhedron('V',P_new));
                    n_triangle = size(T,2);
                    for k=1:n_triangle
                        vertex = transpose(T(k).V);
                        % See transformation (52) on overleaf
                        V = vertex(:,2:end)-vertex(:,1);    % matrix to convert the simplex into the stardard simplex
                        a = vertex(:,1);                    % vector to convert the simplex into the stardard simplex
                        if size(P_new,2)<N % degenerate case
                            m = size(P_new,2);
                            assert(size(P_new,2)==size(Projection.A,2))
                            
                            Matrix = W{l,1}(t,:)*[Projection.A; eye(m)]*V;
                            vector = W{l,1}(t,:)*[Projection.A; eye(m)]*a + W{l,1}(t,:)*[Projection.b; zeros(m,1)]+bias{l,1}(t);

%                             coeff = coefficients(poly, precision(l,t), Matrix, vector); % TBI coefficients of the function poly(Matrix*x + vector) with 
                            coeff = multiCoeff(Matrix, vector,coeffP,'first');                                                            % x \in Standard simplex (unit)
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
%                             coeff = coefficients(poly, precision(l,t), Matrix, vector); % TBI
                            coeff = multiCoeff(Matrix, vector,coeffP,'first');
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
        % The function f:x->z_i inside the convexhull poly_temp. (where 
        % z_i is the i-th neuron)
        % For each neuron we need to extract and collect the contraints
        polytope = Polyhedron('V',poly_temp);
        A = polytope.A;
        b = polytope.b;
        Aeq = polytope.Ae;
        beq = polytope.be;
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
    % poly_new contain the relation between input-output at each layer, we
    % need to project away the input variables.
    if ~isempty(Aeq)
        polytope = Polyhedron('A',poly_new.A,'b',poly_new.b,...
                          'Ae',poly_new.Aeq,'be',poly_new.beq);
    else
        polytope = Polyhedron('A',poly_new.A,'b',poly_new.b);
    end
    M = N+n_neurons(l);
    poly_curr = polytope.projection([N+1:M],'mplp'); %THAO (this is heavy)
    poly_curr = Polyhedron('A',poly_curr.A,'b',poly_curr.b,'Ae',poly_curr.Ae,...
        'be',poly_curr.be,'lb',-ones(1,M-N),'ub',ones(1,M-N));
end

N = size(poly_curr.A,2);
l = n_layer+1;
if N==1
    interval = poly_curr.P;    
    if interval(1)>interval(2)
        interval = [interval(2) interval(1)];
    end
    poly_temp = [interval(1) (W{l,1}*interval(1) + bias{l,1}(t))';...
                interval(1) (W{l,1}*interval(1) + bias{l,1}(t))'];
else
    poly_temp = [];
    P = poly_curr;
    if ~isempty(P)
        if isempty(P.Ae)
            P_new = P.V;
            Project = [];
        else
            [P_new,Project] = degeneration('vert',P.V); % degeneration so is not full dimension, refering 
        end 
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
    polytope = Polyhedron('lb',min(V_new),'ub',max(V_new));
else
    polytope = Polyhedron('V',V_new);
end
end

