function [box,Btotal] = NN_boxApprox(poly,precision,W,bias,n_layer,n_neurons,Domain,x)
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

type = 'Garloff'; %'Ray_Nataraj'
[polyP,polyQ] = numden(poly); % extract numerator and demonimator, it works 
                              % even if we have a polynomial expression, 
                              % in this case polyQ is a constant

% Extraction of the coefficients in power-ascendent order:
% p(x) = coeffP[1] + coeffP[2]*x + coeffP[3]*x^2 + ...
coeffP = fliplr(double(coeffs(polyP(x(1)),x(1),'All'))); 
coeffQ = fliplr(double(coeffs(polyQ(x(1)),x(1),'All')));
degP = length(coeffP)-1;
degQ = length(coeffQ)-1;

% coeffP and coeffQ must have the same length, we add a 0 if needed
if degP>degQ
    degTOT = degP;
    coeffQ = [coeffQ, zeros(1,degP-degQ)];
else
    degTOT = degQ;
    coeffP = [coeffP, zeros(1,degQ-degP)];
end

coeffP = coeffP';
coeffQ = coeffQ';

% box_curr is the current domain, at each layer
% Btotal is an output and it saves the current domain
box_curr = Domain;
Btotal = cell(1,n_layer+2);
Btotal{1} = box_curr;

for l=1:n_layer
    N = size(box_curr,1); %N is the number of input variables
    box_new = zeros(n_neurons(l),2);

    interval = box_curr;
    interval_new = W{l,1}*interval + bias{l,1};    
    interval_new = sort(interval_new,2);
    
%     poly_curr = Polyhedron('lb',box_curr(:,1),'ub',box_curr(:,2));
    for t=1:n_neurons(l)
        bcoeff = [];  
        if sign(interval_new(t,1))*sign(interval_new(t,2))<0 && N>1     
            % We need to split around 0
            A = -W{l,1}(t,:);
            b = bias{l,1}(t);
%             A = [poly_curr.A; -W{l,1}(t,:)];
%             b = [poly_curr.b; bias{l,1}(t)];
            Pplus = Polyhedron('A',A,'b',b,'lb',box_curr(:,1),'ub',box_curr(:,2));

            A =  W{l,1}(t,:);
            b =  -bias{l,1}(t);
            Pminus = Polyhedron('A',A,'b',b,'lb',box_curr(:,1),'ub',box_curr(:,2));
            
            % We compute the Berstein coefficients for both
            if ~Pplus.isEmptySet
                P = Pplus;
                if isempty(P.Ae)
                    P_new = P.V;
                    Projection = [];
                else
                    [P_new,Projection] = degeneration('vert',P.V); % degeneration so is not full dimension, refering 
                end                                         % to overleaf (Answers 1 for the degenerate case) (1 CASE TBI)
                if size(P_new,1)==1 % single point
                    point = [Projection.A*P_new+Projection.b , P_new];
                    bcoeff = [bcoeff; evaluation(poly, precision(l,t), W{l,1}(t,:)*point'+ bias{l,1}(t))];
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
                        else
                            Matrix = W{l,1}(t,:)*V;
                            vector = W{l,1}(t,:)*a+bias{l,1}(t);
                        end
                        coeffP1 = multiCoeff(Matrix, vector,coeffP,'first');
                        coeffQ1 = multiCoeff(Matrix, vector,coeffQ,'first');
                        if N==1
                            bcoeffP = BernsteinCoeff_1D(coeffP1,degTOT,[0 1],'Garloff');
                            bcoeffQ = BernsteinCoeff_1D(coeffQ1,degTOT,[0 1],'Garloff');
                        else
                            bcoeffP = GeneralizedBernsteinCoeff_nD(coeffP1,degTOT,[]);
                            bcoeffP = bcoeffP';
                            bcoeffQ = GeneralizedBernsteinCoeff_nD(coeffQ1,degTOT,[]);
                            bcoeffQ = bcoeffQ';
                        end
                        bcoeff = [bcoeff; bcoeffP./bcoeffQ];
                    end
                end
            end
            % We compute the Berstein coefficients for both
            if ~Pminus.isEmptySet
                P = Pminus;
                if isempty(P.Ae)
                    P_new = P.V;
                    Projection = [];
                else
                    [P_new,Projection] = degeneration('vert',P.V); % degeneration so is not full dimension, refering 
                end                                         % to overleaf (Answers 1 for the degenerate case) (1 CASE TBI)
                if size(P_new,1)==1 % single point
                    point = [Projection.A*P_new+Projection.b , P_new];
                    bcoeff = [bcoeff; evaluation(poly, precision(l,t), W{l,1}(t,:)*point'+ bias{l,1}(t))];
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
                        else
                            Matrix = W{l,1}(t,:)*V;
                            vector = W{l,1}(t,:)*a+bias{l,1}(t);
                        end
                        coeffP1 = multiCoeff(Matrix, vector,coeffP,'first');
                        coeffQ1 = multiCoeff(Matrix, vector,coeffQ,'first');
                        if N==1
                            bcoeffP = BernsteinCoeff_1D(coeffP1,degTOT,[0 1],'Garloff');
                            bcoeffQ = BernsteinCoeff_1D(coeffQ1,degTOT,[0 1],'Garloff');
                        else
                            bcoeffP = GeneralizedBernsteinCoeff_nD(coeffP1,degTOT*ones(1,N),[]);
                            bcoeffP = bcoeffP';
                            bcoeffQ = GeneralizedBernsteinCoeff_nD(coeffQ1,degTOT*ones(1,N),[]);
                            bcoeffQ = bcoeffQ';
                        end
                        bcoeff = [bcoeff; bcoeffP./bcoeffQ];
                    end
                end
            end
        elseif sign(interval_new(t,1))*sign(interval_new(t,2))<0 && N==1 
            coeffP1 = multiCoeff(W{l,1}(t,:), bias{l,1}(t),coeffP,'last');
            coeffQ1 = multiCoeff(W{l,1}(t,:), bias{l,1}(t),coeffQ,'last');
            
            bcoeffP = BernsteinCoeff_1D(coeffP1,degTOT,[box_curr(1,1),0],type);
            bcoeffQ = BernsteinCoeff_1D(coeffQ1,degTOT,[box_curr(1,1),0],type);
            bcoeff = [bcoeff; bcoeffP./bcoeffQ];
            
            bcoeffP = BernsteinCoeff_1D(coeffP1,degTOT,[0, box_curr(1,2)],type);
            bcoeffQ = BernsteinCoeff_1D(coeffQ1,degTOT,[0, box_curr(1,2)],type);
            bcoeff = [bcoeff; bcoeffP./bcoeffQ];
        else
            coeffP1 = multiCoeff(W{l,1}(t,:), bias{l,1}(t),coeffP,'last');
            coeffQ1 = multiCoeff(W{l,1}(t,:), bias{l,1}(t),coeffQ,'last');
            if N==1
                bcoeffP = BernsteinCoeff_1D(coeffP1,degTOT,box_curr,type);
                bcoeffQ = BernsteinCoeff_1D(coeffQ1,degTOT,box_curr,type);
            else
                bcoeffP = BernsteinCoeff_nD(coeffP1,degTOT*ones(1,N),box_curr,type);
                bcoeffP = bcoeffP';
                bcoeffQ = BernsteinCoeff_nD(coeffQ1,degTOT*ones(1,N),box_curr,type);
                bcoeffQ = bcoeffQ';
            end
            bcoeff = [bcoeff; bcoeffP./bcoeffQ];
        end
        
        box_new(t,:) = min(max([min(bcoeff),max(bcoeff)],-1),1);
    end

    box_curr = box_new;
    Btotal{l+1} = box_curr;
end
% LAST STEP: from the last hidden layer to the output we just have the
% linear function y=Wx+bias
b_matrix = ((box_curr(:,2)-box_curr(:,1)).*ones(size(box_curr,1), n_neurons(n_layer+1)))';
W_new = W{n_layer+1,1}.*b_matrix;
bias_new = W{n_layer+1,1}*box_curr(:,1) + bias{n_layer+1,1}';

valpos = W_new;
valneg = valpos;
valpos(valpos<0)=0;
valneg(valneg>0)=0;

box_curr = [sum(valneg,2)+bias_new, sum(valpos,2)+bias_new];
        
Btotal{n_layer+2} = box_curr;
box = box_curr;
end

