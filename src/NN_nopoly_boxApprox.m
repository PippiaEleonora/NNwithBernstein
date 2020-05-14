function [box,Btotal] = NN_nopoly_boxApprox(W,bias,n_layer,n_neurons,Domain)
% NN_BOXAPPROXIMATION - a box approximation of a neural network.
%
% box = NN_boxApproximation(W,bias,n_layer,n_neurons,Domain) return
% an hyper-rectangular domanin box (NX2) with N the number of output.
%
%       Input description:
%       - W: is a cell structure (mX1) with the weight matrix for each layer
%       - bias: is a cell structure (mX1) with the bias vector for each layer
%       - n_layer: is the number of layers (hidden layers + output layer)
%       - n_neurons: is a vector with the number of neurons for each layer
%                   (again hidden layers + output layer)
%       - Domain: is the rectangular domain of the input layer (nX2 with n
%                 the number of input). The i-th row is an interval [lb,ub] 
%                 for the i-th input variable
%
% [box,Btotal] = NN_boxApproximation(poly,W,bias,n_layer,n_neurons,x,Domain) 
% return the hyper-rectangular domain 'box' and return also a cell structure 
% with the hyper-rectangular approximation for each layer
%
%
% Comments inside the code are referring to the document:
%           doc/Hybrid_Systems_with_NN.pdf
%


% box_curr is the current domain, at each layer
% Btotal is an output and it saves the current domain
box_curr = Domain;
Btotal = cell(1,n_layer+2);
Btotal{1} = box_curr;

% For-loop for the hidden layers
for l=1:n_layer
    % We split the computation of the new box in two steps, the matrix
    % product y=Wx+bias and the activation function z=poly(y) as descrived
    % in equations (37) and (38)
    
    % FIRST STEP (y=Wx+bias)
    % Computes the first transformation according to equation (39) and (40) 
    % (in the following W_new is W^* and bias_new is a^* of the document notations)
    b_matrix = ((box_curr(:,2)-box_curr(:,1)).*ones(size(box_curr,1), n_neurons(l)))';
    W_new = W{l,1}.*b_matrix;
    bias_new = W{l,1}*box_curr(:,1) + bias{l,1};

    
    % Computation of the bounds according to equation (42)
    valpos = W_new;
    valneg = valpos;
    valpos(valpos<0)=0;
    valneg(valneg>0)=0;
    box_update = [sum(valneg,2)+bias_new, sum(valpos,2)+bias_new]; 
    % box_update is the bound for the variable y, so the domain of the
    % activation function
     
    % SECOND STEP (z=tanh(y)) 
    % box_curr is the overapproximation obtained with Bernstein, is a matrix
    % nX2 with n=number of neurons. The output z_i = tanh(y) is inside the 
    % interval z_i \in [box_new(i,1), box_new(i,2)]
    box_curr = tansig(box_update);
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

