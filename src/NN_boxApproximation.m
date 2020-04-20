function [box,Btotal] = NN_boxApproximation(poly,W,bias,n_layer,n_neurons,x,Domain,tol_box)
% NN_BOXAPPROXIMATION - a box approximation of a neural network.
%
% box = NN_boxApproximation(poly,W,bias,n_layer,n_neurons,x,Domain) return
% an hyper-rectangular domanin box (NX2) with N the number of output.
%
%       Input description:
%       - poly: is the polynomial or rational approximation of the 
%               activation function 
%       - W: is a cell structure (mX1) with the weight matrix for each layer
%       - b: is a cell structure (mX1) with the bias vector for each layer
%       - n_layer: is the number of layers (hidden layers + output layer)
%       - n_neurons: is a vector with the number of neurons for each layer
%                   (again hidden layers + output layer)
%       - x: is a single simbolic variable 
%       - Domain: is the rectangular domain of the input layer (nX2 with n
%                 the number of input). The i-th row is an interval [lb,ub] 
%                 for the i-th input variable 
%
% [box,Btotal] = NN_boxApproximation(poly,W,bias,n_layer,n_neurons,x,Domain) 
% return the hyper-rectangular domain 'box' and return also a cell structure 
% with the hyper-rectangular approximation for each layer
%

type = 'Garloff'; %'Ray_Nataraj'
[polyP,polyQ] = numden(poly);

coeffP = fliplr(double(coeffs(polyP(x),x,'All')));
coeffQ = fliplr(double(coeffs(polyQ(x),x,'All')));
degP = length(coeffP)-1;
degQ = length(coeffQ)-1;

if degP>degQ
    degTOT = degP;
    coeffQ = [coeffQ, zeros(1,degP-degQ)];
else
    degTOT = degQ;
    coeffP = [coeffP, zeros(1,degQ-degP)];
end

coeffP = coeffP';
coeffQ = coeffQ';

box_curr = Domain;
Btotal = cell(1,n_layer+2);
Btotal{1} = box_curr;
for l=1:n_layer
    b_matrix = ((box_curr(:,2)-box_curr(:,1)).*ones(size(box_curr,1), n_neurons(l)))';
    W_new = W{l,1}.*b_matrix;
    bias_new = W{l,1}*box_curr(:,1) + bias{l,1};

    valpos = W_new;
    valneg = valpos;
    valpos(valpos<0)=0;
    valneg(valneg>0)=0;

    box_update = [sum(valneg,2)+bias_new, sum(valpos,2)+bias_new];
        
    box_new = zeros(n_neurons(l),2);
    
    for t=1:n_neurons(l)
        if ~isempty(tol_box)
            if tol_box(1)>box_update(t,1) && tol_box(2)<box_update(t,2)
                box_new(t,1) = -1;
                box_new(t,2) = 1;
            elseif tol_box(1)>box_update(t,2)
                box_new(t,:) = [-1 -1];
            elseif tol_box(2)<box_update(t,1)
                box_new(t,:) = [1 1];
            elseif tol_box(1)>box_update(t,1) && tol_box(2)>=box_update(t,2)
                box_temp = [tol_box(1), box_update(t,2)];
                    if degQ>0
                        if sign(box_temp(1))*sign(box_temp(2))<0
                            bcoeffP1 = BernsteinCoeff_1D(coeffP,degTOT,[box_temp(1), 0],type);
                            bcoeffQ1 = BernsteinCoeff_1D(coeffQ,degTOT,[box_temp(1), 0],type);
                            bcoeffP2 = BernsteinCoeff_1D(coeffP,degTOT,[0, box_temp(2)],type);
                            bcoeffQ2 = BernsteinCoeff_1D(coeffQ,degTOT,[0, box_temp(2)],type);

                            bcoeffP = [bcoeffP1; bcoeffP2];
                            bcoeffQ = [bcoeffQ1; bcoeffQ2];

                        else
                            bcoeffP = BernsteinCoeff_1D(coeffP,degTOT,box_temp(:),type);
                            bcoeffQ = BernsteinCoeff_1D(coeffQ,degTOT,box_temp(:),type);
                        end
                        assert(sign(min(bcoeffQ))*sign(max(bcoeffQ))>0)
                        bcoeff = bcoeffP./bcoeffQ;
                    else
                        bcoeff = BernsteinCoeff_1D(coeffP,degTOT,box_temp(:),type)./coeffQ(1);
                    end
                box_new(t,1) = -1;
                box_new(t,2) = min(max(max(bcoeff),-1),1);
            elseif tol_box(1)<=box_update(t,1) && tol_box(2)<box_update(t,2)
                box_temp = [box_update(t,1),tol_box(2)];
                    if degQ>0
                        if sign(box_temp(1))*sign(box_temp(2))<0
                            bcoeffP1 = BernsteinCoeff_1D(coeffP,degTOT,[box_temp(1), 0],type);
                            bcoeffQ1 = BernsteinCoeff_1D(coeffQ,degTOT,[box_temp(1), 0],type);
                            bcoeffP2 = BernsteinCoeff_1D(coeffP,degTOT,[0, box_temp(2)],type);
                            bcoeffQ2 = BernsteinCoeff_1D(coeffQ,degTOT,[0, box_temp(2)],type);

                            bcoeffP = [bcoeffP1; bcoeffP2];
                            bcoeffQ = [bcoeffQ1; bcoeffQ2];

                        else
                            bcoeffP = BernsteinCoeff_1D(coeffP,degTOT,box_temp(:),type);
                            bcoeffQ = BernsteinCoeff_1D(coeffQ,degTOT,box_temp(:),type);
                        end
                        assert(sign(min(bcoeffQ))*sign(max(bcoeffQ))>0)
                        bcoeff = bcoeffP./bcoeffQ;
                    else
                        bcoeff = BernsteinCoeff_1D(coeffP,degTOT,box_temp(:),type)./coeffQ(1);
                    end
                box_new(t,1) = max(min(min(bcoeff),1),-1);
                box_new(t,2) = 1;
            else
                box_temp = box_update(t,:);
                    if degQ>0
                        if sign(box_temp(1))*sign(box_temp(2))<0
                            bcoeffP1 = BernsteinCoeff_1D(coeffP,degTOT,[box_temp(1), 0],type);
                            bcoeffQ1 = BernsteinCoeff_1D(coeffQ,degTOT,[box_temp(1), 0],type);
                            bcoeffP2 = BernsteinCoeff_1D(coeffP,degTOT,[0, box_temp(2)],type);
                            bcoeffQ2 = BernsteinCoeff_1D(coeffQ,degTOT,[0, box_temp(2)],type);

                            bcoeffP = [bcoeffP1; bcoeffP2];
                            bcoeffQ = [bcoeffQ1; bcoeffQ2];

                        else
                            bcoeffP = BernsteinCoeff_1D(coeffP,degTOT,box_temp(:),type);
                            bcoeffQ = BernsteinCoeff_1D(coeffQ,degTOT,box_temp(:),type);
                        end
                        assert(sign(min(bcoeffQ))*sign(max(bcoeffQ))>0)
                        bcoeff = bcoeffP./bcoeffQ;
                    else
                        bcoeff = BernsteinCoeff_1D(coeffP,degTOT,box_temp(:),type)./coeffQ(1);
                    end
                box_new(t,1) = max(min(min(bcoeff),1),-1);
                box_new(t,2) = min(max(max(bcoeff),-1),1);
            end
        else
            box_temp = box_update(t,:);
                if degQ>0
                    if sign(box_temp(1))*sign(box_temp(2))<0
                        bcoeffP1 = BernsteinCoeff_1D(coeffP,degTOT,[box_temp(1), 0],type);
                        bcoeffQ1 = BernsteinCoeff_1D(coeffQ,degTOT,[box_temp(1), 0],type);
                        bcoeffP2 = BernsteinCoeff_1D(coeffP,degTOT,[0, box_temp(2)],type);
                        bcoeffQ2 = BernsteinCoeff_1D(coeffQ,degTOT,[0, box_temp(2)],type);

                        bcoeffP = [bcoeffP1; bcoeffP2];
                        bcoeffQ = [bcoeffQ1; bcoeffQ2];

                    else
                        bcoeffP = BernsteinCoeff_1D(coeffP,degTOT,box_temp(:),type);
                        bcoeffQ = BernsteinCoeff_1D(coeffQ,degTOT,box_temp(:),type);
                    end
                    assert(sign(min(bcoeffQ))*sign(max(bcoeffQ))>0)
                    bcoeff = bcoeffP./bcoeffQ;
                else
                    bcoeff = BernsteinCoeff_1D(coeffP,degTOT,box_temp(:),type)./coeffQ(1);
                end
            box_new(t,1) = max(min(min(bcoeff),1),-1);
            box_new(t,2) = min(max(max(bcoeff),-1),1);
        end

        
    end
    box_curr = box_new;
    Btotal{l+1} = box_curr;  
end

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

