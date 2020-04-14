clear all
close all
clc

%% Create NN
Domain = [-1 7];
inputs = rand(1,100)*(Domain(2)-Domain(1))+Domain(1);
targets = exp(inputs);

n_neurons = [3 2]; %number of neurons per layer
n_layer = length(n_neurons); %number of hidden layers

OUT.lb = min(targets);
OUT.ub = max(targets);

IN.lb = min(inputs);
IN.ub = max(inputs);


net = createNN(inputs,targets,n_neurons); %neural network
% gensim(net); %to generate a simulink block

W = cell(n_layer+1,1);
bias = net.b;                                                
for l=1:n_layer+1
    if l==1
        W{l,1} = net.IW{1};
    else
        W{l,1} = net.LW{l,l-1};
    end
end

%% Create approximation of tanh
answer = questdlg('Select type of approximation:','Type of approximate', ...
	'Polynomial','Chebyshev','Rational','Polynomial');

deg = [9 9];
poly = createApprox(answer,deg,0);
Iconfid = [-4 4];

%% Box Overapproximation
tic
Domain_new = (Domain-IN.lb).*2./(IN.ub-IN.lb) -1;
z = sym('z');
[box_old, B] = NN_boxApproximation(poly,W,bias,n_layer,[n_neurons 1],z,Domain_new, Iconfid);
box1 = (box_old+1).*(OUT.ub-OUT.lb)./2 + OUT.lb;
lunghezza1 = box1(2)-box1(1)
toc
%% Box Overapproximation splitting the domain
tic
box_split = box1;
lunghezza_old = (box1(2)-box1(1))*(Domain(2)-Domain(1));
Domain_cur = Domain;
progress = 1;
steps = 1;
totstep = 20;
split = 2;
while progress && steps<totstep
    Domain_split = zeros(split,2);
    box_cur = zeros(split,2);
    step_split = (Domain(2)-Domain(1))/split;
    lunghezza = zeros(split,1);
    for i=1:split
        Domain_temp = [Domain_cur(1)+step_split*(i-1) Domain_cur(1)+step_split*(i)];
        Domain_split(i,:) = (Domain_temp-IN.lb).*2./(IN.ub-IN.lb) -1;
    
        [box_temp] = NN_boxApproximation(poly,W,bias,n_layer,[n_neurons 1],z,Domain_split(i,:), Iconfid);
        box_cur(i,:) = (box_temp+1).*(OUT.ub-OUT.lb)./2 + OUT.lb;
        lunghezza(i) = (box_cur(i,2)-box_cur(i,1))*step_split;
    end
    sum(lunghezza)/sum(lunghezza_old)
    if sum(lunghezza)/sum(lunghezza_old)>1 && steps>2
        progress = 0;
    else
        progress = 1;
        box_split = box_cur;
        lunghezza_old = lunghezza;
    end
    
    split = split + 1;
    steps = steps+1;
end
if steps==totstep
    display('Max number of steps')
else
    display('No more progress')
end

toc

%% Polytope approximation
% Function to convert vertex into linear constraints and viceversa do not
% work correctly
%
% 
Domain_new = (Domain-IN.lb).*2./(IN.ub-IN.lb) -1;
[D.A,D.b,D.Aeq,D.beq]=vert2lcon(Domain_new');
x=sym('x',[1,max([n_neurons,size(Domain,1),1])]); %WARNING: too much, maybe 
                                                % we create different vectors 
                                                % for each layer
tic
[polytope_old] = NN_polyApproximation(poly,x,[],Iconfid,W,bias,n_layer,[n_neurons 1],D);
toc

polytope = (polytope_old.b+1).*(OUT.ub-OUT.lb)./2 + OUT.lb;
%%
if size(Domain,1)==1
    xx = linspace(Domain_new(1), Domain_new(2), 100);
    for l=1:n_layer
        yy = W{l}*xx+bias{l};
        xx = double(poly(W{l}*xx+bias{l}));
        for t=1:n_neurons(l)
            xx(t,yy(t,:)>Iconfid(2)) = 1;
            xx(t,yy(t,:)<Iconfid(1)) = -1;
        end
    end
    xx = W{n_layer+1}*xx+bias{n_layer+1};
    xx  = (xx+1).*(OUT.ub-OUT.lb)./2 + OUT.lb;
    polyApprox = [min(xx), max(xx)]
    netApprox = [min(net(linspace(Domain(1), Domain(2)))),max(net(linspace(Domain(1), Domain(2))))]
    figure
    plot(linspace(Domain(1), Domain(2), 100), xx, 'r')
    hold on
    plot(linspace(Domain(1), Domain(2), 100), net(linspace(Domain(1), Domain(2))), 'blue')
    if exist('box1','var')
        hold on
        plot(linspace(Domain(1), Domain(2), 100), box1(1)*ones(1,100), 'black')
        hold on
        plot(linspace(Domain(1), Domain(2), 100), box1(2)*ones(1,100), 'black')
    end
    if exist('box_split','var')
        split = size(box_split,1);
        step_split = (Domain(2)-Domain(1))/split;
        vertex_split = zeros(split*4,2);
        for i=1:split
            vertex_split(4*i-3,:) = [Domain(1)+step_split*(i-1) box_split(i,1)];
            vertex_split(4*i-2,:) = [Domain(1)+step_split*(i-1) box_split(i,2)];
            vertex_split(4*i-1,:) = [Domain(1)+step_split*(i) box_split(i,1)];
            vertex_split(4*i,:) = [Domain(1)+step_split*(i) box_split(i,2)];
            Domain_temp = [Domain(1)+step_split*(i-1) Domain(1)+step_split*(i)];
%             plot(linspace(Domain_temp(1), Domain_temp(2), 100), box_split(i,1)*ones(1,100), 'red--')
%             hold on
%             plot(linspace(Domain_temp(1), Domain_temp(2), 100), box_split(i,2)*ones(1,100), 'red--')
        end
        k = convhull(vertex_split);
        hold on
        plot(vertex_split(k,1), vertex_split(k,2))
    end

    if exist('polytope','var')
        hold on
        plot(linspace(Domain(1), Domain(2), 100), polytope(1)*ones(1,100), 'red--')
        hold on
        plot(linspace(Domain(1), Domain(2), 100), polytope(2)*ones(1,100), 'red--')
    end
end
