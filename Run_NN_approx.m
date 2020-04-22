clear all
close all
clc


%% Create NN or load an existing one
ifload = 1;
if ifload
    load('test/exp')
else
    Domain = [-5 7];
    inputs = rand(1,100)*(Domain(2)-Domain(1))+Domain(1);
    % targets = exp(inputs);
    targets = (inputs).^3;

    n_neurons = [2 5]; %number of neurons per layer
    n_layer = length(n_neurons); %number of hidden layers

    net = createNN(inputs,targets,n_neurons); %neural network
    n_neurons = [n_neurons 1];
    % gensim(net); %to generate a simulink block
end


%%
if ~exist('n_layer','var')
    n_layer = length(net.IW)-1;
    n_neurons = zeros(1,n_layer+1);
end

IN.lb = net.inputs{1}.range(:,1);
IN.ub = net.inputs{1}.range(:,2);

OUT.lb = net.outputs{n_layer+1}.range(:,1);
OUT.ub = net.outputs{n_layer+1}.range(:,2);

W = cell(n_layer+1,1);
bias = net.b;                                                
for l=1:n_layer+1
    if l==1
        W{l,1} = net.IW{1};
        n_neurons(l) = size(W{l,1},1);
    else
        W{l,1} = net.LW{l,l-1};
        n_neurons(l) = size(W{l,1},1);
    end
end

%% Create approximation of tanh
answer = questdlg('Select type of approximation:','Type of approximate', ...
	'Polynomial','Chebyshev','Rational','Polynomial');

deg = [9 9];
Iconfid = [-4 4];
[poly,error] = createApprox(answer,deg,1,Iconfid,0);


%% Box Overapproximation
tic
Domain_new = (Domain-IN.lb).*2./(IN.ub-IN.lb) -1;
z = sym('z');
[box_old, B] = NN_boxApproximation(poly,W,bias,n_layer,n_neurons,z,Domain_new, Iconfid);
box1 = (box_old+1).*(OUT.ub-OUT.lb)./2 + OUT.lb
lunghezza1 = box1(2)-box1(1)
toc
%% Box Overapproximation splitting the domain
tic
box_split = box1;
lunghezza_old = (box1(:,2)-box1(:,1)).*(Domain(:,2)-Domain(:,1));
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
%     sum(lunghezza)/sum(lunghezza_old)
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

%% FIX Box Splitting
split = 20;
Domain_split = zeros(split,2);
box_cur = zeros(split,2);
step_split = (Domain(2)-Domain(1))/split;
lunghezza = zeros(split,1);

tic
for i=1:split
    Domain_temp = [Domain(1)+step_split*(i-1) Domain(1)+step_split*(i)];
    Domain_split(i,:) = (Domain_temp-IN.lb).*2./(IN.ub-IN.lb) -1;

    [box_temp] = NN_boxApproximation(poly,W,bias,n_layer,[n_neurons 1],z,Domain_split(i,:), Iconfid);
    box_cur(i,:) = (box_temp+1).*(OUT.ub-OUT.lb)./2 + OUT.lb;
    lunghezza(i) = (box_cur(i,2)-box_cur(i,1))*step_split;
end
box_split = box_cur;
toc

%% Polytope approximation
% Function to convert vertex into linear constraints and viceversa do not
% work correctly
%
% 
Domain_new = (Domain-IN.lb).*2./(IN.ub-IN.lb) -1;
for i=1:size(Domain_new,1)
    if Domain_new(i,1)>Domain_new(i,2)
        temp = Domain_new(i,1);
        Domain_new(i,1) = Domain_new(i,2);
        Domain_new(i,2) = temp;
    end
end
D.A = [eye(size(Domain_new,1)); -eye(size(Domain_new,1))];
D.b = [Domain_new(:,2); -Domain_new(:,1)];
D.Aeq = [];
D.beq = [];
x=sym('x',[1,max([n_neurons,size(Domain,1),1])]); %WARNING: too much, maybe 
                                                % we create different vectors 
                                                % for each layer
tic
[polytope_old] = NN_polyApproximation(poly,x,[],Iconfid,W,bias,n_layer,[n_neurons 1],D);
toc

polytope = (polytope_old.b+1).*(OUT.ub-OUT.lb)./2 + OUT.lb;

%% OLD IMPLEMENTATION
Domain_new = (Domain-IN.lb).*2./(IN.ub-IN.lb) -1;
x=sym('x',[1,max([n_neurons,size(Domain,1)])]); %WARNING: too much, maybe 
                                                % we create different vectors 
                                                % for each layer
Ppolynomial = cell(n_layer+1,max(n_neurons));
for l=1:n_layer+1
    for k=1:n_neurons(l)
        clear f
        if l==1
            f(x(1:size(Domain_new,1))) = poly(W{l,1}(k,:)*transpose(x(1:size(Domain_new,1)))+bias{l,1}(k));
        elseif l==n_layer+1
            f(x(1:n_neurons(l-1))) = W{l,1}(k,:)*transpose(x(1:n_neurons(l-1)))+bias{l,1}(k);
        else
            f(x(1:n_neurons(l-1))) = poly(W{l,1}(k,:)*transpose(x(1:n_neurons(l-1)))+bias{l,1}(k));
        end
        Ppolynomial{l,k} = f;
    end
end

tic
[polytope_old] = SimplexPOLYapproximation(Ppolynomial,poly,W,bias,n_layer+1,n_neurons,x,Domain_new');
toc

polytope = (polytope_old+1).*(OUT.ub-OUT.lb)./2 + OUT.lb;
 

%% Plot results for single input
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
    plot(linspace(Domain(1), Domain(2), 100), net(linspace(Domain(1), Domain(2))), 'blue','LineWidth',2)
    
%     if exist('vector_legend','var')
%         clear vector_legend
%     end
%     vector_legend{1} = 'tanh';
    cur = 1;
    if exist('box1','var')
        hold on
        plot(linspace(Domain(1), Domain(2), 100), box1(1)*ones(1,100), 'black','LineWidth',2)
        hold on
        plot(linspace(Domain(1), Domain(2), 100), box1(2)*ones(1,100), 'black','LineWidth',2)
%         vector_legend{cur+1} = 'box approximation';
%         vector_legend{cur+2} = ' ';
%         cur = cur+1;
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
        end
        k = convhull(vertex_split);
        hold on
        plot(vertex_split(k,1), vertex_split(k,2),'r','LineWidth',2)
%         vector_legend{cur+1} = 'convex hull of the intervals';
%         vector_legend{cur+2} = 'split intervals';
%         cur = cur+1;
        for i=1:split
            Domain_temp = [Domain(1)+step_split*(i-1) Domain(1)+step_split*(i)];
            hold on
            plot(linspace(Domain_temp(1), Domain_temp(2), 100), box_split(i,1)*ones(1,100), 'black')
            hold on
            plot(linspace(Domain_temp(1), Domain_temp(2), 100), box_split(i,2)*ones(1,100), 'black')
        end
    end

    if exist('polytope','var')
        hold on
        plot(linspace(Domain(1), Domain(2), 100), min(polytope)*ones(1,100), 'red--','LineWidth',2)
        hold on
        plot(linspace(Domain(1), Domain(2), 100), max(polytope)*ones(1,100), 'red--','LineWidth',2)
%         vector_legend{cur+1} = 'lower bound with polytope';
%         vector_legend{cur+2} = 'upper bound with polytope';
%         cur = cur+2;
    end
%     legend(vector_legend)
end
