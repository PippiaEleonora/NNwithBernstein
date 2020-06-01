clear all
close all
clc


%% Load an existing NN
load('test/exp23')
% load('test/exp')
% load('test/cubic')
% load('test/foxholes1_17042020')

% load('test/NN_bad_design')
% load('test/NN_ok_design')


%%
n_layer = length(net.IW)-1;
n_neurons = zeros(1,n_layer+1);

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

% Define domain for NN_bad and NN_ok
if ~exist('Domain','var')
    noiseFraction = 0.001;
    ref = -0.2;
    noise = ref*noiseFraction;
    y = [ref-noise, ref+noise];
    u = y.*10;
    Domain = [[ref ref].*ones(4,1); y.*ones(4,1);  u.*ones(2,1)];
end

%% Create approximation of tanh
answer = questdlg('Select type of approximation:','Type of approximate', ...
	'Polynomial','Chebyshev','Rational','Polynomial');

deg = [9 9];
Iconfid = [-4 4];
ifplot = 0;
ifconfidence = 0;
[poly,error] = createApprox(answer,deg,ifplot,Iconfid,ifconfidence);


%% Box Overapproximation
tic
Domain_new = (Domain-IN.lb).*2./(IN.ub-IN.lb) -1;
z = sym('z');
[box_old, B] = NN_boxApproximation(poly,W,bias,n_layer,n_neurons,z,Domain_new, Iconfid);
box1 = (box_old+1).*(OUT.ub-OUT.lb)./2 + OUT.lb
lunghezza1 = box1(2)-box1(1)
toc


%% Box Overapproximation (without Bernstein)
tic
Domain_new = (Domain-IN.lb).*2./(IN.ub-IN.lb) -1;
[box_old] = NN_nopoly_boxApprox(W,bias,n_layer,n_neurons,Domain_new);
box = (box_old+1).*(OUT.ub-OUT.lb)./2 + OUT.lb;
toc


%% Box Overapproximation splitting the domain
split = 20;
Domain_split = zeros(split,2);
box_cur = zeros(split,2);
step_split = (Domain(2)-Domain(1))/split;
lunghezza = zeros(split,1);

tic
for i=1:split
    Domain_temp = [Domain(1)+step_split*(i-1) Domain(1)+step_split*(i)];
    Domain_split(i,:) = (Domain_temp-IN.lb).*2./(IN.ub-IN.lb) -1;

    [box_temp] = NN_nopoly_boxApprox(W,bias,n_layer,n_neurons,Domain_split(i,:));
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
A = [eye(size(Domain_new,1)); -eye(size(Domain_new,1))];
b = [Domain_new(:,2); -Domain_new(:,1)];
% D.Aeq = [];
% D.beq = [];
x=sym('x',[1,max([n_neurons,size(Domain,1),1])]); %WARNING: too much, maybe 
                                                % we create different vectors 
                                                % for each layer
% tic
% [polytope_old] = NN_polyApproximation(poly,x,[],Iconfid,W,bias,n_layer,[n_neurons 1],D);
% toc
D = Polyhedron(A,b);
tic
[polytope_old] = NN_polyApprox(poly,[],Iconfid,W,bias,n_layer,[n_neurons 1],D,x);
toc

polytope = (polytope_old.b+1).*(OUT.ub-OUT.lb)./2 + OUT.lb;
 

%% Plot results for single input
if size(Domain,1)==1
    Domain_new = (Domain-IN.lb).*2./(IN.ub-IN.lb) -1;
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
    
    clear vector_legend p
    figure
    p(1)=plot(linspace(Domain(1), Domain(2), 100), net(linspace(Domain(1), Domain(2))), 'black','LineWidth',4);
    hold on
    p(2)=plot(linspace(Domain(1), Domain(2), 100), xx, 'r','LineWidth',1);
    vector_legend{1} = 'Neural Network';
    vector_legend{2} = 'Approximation';
    
    cur = 1;
    if exist('box','var')
        hold on
        p(length(p)+1)=plot(linspace(Domain(1), Domain(2), 100), box(1)*ones(1,100), 'blue','LineWidth',2);
        hold on
        plot(linspace(Domain(1), Domain(2), 100), box(2)*ones(1,100), 'blue','LineWidth',2)
        vector_legend{length(vector_legend)+1} = 'box approx';
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
        p(length(p)+1)=plot(vertex_split(k,1), vertex_split(k,2),'-.r','LineWidth',2);
        vector_legend{length(vector_legend)+1} = 'convex hull';
        
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
    legend(p(:),vector_legend)
end
