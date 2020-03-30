%% NN
Domain = [0 7];
inputs = rand(1,100)*(Domain(2)-Domain(1))+Domain(1);
targets = exp(inputs);

% Create a Fitting Network
hiddenLayerSize = 20;
hiddenLayerNumber = 2;

net = feedforwardnet(repmat(hiddenLayerSize,1,hiddenLayerNumber),'trainlm');
for i=1:hiddenLayerNumber
    net.layers{i}.transferFcn = 'tansig';
end
net.divideFcn = 'divideind';

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainInd = 1:2:length(inputs);
net.divideParam.valInd = 2:4:length(inputs);
net.divideParam.testInd = 4:4:length(inputs);

% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
performance = perform(net,targets,outputs);

errors = gsubtract(outputs,targets);
disp(performance);
ploterrhist(errors,'bins',100)


OUTlb = min(targets);
OUTub = max(targets);

INlb = min(inputs);
INub = max(inputs);


Domain_new = (Domain-INlb).*2./(INub-INlb) -1;
% gensim(net);

%%

n_layer = hiddenLayerNumber+1; %number of layers
n_neurons = [hiddenLayerSize*ones(1,hiddenLayerNumber) 1]; %number of neurons per layer

z = sym('z');
x=sym('x',[1,max([n_neurons,size(Domain,1)])]); %WARNING: too much, maybe 
                                                % we create different vectors 
                                                % for each layer
W = cell(n_layer,1);
bias = net.b;                                                
for l=1:n_layer
    if l==1
        W{l,1} = net.IW{1};
    else
        W{l,1} = net.LW{l,l-1};
    end
end

answer = questdlg('Select type of approximation:','Type of approximate', ...
	'Polynomial','Rational','Polynomial');

switch answer
    case 'Polynomial'
        maxdeg = 9;
        f(z) = tanh(0)*z^0;
        d(z) = tanh(z);
        for k=1:maxdeg
            d(z) = diff(d(z));
            f(z) = f(z)+d(0)*z^k;
        end
        poly = simplify(f);
    case 'Rational'
        maxdeg = 10;
        const = 3:2:maxdeg*2;
        f(z) = z^2/const(end);
        for k=length(const)-1:-1:1
            f(z) = z^2/(const(k)+f(z));
        end
        f(z) = z/(1+f(z));

        poly = simplify(f);
end
%%
tic
[box_old, B] = NN_boxApproximation(poly,W,bias,n_layer,n_neurons,z,Domain_new);
box1 = (box_old+1).*(OUTub-OUTlb)./2 + OUTlb;
toc

%%
if size(Domain,1)==1 && n_neurons(end)==1
    xx = linspace(Domain_new(1), Domain_new(2), 100);
    for l=1:n_layer-1
        xx = eval(poly(W{l}*xx+bias{l}));
        xx = min(xx,1);
        xx = max(xx,-1);
    end
    xx = W{n_layer}*xx+bias{n_layer};
    xx  = (xx+1).*(OUTub-OUTlb)./2 + OUTlb;
    polyApprox = [min(xx), max(xx)]
    netApprox = [min(net(linspace(Domain(1), Domain(2)))),max(net(linspace(Domain(1), Domain(2))))]
    figure
    plot(linspace(Domain(1), Domain(2), 100), xx, 'r')
    hold on
    plot(linspace(Domain(1), Domain(2), 100), net(linspace(Domain(1), Domain(2))), 'blue')
    hold on
    plot(linspace(Domain(1), Domain(2), 100), box1(1)*ones(1,100), 'black')
    hold on
    plot(linspace(Domain(1), Domain(2), 100), box1(2)*ones(1,100), 'black')
end
