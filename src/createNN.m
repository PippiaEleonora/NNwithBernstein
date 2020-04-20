function net = createNN(inputs,targets,n_neurons)
%CREATENN Generate a neural network form the dataset [inputs,targets] with
%length(n_neurons) hidden layers. The i-th layer has n_neurons(i) neurons. 
%
% This function returns the neural network 'net' 
%
%%
hiddenLayerSize = n_neurons;
hiddenLayerNumber = 1;

net = feedforwardnet(repmat(hiddenLayerSize,1,hiddenLayerNumber),'trainlm');
for i=1:hiddenLayerNumber
    net.layers{i}.transferFcn = 'tansig';
end
net.divideFcn = 'divideind';

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainInd = 1:2:size(inputs,2);
net.divideParam.valInd = 2:4:size(inputs,2);
net.divideParam.testInd = 4:4:size(inputs,2);

%% Train the Network
[net] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
performance = perform(net,targets,outputs);

errors = gsubtract(outputs,targets);
disp(performance);
ploterrhist(errors,'bins',100)
end

