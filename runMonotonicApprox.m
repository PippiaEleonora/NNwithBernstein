clear all
clc
% Add folder and subfolders to the path.
addpath(genpath(strcat(pwd,'/src')));
%%
fileName = 'mnist_tanh_3_5.txt';
[W,bias,n_neurons,input] = readTF(fileName);
n_layer = length(W);
Domain = [-ones(input,1), ones(input,1)];
[box,B] = NN_nopoly_boxApprox(W,bias,n_layer,n_neurons,Domain,0);
