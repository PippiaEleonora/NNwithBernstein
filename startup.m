clear all
close all
clear functions
clc

% Determine the folder's path.
curr_folder = pwd;
folder = strcat(curr_folder,'/src');
% Add folder and subfolders to the path.
addpath(genpath(folder));