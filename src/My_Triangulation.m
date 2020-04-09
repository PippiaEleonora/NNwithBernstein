function [PT,n_triangle,polytope] = My_Triangulation(polytope)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%% Triangulation (give an error if the triangulation is not unique)
curr = 0;
try PT = delaunayn(polytope);
catch ME
    while (strcmp(ME.identifier,'MATLAB:catenate:dimensionMismatch'))
        curr = curr+1;
        c = mean(polytope(:,curr:end),1);
        polytope = [polytope; c];
        PT = delaunayn(polytope);
        try 
            PT = delaunayn(polytope);
            ME.identifier = '';
        catch ME
        end
    end
end 
n_triangle = size(PT,1);
end

