function Z = Foxholes(X,Y)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
a = [-32 -16 0 16 32];
A = [a a a a a; ...
    -32*ones(1,5) -16*ones(1,5) zeros(1,5) 16*ones(1,5) 32*ones(1,5)];

val = 0.*X;
for i=1:25
    curr = i + (X-A(1,i)).^6 + (Y-A(2,i)).^6;
    val = val + 1./curr;
end
val = val + 1/500;
Z = 1./val;
end

