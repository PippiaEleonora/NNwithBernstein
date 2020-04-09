function [V_new,Project] = degeneration(A,b,Aeq,beq)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
if isempty(Aeq) ||  isempty(A)
    V_new = lcon2vert(A,b,Aeq,beq);
    Project.A = [];
    Project.b = [];
% elseif isempty(A) %single point
%     V = lcon2vert(P.lcon{:})
else %is degenerate
    Aeq_1 = Aeq(:,1:m1);
    Aeq_2 = Aeq(:,m1+1:end);

    A_1 = A(:,1:m1);
    A_2 = A(:,m1+1:end);

    assert(det(Aeq_1)~=0) % DEAL WITH THIS CASE!

    invAeq_1 = Aeq_1^-1;
    assert(det(Aeq_1)~=0)
    A_new = A_2-(A_1*invAeq_1*Aeq_2);
    b_new =b-(A_1*invAeq_1*beq);

    V_new=lcon2vert(A_new,b_new);
    
    Project.A = -invAeq_1*Aeq_2;
    Project.b = invAeq_1*beq;
%     proj_sym(x) = invAeq_1*(beq - Aeq_2*transpose(x(1,1:N-m1)));
end
end

