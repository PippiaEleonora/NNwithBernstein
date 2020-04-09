function [polytope] = NN_polyApproximation(poly,Iconfident,W,bias,n_layer,n_neurons,x,Domain)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
polytope_curr = Domain;
for l=1:n_layer+1
    N = size(polytope_curr.A,2);
    poly_new.A = [];
    poly_new.b = [];
    poly_new.Aeq = [];
    poly_new.beq = [];
    for  t=1:n_neurons(l)
        if l<n_layer+1
            Pplus = intersectionHull('lcon',polytope_curr.A,polytope_curr.b,...
                'lcon', -W{l,1}(t,:),bias{l,1}(t)-Iconfident(2));
            Pminus = intersectionHull('lcon',polytope_curr.A,polytope_curr.b,...
                'lcon', -W{l,1}(t,:),bias{l,1}(t)-Iconfident(2));
            P = intersectionHull('lcon',polytope_curr.A,polytope_curr.b,...
                'lcon',[W{l,1}(t,:); -W{l,1}(t,:)],[-bias{l,1}(t)+Iconfident(2); bias{l,1}(t)-Iconfident(1)]);

            if  ~isempty(P.vert) %P.lcon = {A,b,Aeq,beq}
                [P_new,Project] = degeneration(P.lcon{:});
                [PT,n_triangle,subpolytope_new] = My_Triangulation(P_new); 
                if size(P_new,1)==1
                    val = P_new;
                    Pcell = num2cell(P_new);
                    bcoeff = poly{l,t}(W{l,1}(t,:)*Pcell{:}+bias{l,1}(t));
                    poly_temp = [val,bcoeff];
                else
                    v_triangle = [];
                    for i=1:n_triangle
                        vertex = subpolytope_new(PT(i,:),:)';
                        V = vertex(:,2:end)-vertex(:,1);
                        b = vertex(:,1);
    %                     p = zeros(1,length(b));
                        for idx=1:length(b)
                            p(idx) = x(1:size(vertex,1))*(V(idx,:)') + b(idx);
                        end
                        if size(P_new,2)<N
    %                         proj_cell = zeros(1,m1);
                            for idx=1:m1
                                proj_cell(idx) = Project.b + (x(1:size(vertex,1))*(V')+b')*Project.A(idx,:)';
                            end
                        else
                            proj_cell = [];
                        end
                        [C,mon] = coeffs(poly(W{l,1}(t,:)*transpose([proj_cell,p])+bias{l,1}(t)),x(1:size(vertex,1)));
                        C = double(C);
                        char_mon = arrayfun(@char, mon, 'uniform', 0);
                        idP = zeros(N,length(char_mon));
                        deg = zeros(1,N);
                        for idx = 1:N
                            id_cellP1 = strfind(char_mon,strcat('x',num2str(idx)));
                            id_cellP2 = strfind(char_mon,strcat('x',num2str(idx),'^'));
                            for idx2=1:length(char_mon)
                                if ~isempty(id_cellP1{1,idx2})
                                    if ~isempty(id_cellP2{1,idx2})
                                        final = strfind(char_mon{1,idx2}(id_cellP2{1,idx2}+3:end),strcat('*'));
                                        if isempty(final)
                                            idP(idx,idx2) = str2double(char_mon{1,idx2}(id_cellP2{1,idx2}+3:end));
                                        else
                                            idP(idx,idx2) = str2double(char_mon{1,idx2}(id_cellP2{1,idx2}+3:id_cellP2{1,idx2}+3+final(1)-2));
                                        end
                                    else
                                        idP(idx,idx2) = 1;
                                    end
                                end
                            end
                            deg(idx) = max(idP(idx,:));
                        end

                        

                        if length(deg)<2
                            for idx=1:length(idP)
                                coeff(idP(1,idx)+1) = C(idx);
                            end
                            [bcoeff,v] = BernsteinCoeff_1D(coeff,deg,[0 1],'Garloff');
                            v = v';
                        else
%                             coeff = sparse(deg(1)+1,prod(deg(2:end)+1));
                            coeff = sparse(idP(1,:)+1,id2column(idP(2:end,:),deg(2:end),'first'),C);
                            [bcoeff,v_cell] = GeneralizedBernsteinCoeff_nD(coeff,deg,[]);
                            bcoeff = bcoeff';
                            v = [];
                            for idx=1:N
                                v =[v v_cell{idx}'];  
                            end
                        end
                        v_triangle = [v_triangle;v bcoeff];
                    end
                    poly_temp = v_triangle;
                end
            end
            if ~isempty(Pplus.vert)
                poly_temp = [poly_temp; Pplus.vert ones(size(Pplus.vert,1),1)];
            end
            if ~isempty(Pminus.vert)
                poly_temp = [poly_temp; Pminus.vert -ones(size(Pminus.vert,1),1)];
            end
            [A, b, Aeq, beq]  = vert2lcon(poly_temp);

            if ~isempty(A)
                n1 = size(A,1);
                A = [A(:,1:N) zeros(n1,t-1) A(:,N+1) zeros(n1,n_neurons(l)-t)];
            end
            if ~isempty(Aeq)
                n1 = size(Aeq,1);
                Aeq = [Aeq(:,1:N) zeros(n1,t-1) Aeq(:,N+1) zeros(n1,n_neurons(l)-t)];
            end
            poly_new.A = [poly_new.A;A];
            poly_new.b = [poly_new.b;b];
            poly_new.Aeq = [poly_new.Aeq;Aeq];
            poly_new.beq = [poly_new.beq;beq];
        else
            [P_new,Project] = degeneration(polytope_curr.A,polytope_curr.b,polytope_curr.Aeq,polytope_curr.beq);
            [PT,n_triangle,subpolytope_new] = My_Triangulation(P_new); 
            if size(P_new,1)==1
                val = P_new;
                Pcell = num2cell(P_new);
                bcoeff = W{l,1}(t,:)*Pcell{:}+bias{l,1}(t);
                poly_temp = [val,bcoeff];
            else
                v_triangle = [];
                for i=1:n_triangle
                    vertex = subpolytope_new(PT(i,:),:)';
                    V = vertex(:,2:end)-vertex(:,1);
                    b = vertex(:,1);
                    m = size(vertex,2);

                    W_new = W{l,1}*V;
                    bias_new = W{l,1}*b + bias{l,1};

                    bcoeff = zeros(n_neurons(l), m);
                    bcoeff(:,1) = bias_new;
                    for idx = 2:m
                        bcoeff(:,idx) = W_new*[zeros(idx-2,1); 1; zeros(m-idx,1)] + bias_new;
                    end
                    bcoeff = bcoeff';
                    if ~isempty(Project.A)
                        project_vertex = Project.A*vertex'+Project.b;
                        v = [project_vertex vertex'];
                    else
                        v = vertex';
                    end
                    v_triangle = [v_triangle;v bcoeff]; 
                end
                poly_temp = v_triangle;
            end
            [A, b, Aeq, beq]  = vert2lcon(poly_temp);

            if ~isempty(A)
                n1 = size(A,1);
                A = [A(:,1:N) zeros(n1,t-1) A(:,N+1) zeros(n1,n_neurons(l)-t)];
            end
            if ~isempty(Aeq)
                n1 = size(Aeq,1);
                Aeq = [Aeq(:,1:N) zeros(n1,t-1) Aeq(:,N+1) zeros(n1,n_neurons(l)-t)];
            end
            poly_new.A = [poly_new.A;A];
            poly_new.b = [poly_new.b;b];
            poly_new.Aeq = [poly_new.Aeq;Aeq];
            poly_new.beq = [poly_new.beq;beq];
        end
    end
    V_new = lcon2vert(poly_new.A,poly_new.b,poly_new.Aeq,poly_new.beq);
    V_new = V_new(:,N+1:end);
    if size(V_new,2)<2
        polytope_curr.A = [1 -1];
        polytope_curr.b = [max(V_new), min(V_new)];
        polytope_curr.Aeq = [];
        polytope_curr.beq = [];
    else
        [polytope_curr.A,polytope_curr.b,polytope_curr.Aeq,polytope_curr.beq] = vert2lcon(V_new);
    end
end
polytope = polytope_curr;
end