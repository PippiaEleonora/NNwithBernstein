function [polytope] = NN_polyApproximation(poly,Iconfident,W,bias,n_layer,n_neurons,x,Domain)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
polytope_curr = Domain;
coeffP = fliplr(double(coeffs(poly(x(1)),x(1),'All')));
for l=1:n_layer+1
    if l==2
        v_domain = lcon2vert(Domain.A,Domain.b,Domain.Aeq,Domain.beq);
        xx = linspace(min(v_domain),max(v_domain(2)),100);
        figure
        y1 = tansig(W{l-1,1}(1,:)*xx'+bias{l-1,1}(1));
        y2 = tansig(W{l-1,1}(2,:)*xx'+bias{l-1,1}(2));
        y3 = tansig(W{l-1,1}(3,:)*xx'+bias{l-1,1}(3));
        hold on
        ver_new = lcon2vert(polytope_curr.A,polytope_curr.b,polytope_curr.Aeq,polytope_curr.beq);
        conv = convhull(ver_new);
        scatter3(y1,y2,y3)
        hold on
        trisurf(conv,ver_new(:,1),ver_new(:,2),ver_new(:,3), 'FaceAlpha',0.5);
        poly_old = polytope_curr;
    elseif l==3
        v_domain = lcon2vert(poly_old.A,poly_old.b,poly_old.Aeq,poly_old.beq);
        figure
        y1_old = y1;
        y1 = tansig(W{l-1,1}(1,:)*[y1,y2,y3]'+bias{l-1,1}(1));
        y2 = tansig(W{l-1,1}(2,:)*[y1_old,y2,y3]'+bias{l-1,1}(2));
        ver_new = lcon2vert(polytope_curr.A,polytope_curr.b,polytope_curr.Aeq,polytope_curr.beq);
        conv = convhull(ver_new);
        figure
        plot(y1,y2)
        hold on
        plot(ver_new(conv,1),ver_new(conv,2))
    end
    N = size(polytope_curr.A,2);
    poly_new.A = [];
    poly_new.b = [];
    poly_new.Aeq = [];
    poly_new.beq = [];
    for  t=1:n_neurons(l)
        if l<n_layer+1
            if N>1
                Pplus = intersectionHull('lcon',polytope_curr.A,polytope_curr.b,...
                    'lcon', -W{l,1}(t,:),bias{l,1}(t)-Iconfident(2));
                Pminus = intersectionHull('lcon',polytope_curr.A,polytope_curr.b,...
                    'lcon', W{l,1}(t,:),-bias{l,1}(t)+Iconfident(1));
                P = intersectionHull('lcon',polytope_curr.A,polytope_curr.b,...
                    'lcon',[W{l,1}(t,:); -W{l,1}(t,:)],[-bias{l,1}(t)+Iconfident(2); bias{l,1}(t)-Iconfident(1)]);
                poly_temp = [];
                if  ~isempty(P.vert) %P.lcon = {A,b,Aeq,beq}
                    [P_new,Project] = degeneration(P.lcon{:});
                    [PT,n_triangle,subpolytope_new] = My_Triangulation(P_new); 
                    if size(P_new,1)==1
                        val = P_new;
                        Pcell = num2cell(P_new);
                        bcoeff = poly{l,t}(W{l,1}(t,:)*Pcell{:}+bias{l,1}(t));
                        bcoeff = min(max(bcoeff,-1),1);
                        poly_temp = [val,bcoeff];
                    else
                        v_triangle = [];
                        for i=1:n_triangle

                            vertex = subpolytope_new(PT(i,:),:)';
                            V = vertex(:,2:end)-vertex(:,1);
                            b = vertex(:,1);
                            for idx=1:length(b)
                                p(idx) = x(1:size(vertex,1))*(V(idx,:)') + b(idx);
                            end
                            if size(P_new,2)<N
                                for idx=1:m1
                                    proj_cell(idx) = Project.b + (x(1:size(vertex,1))*(V')+b')*Project.A(idx,:)';
                                end
                            else
                                proj_cell = [];
                            end

                            if N>1
                                if size(P_new,2)<N
                                    boh = 1
                                else  
                                    coeff = multiCoeff(W{l,1}(t,:)*V,W{l,1}(t,:)*b+bias{l,1}(t),coeffP);
                                end
                            else
                                coeff = fliplr(double(coeffs(poly(W{l,1}(t,:)*transpose([proj_cell,p])+bias{l,1}(t)),x(1:size(vertex,1)),'All')));
                            end
                            deg =(length(coeffP)-1)*ones(1,size(P_new,2));

                            if length(deg)<2
                                [bcoeff,v] = BernsteinCoeff_1D(coeff,deg,[0 1],'Garloff');
                                bcoeff = min(max(bcoeff,-1),1);
                                v = (V*v+b)';
                            else
                                [bcoeff,v_cell] = GeneralizedBernsteinCoeff_nD(coeff,deg,[]);
                                bcoeff = min(max(bcoeff,-1),1);
                                bcoeff = bcoeff';
                                v = [];
                                for idx=1:N
                                    v =[v v_cell{idx}'];  
                                end
                                v = (V*v'+b)';
                            end
                            v_triangle = [v_triangle;v bcoeff];
                        end
                        poly_temp = v_triangle;
                        if l==1
                            v_domain = P.vert;
                            xx = linspace(min(v_domain),max(v_domain(2)),100);
                            figure
                            y1 = tansig(W{l,1}(t,:)*xx'+bias{l,1}(t));
                            conv = convhull(poly_temp);
                            plot(xx,y1)
                            hold on
                            plot(poly_temp(conv,1),poly_temp(conv,2));
                        end
                    end
                end
                if ~isempty(Pplus.vert)
                    poly_temp = [poly_temp; Pplus.vert ones(size(Pplus.vert,1),1)];
                end
                if ~isempty(Pminus.vert)
                    poly_temp = [poly_temp; Pminus.vert -ones(size(Pminus.vert,1),1)];
                end
            else
                vertex = lcon2vert(polytope_curr.A,polytope_curr.b,polytope_curr.Aeq,polytope_curr.beq);
                interval = W{l,1}(t,:).*vertex + bias{l,1}(t);
                interval = [min(interval), max(interval)];
                
                Pminus = [interval(1), min(interval(2),Iconfident(1))];
                Pplus = [max(interval(1),Iconfident(2)), interval(2)];
                P = [max(interval(1),Iconfident(1)), min(interval(2),Iconfident(2))];
                poly_temp = [];
                if Pminus(2)>Pminus(1)
                    conversion = (Pminus-bias{l,1}(t))./W{l,1}(t,:);
                    poly_temp = [poly_temp; conversion' [-1; -1]];
                end
                if Pplus(2)>Pplus(1)
                    conversion = (Pplus-bias{l,1}(t))./W{l,1}(t,:);
                    poly_temp = [poly_temp; conversion' [1; 1]];
                end
                if P(2)>P(1)
                    [bcoeff,v] = BernsteinCoeff_1D(coeffP,length(coeffP)-1,P,'Garloff');
                    bcoeff = min(max(bcoeff,-1),1);
                    v = (v-bias{l,1}(t))./W{l,1}(t,:);
                    poly_temp = [poly_temp; v' bcoeff];
                end
            end
            if l==1
                v_domain = lcon2vert(Domain.A,Domain.b,Domain.Aeq,Domain.beq);
                xx = linspace(min(v_domain),max(v_domain(2)),100);
                figure
                y1 = tansig(W{l,1}(t,:)*xx'+bias{l,1}(t));
                conv = convhull(poly_temp);
                plot(xx,y1)
                hold on
                plot(poly_temp(conv,1),poly_temp(conv,2));
            end
            [A, b, Aeq, beq]  = vert2lcon(poly_temp);
            poly_temp_new = lcon2vert(A,b,Aeq,beq);
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
            min(lcon2vert(poly_new.A,poly_new.b,[poly_new.Aeq; zeros(n_neurons(l)-t,N+t) eye(n_neurons(l)-t)],[poly_new.beq; zeros(n_neurons(l)-t,1)]))
                        
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