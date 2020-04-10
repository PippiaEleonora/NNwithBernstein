function id = column2id(column,deg,type)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
n = length(deg);
id = zeros(1,n);
current = column-1;
switch type
    case 'first'
        for i=n:-1:2
            if current/prod(deg(1:i-1)+1)>=1
                id(i) = floor(current/prod(deg(1:i-1)+1));
                current = current-prod(deg(1:i-1)+1)*id(i);
            end
        end
        id(1) = current;
    case 'last'
        for i=1:n-1
            if current/prod(deg(i+1:end)+1)>=1
                id(i) = floor(current/prod(deg(i+1:end)+1));
                current = current-prod(deg(1:end-i)+1)*id(i);
            end
        end
        id(n) = current;
end
end