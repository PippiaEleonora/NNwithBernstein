function val = id2column(idx,deg,type)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
switch type
    case 'first'
        val = idx(:,1)+1;
        for i=2:size(idx,2)
            val = val+prod((deg(1:i-1)+1)).*idx(:,i);
        end
    case 'last'
        val = idx(:,end) + 1;
        for i=size(idx,2)-1:-1:1
            val = val+prod((deg(i+1:end)+1)).*idx(:,i);
        end
end
end