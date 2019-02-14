function [d21,j11]= smallestDistance(LaKu,LoKu,La0,Lo0,sizeKu,npolosy)
dist21=zeros();
for j1=1:sizeKu(2)
    dist21(j1) = deg2km(distance(LaKu(npolosy,j1),LoKu(npolosy,j1),La0,Lo0));
end
[d21,j11]=min(dist21);