load HestonIVSgrid_NI91529.mat

% This is tailored to the larger dataset.


tic 
a = size(HestonIVS2D)
A = reshape(HestonIVS2D, [a(1), a(2)*a(3)]);

A = A(1:30000,:);

size(A)
maximum = max(pdist(A))
pairwiseD = pdist(A); % this consumes too much memories
size(pairwiseD)

% maxd = 0;
% for i = 1:(a(1)-1)
%     for j = (i+1):a(1)
%         vec1 = A(i,:);
%         vec2 = A(j,:);
%             
%         d = norm(vec1 - vec2);
%         if d > maxd
%             maxd = d;
%         end 
%     end
% %     if rem(i,1000) == 0
% %         disp(i)
% %     end
%     disp(i)
% end
% 
% maxd

tend = toc 