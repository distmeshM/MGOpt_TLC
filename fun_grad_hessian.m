function [grad, hessian] = fun_grad_hessian(xy, t, zw, g2GIdx, nzidx, Hnonzeros0, H, alpha, restD)
% tic
[e,g,h]=en_grad_hessian2(zw',t',restD);
% toc
% tic
% [e1,g1,h1]=cpu_en_grad_hessian2(zw',t',restD);
% toc

% ZW = gpuArray(zw);
% T = gpuArray(t);
% RESTd = gpuArray(restD);
% tic
% [e2,g2,h2]=cuda_en_grad_hessian2(ZW',T',RESTd);
% toc
grad = accumarray(g2GIdx(:), g(:));
% grad = accumarray2(g2GIdx(:), g(:), int32(size(zw, 1)*2));
hs = accumarray(nzidx(:), h(:)) + Hnonzeros0;
hessian = replaceNonzeros(H, hs);
end