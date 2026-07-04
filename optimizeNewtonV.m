function zw = optimizeNewtonV(t, zw, g2GIdx, B, maxit, nzidx, Hnonzeros0, H, beta, gamma, restD, v)
% figure;
% h = drawmesh(t,zw);
for it = 1:maxit
[grad, hessian] = fun_grad_hessian([], t, zw, g2GIdx, nzidx, Hnonzeros0, H, [], restD);
% [grad, diag] = fun_grad_diag(t, zw, g2GIdx, restD);
nv = size(zw, 1);
internal = true(nv, 1); internal(B) = false;
active = [internal; internal];
dir = zeros(nv*2, 1);
dir(active) = -hessian(active, active) \ (grad(active) - v(active));
dir = reshape(dir, size(dir, 1) / 2, 2);
dir(B, :) = 0;
[zw, e, ls_t] = lineSearchV(t, zw, grad, dir, v(:), 1, beta, gamma, restD);

% set(h, 'Vertices', zw);drawnow;
% fprintf("Optimize nVertive: %d, it: %d, en: %f, lambda: %f, step: %f, flipTri: %d\n", size(zw, 1), it, e, ls_t, ls_t * norm(dir), sum(area_2d(zw, t) <= 0))
    
end
end