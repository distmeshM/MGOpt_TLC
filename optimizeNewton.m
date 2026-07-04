function [zw, it, nflip] = optimizeNewton(xy, t, zw, g2GIdx, B, maxit, nzidx, Hnonzeros0, H, alpha, beta, gamma, restD, h, solver)
% en0 = en_mini_surface_4d(xy, t, zw);
% figure; h = drawmesh(t, zw);
for it = 1:maxit
tt = tic;
[grad, hessian] = fun_grad_hessian(xy, t, zw, g2GIdx, nzidx, Hnonzeros0, H, alpha, restD);
% [grad, diag] = fun_grad_diag(t, zw, g2GIdx, restD);
nv = size(zw, 1);
internal = true(nv, 1); internal(B) = false;
active = [internal; internal];
dir = zeros(nv*2, 1);
% hessian2 = extractSparseDiagonals(hessian);
% tt2 = tic;
dir(active) = -hessian(active, active) \ grad(active);
% dir(active) = mysolver(hessian(active,active), -grad(active));

% dir(active) = solver.refactor_solve(hessian(active,active), -grad(active));
% TimeP = toc(tt2);

% tt3 = tic;
% dir(active) = mysolver(hessian(active,active), -grad(active));
% TimeC = toc(tt3);
% TimeC/TimeP

% d = diag(hessian);
% dir(active) = -d(active) .\ grad(active);
% dir(active) = -(abs(diag(active))) .\ grad(active);
dir = reshape(dir, size(dir, 1) / 2, 2);
% dir = -(grad - v);
dir(B, :) = 0;
[zw, e, ls_t] = lineSearch2(t, zw, grad, dir, 1, beta, gamma, restD);

% set(h, 'Vertices', zw);drawnow;
% set(h, 'Vertices', zw);
% drawnow;
% m = -20:20; lambdas = [0.5 .^ m, 0];
% ens = zeros(length(lambdas), 1);
% for i = 1:length(lambdas)
%     ens(i) = en_mini_surface_4d(xy, t, zw + lambdas(i) * dir);
% end
% [e, id] = min(ens);
% zw = zw + lambdas(id) * dir;

fprintf("Optimize nVertive: %d, it: %d, en: %f, lambda: %f, step: %f, flipTri: %d, time: %fs\n", size(zw, 1), it, e, ls_t, ls_t * norm(dir), sum(area_2d(zw, t) <= 0), toc(tt))
    % if sum(area_2d(zw,t)<=0) < 10 && e < en0 * 2
    %     zw = zw * 2;
    % end
    nflip = sum(area_2d(zw,t)<=0);
    if nflip==0
        break;
    end
end
end