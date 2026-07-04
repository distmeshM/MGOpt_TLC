function zw = optimize(xy, t, restD, zw, g2GIdx, B, maxit, v, level)
if nargin <= 7
    v = zw * 0;
end
for it = 1:maxit
[grad, hessian] = fun_grad_diag(t,zw,g2GIdx,restD,level);
dir = -(abs(hessian)).\(grad - v(:));
dir = reshape(dir, length(dir)/2, 2);
dir(B, :) = 0;

[zw, en, ls_t] = lineSearchV2(t, zw, grad, dir, v(:), 1, 0.7, 0.5, restD);
end
end