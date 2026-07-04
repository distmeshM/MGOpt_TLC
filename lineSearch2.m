function [zw, enNew, ls_t] = lineSearch2(t, zw, grad, dir, ls_t, beta, gamma, restD)
    dg = dot(grad, dir(:)) * gamma;
    % enlist = enlist_4d(xy, t, zw);
    enlist = en_grad_hessian2((zw)',int32(t'),restD');
    % enlistNew = enlist_4d(xy,t,zw+ls_t*dir);
    enlistNew = en_grad_hessian2((zw+ls_t*dir)',int32(t'),restD');
    enlistDiff = enlistNew - enlist;
    enDiff = sum(enlistDiff);
    while enDiff > dg * ls_t
        ls_t = ls_t * beta;
        % enlistNew = enlist_4d(xy,t,zw+ls_t*dir);
        enlistNew = en_grad_hessian2((zw+ls_t*dir)',int32(t'),restD');
        enlistDiff = enlistNew - enlist;
        enDiff = sum(enlistDiff);
    end
    zw = zw + ls_t * dir;
    enNew = sum(enlist) + enDiff;
end