function [zw, enNew, ls_t] = lineSearchV(t, zw, grad, dir, v, ls_t, beta, gamma, restD)
    T = int32(t'); ZW = zw'; DIR = dir';
    [A,B,C] = enPre(ZW,T,DIR,restD);
    dg = dot(grad-v, dir(:)) * gamma; dv = dot(dir(:), v);
    if dg > 0
        ls_t = 0; enNew = -1;
        fprintf("Increase direction.\n")
        return;
    end
    % enlist = en_grad_hessian2((zw)',int32(t'),restD');
    % enlistNew = en_grad_hessian2((zw+ls_t*dir)',int32(t'),restD');
    enlist = en4dwithPre(A,B,C,0);
    enlistNew = en4dwithPre(A,B,C,ls_t);
    enlistDiff = enlistNew - enlist;
    enDiff = sum(enlistDiff) - ls_t*dv;
    while enDiff > dg * ls_t && ls_t > 1e-12
        ls_t = ls_t * beta;
        % enlistNew = en_grad_hessian2((zw+ls_t*dir)',int32(t'),restD');
        enlistNew = en4dwithPre(A,B,C,ls_t);
        enlistDiff = enlistNew - enlist;
        enDiff = sum(enlistDiff) - ls_t*dv;
    end
    zw = zw + ls_t * dir;
    enNew = sum(enlist) + enDiff;
    % fprintf("lineSearch: ls_t = %f, step = %f\n", ls_t, ls_t * norm(dir));
end