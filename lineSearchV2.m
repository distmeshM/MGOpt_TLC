function [zw, enNew, ls_t] = lineSearchV2(t, zw, grad, dir, v, ls_t, beta, gamma, restD)
    T = int32(t'); ZW = zw'; DIR = dir';
    [A,B,C] = enPre(ZW,T,DIR,restD);
    % [X1,X2,X3]=enPre(ZW,T);
    % [D1,D2,D3]=enPre(DIR,T);
    dg = dot(grad-v, dir(:)) * gamma; dv = dot(dir(:), v);
    % enlist = en4d(ZW,T,restD);
    % enlistNew = en4d(ZW+ls_t*DIR,T,restD);
    enlist = en4dwithPre(A,B,C,0);
    enlistNew = en4dwithPre(A,B,C,ls_t);
    enlistDiff = enlistNew - enlist;
    enDiff = sum(enlistDiff) - ls_t*dv;
    while enDiff > dg * ls_t && ls_t > 1e-12
        ls_t = ls_t * beta;
        % enlistNew = en4d(ZW+ls_t*DIR,T,restD);
        enlistNew = en4dwithPre(A,B,C,ls_t);
        enlistDiff = enlistNew - enlist;
        enDiff = sum(enlistDiff) - ls_t*dv;
    end
    zw = zw + ls_t * dir;
    enNew = sum(enlist) + enDiff;
    % fprintf("lineSearch: ls_t = %d, step = %d\n", ls_t, ls_t * norm(dir));
end