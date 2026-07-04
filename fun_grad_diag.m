function [grad, hs] = fun_grad_diag(t, zw, g2GIdx, restD, level)
zw2 = zw'; 
t2 = t'; 
if nargout == 2
    [e,g,h]=en_grad_diag2(zw2,t2,restD);
    % grad = accumarray(g2GIdx(:), g(:));
    % hs = accumarray(g2GIdx(:), h(:));
    % grad = addOrder * g(:);
    % hs = addOrder * h(:);
    grad = SpMV(level, g(:));
    hs = SpMV(level, h(:));
elseif nargout == 1
    [e,g]=en_grad_diag2(zw2,t2,restD);
    % grad = accumarray(g2GIdx(:), g(:));
    % grad = addOrder * g(:);
    grad = SpMV(level, g(:));
end
end