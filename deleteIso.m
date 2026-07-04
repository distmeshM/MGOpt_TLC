function [x1, t1] = deleteIso(x, t)
    nv = size(x, 1);
    order = zeros(nv, 1);
    ids = unique(t(:));
    nvc = size(ids, 1);
    order(ids) = 1:nvc;
    x1 = x(ids,:);
    t1 = order(t);
end