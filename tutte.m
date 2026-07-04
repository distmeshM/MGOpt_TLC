function z = tutte(z, t, B, type)
    if nargin <= 3
        type = 'uniform';
    end
    nv = size(z, 1);
    L = laplacian(z, t, type);
    I = setdiff(1:nv, B);
    z(I, :) = -L(I,I)\(L(I,B)*z(B, :));
end