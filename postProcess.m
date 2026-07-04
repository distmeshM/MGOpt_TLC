function [w, maxIter, nflip] = postProcess(w, t0, flip, vertTri, A, maxit, x)
tris = flip;
if iscell(vertTri)
    for i = 1:3
        xflip = unique(t0(tris, :));
        cells = vertTri(xflip,:);
        tris = cellUnion(cells);
    end
elseif issparse(vertTri)
    for i = 1:3
        xflip = unique(t0(tris, :));
        [tris,~,~] = find(vertTri(:,xflip));
        tris = unique(tris);
    end
else
    error("Invalid input.");
end
t1 = t0(tris,:);
ids = unique(t1);
[x2d, t] = deleteIso(w, t1);
% write_obj_with_texture('bad.obj',x,x2d,t);
t = int32(t);
nv = size(x2d, 1); nf = size(t, 1);

Adj = A(tris, tris);
G = graph(Adj);

[components, componentSizes] = conncomp(G, 'OutputForm', 'cell');
nComponent = length(components);

alphas = zeros(nComponent, 1);
for component = 1:nComponent
    A1 = sum(area_2d(x2d, t(components{component},:)));
    A2 = sum(area_2d(x2d, t(components{component},:))*0+sqrt(3)/4);
    alphas(component) = (1e-2 * (A1/A2));
end



% ratio = A1 / A2;
% alpha = (1e-6 * (A1/A2));
beta = 0.7;
gamma = 0.5;

cenZWs = zeros(nComponent, 2);
nflip = 0;
zw = x2d;
maxIter = 0;
for component = 1:nComponent
    tloc = t(components{component},:);
    xId = unique(tloc);
    [xloc, tloc] = deleteIso(x2d, tloc);
    nv = size(xloc,1);
    nf = size(tloc,1);
    [B, H] = findBoundary(xloc, tloc);
    B = union(B,cellUnion(H));
    lambda = 0e-12;
    L = laplacian(xloc, tloc, 'uniform');
    %%
    xloc = tutte(xloc,tloc,B);
    H = [L L; L L];

    [xmesh, ymesh] = meshgrid(1:6, 1:6);
    t2 = [tloc tloc+nv]';
    g2GIdx = uint64(t2);
    Mi = t2(xmesh(:), :);
    Mj = t2(ymesh(:), :);

    Hnonzeros0 = zeros(nnz(H),1);
    idxDiagH = ij2nzIdxs(H, uint64(1:nv*2), uint64(1:nv*2));
    Hnonzeros0(idxDiagH) = lambda*2;
    nzidx = ij2nzIdxs(H, uint64(Mi), uint64(Mj));

    zloc = xloc * 1;
    cenZloc = sum(zloc) / size(zloc, 1);
    cenZWs(component,:) = cenZloc;
    zloc(B,:) = xloc(B,:);
    zloc = zloc - cenZloc;
    alpha = alphas(component);
    restD = ones(nf, 3) .* alpha;
    restD = restD';
    
    % vid = unique(t(components{component},:));
    iter2=0;iter3=0;
    [zloc, iter1, flip] = optimizeNewton([], int32(tloc), zloc, g2GIdx, B, maxit/3, nzidx, Hnonzeros0, H, alpha, beta, gamma, restD);
    if flip>0
        [zloc, iter2, flip] = optimizeNewton([], int32(tloc), zloc, g2GIdx, B, maxit/3, nzidx, Hnonzeros0, H, alpha, beta, gamma, restD*1e-2);
        if flip>0
            [zloc, iter3, flip] = optimizeNewton([], int32(tloc), zloc, g2GIdx, B, maxit/3, nzidx, Hnonzeros0, H, alpha, beta, gamma, restD*1e-4); 
        end
    end
    iter = iter1+iter2+iter3;
    maxIter = max(maxIter, iter);
    nflip = nflip + flip;
    zw(xId,:) = zloc + cenZloc;
end
if(isnan(sum(sum(zw))))
    nflip = Inf;
end
w(ids, :) = zw;
end