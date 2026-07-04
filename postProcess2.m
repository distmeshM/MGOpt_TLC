function [w, nflip] = postProcess2(w, t0, flip, vertTri, A, maxit, x)
B = findBoundary(w, t0);
isB = false(size(w, 1), 1);
isB(B) = true;
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
Adj = A(tris, tris);
G = graph(Adj);
[components, componentSizes] = conncomp(G, 'OutputForm', 'cell');
nComponent = length(components);
active = false(nComponent, 1);
averageAll = sum(circle_2d(w,t0))/size(t0,1);
activeTriIds = [];
for component = 1:nComponent
    averageComponent = sum(circle_2d(w,t0(tris(components{component}),:)))/componentSizes(component);
    if averageComponent < averageAll * 0.05
        active(component) = true;
        activeTriIds = [activeTriIds, components{component}];
    end
end
activeTriIds = sort(unique(activeTriIds));
tris = tris(activeTriIds);
for i = 1:0
    xflip = unique(t0(tris, :));
    if sum(isB(xflip) > 0)
        break;
    end
    if iscell(vertTri)
        cells = vertTri(xflip,:);
        tris = cellUnion(cells);
    elseif issparse(vertTri)
        [tris,~,~] = find(vertTri(:,xflip));
        tris = unique(tris);
    end
end
Adj = A(tris, tris);
G = graph(Adj);
[components, componentSizes] = conncomp(G, 'OutputForm', 'cell');
nComponent = length(components);
add = [];
for component = 1:nComponent
    componentSizelast = componentSizes(component);
    trisloc = tris(components{component});
    [Bloc,Hloc] = findBoundary(w, t0(trisloc, :));
    while ~isempty(Hloc)
        HB = cellUnion(Hloc);
        if iscell(vertTri)
            cells = vertTri(HB,:);
            trisAdd = cellUnion(cells);
        else
            [trisAdd, ~, ~] = find(vertTri(:,HB));
            trisAdd = unique(trisAdd);
        end
        trisloc = union(trisloc, trisAdd);
        add = union(add, trisAdd);
        [Bloc,Hloc] = findBoundary(w, t0(trisloc, :));
        componentSizeNew = length(trisloc);
        if componentSizeNew == componentSizelast
            error("Compute connected components.");
        end
        componentSizelast = componentSizeNew;
    end
end

tris = union(tris, add);

t1 = t0(tris,:);
ids = unique(t1);
[x2d, t] = deleteIso(w, t1);
x = x(ids,:);
% write_obj_with_texture('bad.obj',x,x2d,t);
t = int32(t);
nv = size(x2d, 1); nf = size(t, 1);




if size(x2d, 1) > 0
[B, H] = findBoundary(x2d, t);
B = union(B,cellUnion(H));


zw = tutte(x2d,t,B,'uniform');
% zw = x2d * 1;
% zw(B,:) = x2d(B,:);



w(ids, :) = zw;
end
nflip = sum(area_2d(w, t0) <= 0);
flip = area_2d(w, t0) <= 0;
if nflip > 0 && nflip < 10
    [w, ~, nflip] = postProcess(w, t0, flip, vertTri, A, 200, x);
end
end