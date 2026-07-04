clear;clc;
%% Read File
addpath surface_multigrid/mex/build/Release
addpath mex/build/Release
maxlevel = 4;
filePath = 'examples/Letters/bunny_i_G/input.obj';

[x, t, x2d] = readObj(filePath);




B = findBoundary(x, t);
L = laplacian(x, t);
xB = x2d(B, :);
pB = xB([2:end,1], :) - xB;
qB = xB - xB([end, 1:end-1], :);
pBz = pB(:,1) + 1i*pB(:,2);
qBz = qB(:,1) + 1i*qB(:,2);
angles = angle(pBz) - angle(qBz);
angles = mod(angles + pi, 2*pi) - pi;
BId = find(angles<-pi/2);
ptIds = B(BId);
ptIds = union(ptIds, path');
Angle = dot(pB, qB, 2) ./ sqrt(dot(pB,pB,2).*dot(qB,qB,2));
crossPt = find(Angle < 0.9); 
constraintPt = int32(B(crossPt) - 1)';
[val, idx] = min(Angle);
x3d = [x2d, x2d(:,1)*0];
[fileDir, ~, ~] = fileparts(filePath); 

% Load/Construct multigrid
multigridFile = fullfile(fileDir, 'multigrid.mat'); 
if ~exist(multigridFile,'file')
    % First construct multigrid
    [xs,ts,Ps,~] = IntrinsicProlongation2(x, int32(t-1), int32(4), 0, constraintPt, 0.01);
    save(multigridFile, 'xs', 'ts', 'Ps');
    fprintf('多网格变量已计算并保存到 %s\n', multigridFile);
else
    % load
    load(multigridFile, 'xs', 'ts', 'Ps');
    fprintf('已从 %s 加载多网格变量\n', multigridFile);
end


nlevel = length(xs);

tol = 1e-10;


Bs = cell(nlevel, 1);
for level = 1:nlevel
    ts{level} = int32(ts{level} + 1);
    Bs{level} = findBoundary(xs{level}, ts{level});
end

nlevel = length(xs);
Bs = cell(nlevel, 1);
for level = 1:nlevel
    Bs{level} = findBoundary(xs{level}, ts{level});
end
nlevel = min(maxlevel, nlevel);
if size(xs{1},1)/size(xs{nlevel},1) < 20
    for level = nlevel-1:-1:2
        if size(xs{level-1},1)/size(xs{level},1)<1.5
            nlevel = min(nlevel, level-1);
        end
    end
end
% Ps2
Ps2 = cell(nlevel-1, 1);
for i = 1:nlevel - 1
    Ps2{i} = Ps{i}' ./ max(sum(Ps{i}', 2),1e-10);
end
draw = false;
for level = 1:nlevel - 1
    Bf = Bs{level}; Bc = Bs{level+1};
    nvf = size(xs{level}, 1); nvc = size(xs{level+1}, 1);
    isBf = false(nvf, 1); isBc = false(nvc, 1);
    isBf(Bf) = true; isBc(Bc) = true;
    sum(sum(Ps{level}(Bf, ~isBc)))
    [vals,ids] = sort(full(sum(Ps{level}(Bf, ~isBc), 2)),'descend');
    Ps{level}(Bf, ~isBc) = 0;
    Ps2{level}(Bc, ~isBf) = 0;
    if min(sum(Ps2{level}, 2)) < 0.01
        Ids = find(sum(Ps2{level}, 2) < 0.01);
        Adj = getAdjMatrix(ts{level+1});
        for id = 1:length(Ids)
            [~,j] = find(Adj(Ids(id),:));
            Ps2{level}(Ids(id), :) = sum(Ps2{level}(j,:))/sum(sum(Ps2{level}(j,:)));
        end
    end
    if min(sum(Ps2{level}, 2)) < 0.01
        keyboard;
    end
    if min(sum(Ps{level}, 2)) < 0.01
        keyboard;
    end
    Ps2{level} = Ps2{level} ./ max(sum(Ps2{level}, 2),1e-10);
    Ps{level} = Ps{level} ./ max(sum(Ps{level}, 2),1e-10);
end


x = xs{1}; t = ts{1};
nv = size(x, 1);
B = Bs{1};
w = x2d * 1;
w(B,:) = x2d(B,:);
g2GIdx = cell(nlevel, 1); addOrders = cell(nlevel, 1);
Hnonzeros0 = cell(nlevel, 1);
nzidx = cell(nlevel, 1);
Hs = cell(nlevel, 1);
for level = 1:nlevel
    x = xs{level}; t = ts{level}; nv = size(x, 1);

    lambda = 0e-12;
    L = laplacian(x, t, 'uniform');

    [xmesh, ymesh] = meshgrid(1:6, 1:6);
    t2 = [t t+nv]';
    g2GIdx{level} = uint64(t2);
    addOrder = sparse(int32(g2GIdx{level}(:)),int32(1:numel(g2GIdx{level}))',double(g2GIdx{level}(:))*0+1);
    addOrders{level} = addOrder;
    Mi = t2(xmesh(:), :);
    Mj = t2(ymesh(:), :);

    if(level == nlevel)
    H = [L L; L L];
    Hs{level} = H;
    Hnonzeros0{level} = zeros(nnz(H),1);
    idxDiagH = ij2nzIdxs(H, uint64(1:nv*2), uint64(1:nv*2));
    Hnonzeros0{level}(idxDiagH) = lambda*2;
    nzidx{level} = ij2nzIdxs(H, uint64(Mi), uint64(Mj));
    end

end
SpMV(addOrders);
x = xs{1}; t = ts{1};
nv = size(x, 1);
zs1 = cell(nlevel, 1); zs2 = cell(nlevel, 1); vs = cell(nlevel, 1); es = cell(nlevel, 1);
zs1{1} = w; zs2{1} = w;
g(B,:) = 0;
fprintf("it: %d, flipTri: %d\n", 0, sum(area_2d(w,t)<=0));
fliptri0 = sum(area_2d(w,t)<=0);
minFlip = fliptri0; minW = w;
[A, vertTri] = getTriTri(t, x);
clear triTri
ws{1} = w; nf = size(ts{1}, 1);
A1 = sum(area_2d(x2d, t));
A2 = sum(area_2d(x,t)*0+sqrt(3)/4);
alpha = 1e-2 * (A1/A2);
ratios = cell(nlevel, 1);
restD = ones(nf, 3) * alpha;
restDs = cell(nlevel, 1);
for level = 1:nlevel
    ratios{level} = alpha * size(ts{1}, 1) / size(ts{level}, 1);
    restDs{level} = ratios{level} * ones(size(ts{level}, 1), 3);
    restDs{level} = restDs{level}';
end
maxit = 5000;
ens = zeros(maxit, 1); flipTri =  zeros(maxit, 1); grads = zeros(maxit, 1);
minAreaMG = zeros(maxit, 1);totalAreaMG =  zeros(maxit, 1); totalFlipMG = zeros(maxit, 1);
timeMG = zeros(maxit, 1);
gfs = cell(nlevel, 1); gcs = cell(nlevel, 1);
v = zeros(nv, 2);
vs{1} = v;
t = ts{1}; x = xs{1}; B = Bs{1};
post = true;


for level = 1:nlevel
    vs{level} = zs1{level} * 0;
end


weight = 1;


%% main iteration
total = tic;
first10 = true; first3 = true; first1 = true;
for iter = 1:maxit
    
    for level = 1:nlevel - 1
        zs2{level} = optimize(xs{level}, ts{level}, restDs{level}, zs1{level}, g2GIdx{level}, Bs{level}, 1, vs{level}*weight, level);
        gf = fun_grad_diag(ts{level}, zs2{level}, g2GIdx{level}, restDs{level}, level);
        zs1{level + 1} = Ps2{level} * zs2{level};
        gc = fun_grad_diag(ts{level+1}, zs1{level+1}, g2GIdx{level+1}, restDs{level+1}, level+1);
        gf = reshape(gf, length(gf)/2, 2);
        gc = reshape(gc, length(gc)/2, 2);
        vs{level + 1} = gc + Ps{level}' * (vs{level} - gf);
    end
    zs2{nlevel} = optimizeNewtonV(ts{nlevel}, zs1{nlevel}, g2GIdx{nlevel}, Bs{nlevel}, 1, nzidx{nlevel}, Hnonzeros0{nlevel}, Hs{nlevel}, 0.7, 0.5, restDs{nlevel}, vs{nlevel}*weight);
    es{nlevel} = zs2{nlevel} -zs1{nlevel};
    for level = nlevel - 1:-1:1
        dir = Ps{level} * es{level+1}; dir(Bs{level}, :);
        dir(Bs{level}, :) = 0;
        grad = fun_grad_diag(ts{level}, zs2{level}, g2GIdx{level}, restDs{level}, level);
        zs2{level} = lineSearchV(ts{level}, zs2{level}, grad, dir, vs{level}(:)*weight, 1/(0.7^2), 0.7, 0.5, restDs{level});
        zs2{level} = optimize(xs{level}, ts{level}, restDs{level}, zs2{level}, g2GIdx{level}, Bs{level}, 1, vs{level}*weight, level);
        es{level} = zs2{level} - zs1{level};
    end
    zs2{1} = optimize(xs{1}, ts{1}, restDs{1}, zs2{1}, g2GIdx{1}, Bs{1}, 2, vs{1}*weight, 1);
    zs1{1} = zs2{1}; w = zs2{1};
    As=area_2d(w,t);
    nflip = sum(As<=0);
    if toc(total) >= 50
        toc(total)
        flip = find(area_2d(w, t) <= 0);
        plotMeshWithFlip2d(w, t, flip, []);
        break;
    end
    
    if nflip <= minFlip
        minFlip = nflip;
        minW = w;
    end
    flipTri(iter) = nflip;
    if mod(iter,100) == 0 && nflip < 10 && nflip > 0
        flip = area_2d(w,ts{1}) <= 0;
        [w, iterPost, nflip] = postProcess(w, t, flip, vertTri, A, 200);
    end
    if mod(iter,100) == 0 && nflip > 0 && minFlip < 10
        flip = area_2d(minW,ts{1}) <= 0;
        [w1, iterPost, nflip] = postProcess(minW, t, flip, vertTri, A, 200, x);
        if nflip == 0
            w = w1;
            post = false;
            minAreaMG(iter) = min(area_2d(w,t));
            totalAreaMG(iter) = sum(area_2d(w,t)<=0);
            break;
        end
    end
    if mod(iter,100) == 0 && nflip > 0 && nflip >= 10
        flip = area_2d(w,ts{1}) <= 0;
        w = postProcess2(w, t, flip, vertTri, A, 200, x);
        nflip = sum(area_2d(w,ts{1}) <= 0);
    end
    if mod(iter,100) == 0 && nflip > 0 && nflip < 10
        flip = area_2d(w,ts{1}) <= 0;
        [w, iterPost, nflip] = postProcess(w, t, flip, vertTri, A, 200, x);
    end
    fprintf("it: %d, flipTri: %d, flipArea: %.4e\n", iter, nflip, sum(min(As, 0)));
    if nflip == 0
        post = false;
        break;
    end
    if nflip == 0
        post = false;
        toc(postTT)
        break;
    end
    if nflip < 10 && first10
        flip = area_2d(w,ts{1}) <= 0;
        [w, iterPost] = postProcess(w, t, flip, vertTri, A, 200);
        for level = 1:nlevel
            restDs{level} = restDs{level}*0.1;
        end
        first10 = false;
    end
    if nflip < 3 && first3
        for level = 1:nlevel
            restDs{level} = restDs{level}*0.1;
        end
        first3 = false;
    end
    if iter == 200
        for level = 1:nlevel
            restDs{level} = restDs{level}*0.1;
        end
    end
    if iter == 400
        for level = 1:nlevel
            restDs{level} = restDs{level}*0.1;
        end
    end
    if iter == 600
        for level = 1:nlevel
            restDs{level} = restDs{level}*0.1;
        end
    end
    totalAreaMG(iter) = -sum(min(area_2d(w,t),0));
    timeMG(iter) = toc(total);
    zs2{1} = w;
    zs1{1} = zs2{1};
end

toc(total);

w = zs2{1};



flip = find(area_2d(w, t) <= 0);
plotMeshWithFlip2d(w, t, flip, []);

writeOBJ(fullfile(fileDir, 'result.obj'), w, t);