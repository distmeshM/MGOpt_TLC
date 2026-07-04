function L=laplacian(x, t, type, normalize, triWts)

% function L = laplacian(x, t, type, normalize)
%
% Compute Laplcian variants for a given triangle mesh
%
% Input parameters:
% x - mesh geometry
% t - mesh connectivity (list of triplets)
% type - variant of Laplacian (can be 'uniform,'cot',
%       'meanvalue','random')
% normalize - binary indicator of whether to normalize the weights
% triWts - for cotangent laplacian, reweight for per triangle
if nargin<3, type = 'cot'; end


if nargin<4, normalize = false; end

if nargin<5, triWts = 1; end

if ~isreal(x), x = [real(x) imag(x)]; end

n = size(x, 1);

if strcmp(type,'uniform')
    VV = sparse(t, t(:, [2 3 1]), true, n, n);
    L = VV | VV';
    L = spdiags(-sum(L,2), 0, double(L));
elseif strcmp(type,'random')
    VV = sparse(t, t(:, [2 3 1]), true, n, n);
    L = VV | VV';

    [i, j] = find(L);
    L = sparse(i, j, rand(size(i)), n, n);
    L = spdiags(-sum(L,2), 0, L);
elseif strcmp(type,'cot')
    cots = cot( meshAngles(x, t) );
    
    if numel(triWts>1), cots = bsxfun(@times, cots, triWts); end
%     areasign = 1-(signedAreas(x, t)<0)*2;
%     cots = bsxfun(@times, cots, areasign);
%     smallarea = abs( signedAreas(x, t) )>1e-8;
%     smallarea = any(cots>2e4 | cots<-2e4, 2);
%     cots = bsxfun(@times, cots, smallarea);

    L = sparse( t(:,[2 3 1 3 1 2]), t(:,[3 1 2 2 3 1]), [cots cots], n, n );
    L = spdiags(-sum(L,2), 0, L);
elseif strcmp(type,'meanvalue')
    el2s = meshFaceEdgeLen2s(x, t);
    angles = meshAnglesFromFaceEdgeLen2(el2s);
    
    w = tan( angles(:, [2 3 1 3 1 2])/2 )./el2s(:, [1 2 3 1 2 3]).^0.5;

%     smallarea = abs( signedAreas(x, t) )>1e-8;
%     w = bsxfun(@times, w, smallarea);

    L = sparse( t(:,[2 3 1 3 1 2]), t(:,[3 1 2 2 3 1]), w, n, n );
    L = spdiags(-sum(L,2), 0, L);    
elseif strcmp(type,'cr')
    L = sparse([],[],[],n,n,n*6);
    % L=sparse(n,n);

    nf = size(t,1);
    for i=1:nf
        xi = t(i,:);
        esi = [1 1 2];
        eei = [2 3 3];
        li = sub2ind( [n n], xi(esi), xi(eei) );
        lit =sub2ind( [n n], xi(eei), xi(esi) );

        
        e1=norm( x(xi(2),:)-x(xi(3),:) );
        e2=norm( x(xi(1),:)-x(xi(3),:) );
        e3=norm( x(xi(1),:)-x(xi(2),:) );

        alphas = acos( [(e2^2+e3^2-e1^2)/(2*e2*e3) (e3^2+e1^2-e2^2)/(2*e3*e1) (e1^2+e2^2-e3^2)/(2*e1*e2)] );
        if abs(sum(alphas)-pi)>1e-8, continue; end
        if abs( abs(e1-e2)-e3 ) < 1e-7 || abs(e1+e2-e3) < 1e-7, continue; end

        L(li) = L(li)-[ cot(alphas(1))*cot(alphas(2))/sin(alphas(3))^2 ...
                        cot(alphas(1))*cot(alphas(3))/sin(alphas(2))^2 ...
                        cot(alphas(2))*cot(alphas(3))/sin(alphas(1))^2 ];
        L(lit) = L(li);       
    end

    L = spdiags(zeros(n,1), 0, L);
    L = spdiags(-sum(L,2), 0, L);
else
    warning( 'unknown Laplacian type: %s', type );
end

if normalize || strcmp(type, 'meanvalue')
    L = bsxfun(@times, L, 1./diag(L) );
    assert( all(~any(isnan(L))) );
end

% if any(L<0), disp('non-delaunay triangulation!'); end
% L( abs(L)<1e-10 ) = 0;
