function [triTriMat, vertTriMat] = getTriTri(t, x)
nv = size(x,1);
nf = size(t,1);
vertTriMat = sparse(repmat((1:nf)',3,1),double(t(:)),ones(nf*3,1),nf,nv);
A=vertTriMat*vertTriMat';
A=A-spdiags(diag(A),0,nf,nf);
triTriMat=(A>=2);
end