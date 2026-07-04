function plotMeshWithFlip2d(x, t, flipIds, iter)
    B = findBoundary(x, t);
    
    figure('Position', [100, 100, 800, 600], 'Color', 'none');
    axes('Color', 'none');   
    edgeColor = [0.3 0.3 0.3];  
    faceColor = [0.9 0.9 0.9];  

    h_background = drawmesh(t, x, [], edgeColor, faceColor);

    h_background.FaceAlpha = 0.3;  
    h_background.EdgeAlpha = 0.5;  
    
    hold on;
    
    if ~isempty(flipIds)
        flip_t = t(flipIds, :);
        
        h_flip = drawmesh(flip_t, x, [], 'r', [1 0.6 0.6]);  % 浅红色填充，红色边
        
        h_flip.FaceAlpha = 0.7;  
        h_flip.EdgeColor = 'r';  
        
        flip_vertices = unique(flip_t(:));
        
    end
    
    if ~isempty(B)
        hold on;
        tB = [B; B(1)];  
        
        if size(x, 2) == 2
            h_boundary = plot(x(tB, 1), x(tB, 2), 'b-', 'LineWidth', 2.5);
        elseif size(x, 2) == 3
            h_boundary = plot3(x(tB, 1), x(tB, 2), x(tB, 3), 'b-', 'LineWidth', 2.5);
        end
    end
    if nargin >= 4
        if isscalar(iter)
            if iter>1
                annotation('textbox', [0.4, 0.2, 0.2, 0.05], ...
                    'String', sprintf('%d iters', iter), ...
                    'Color', 'k', ...
                    'FontSize', 30, ...
                    'HorizontalAlignment', 'center', ...
                    'EdgeColor', 'none', ...
                    'BackgroundColor', 'none');
            else
                annotation('textbox', [0.4, 0.2, 0.2, 0.05], ...
                    'String', sprintf('%d iter', iter), ...
                    'Color', 'k', ...
                    'FontSize', 30, ...
                    'HorizontalAlignment', 'center', ...
                    'EdgeColor', 'none', ...
                    'BackgroundColor', 'none');
            end
        else
            annotation('textbox', [0.4, 0.2, 0.2, 0.05], ...
                    'String', iter, ...
                    'Color', 'k', ...
                    'FontSize', 30, ...
                    'HorizontalAlignment', 'center', ...
                    'EdgeColor', 'none', ...
                    'BackgroundColor', 'none');
        end
    end
    
    axis equal;
    axis tight;
    
    if size(x, 2) == 2
        view(2);  
    elseif size(x, 2) == 3
        view(3);  
    end
    
    grid on;
    box on;
    
    if size(x, 2) >= 1, xlabel('X'); end
    if size(x, 2) >= 2, ylabel('Y'); end
    if size(x, 2) >= 3, zlabel('Z'); end
    
    
    hold off;
end

function B = findBoundary(x, t)
    edges = [t(:,[1,2]); t(:,[2,3]); t(:,[3,1])];
    edges = sort(edges, 2);
    edges = unique(edges, 'rows');
    
    allEdges = [t(:,[1,2]); t(:,[2,3]); t(:,[3,1])];
    allEdges = sort(allEdges, 2);
    [uniqueEdges, ~, ic] = unique(allEdges, 'rows');
    edgeCount = accumarray(ic, 1);
    
    boundaryEdges = uniqueEdges(edgeCount == 1, :);
    
    if ~isempty(boundaryEdges)
        B = boundaryEdges(1, :)';
        currentEdge = B(end);
        remainingEdges = boundaryEdges(2:end, :);
        
        while ~isempty(remainingEdges)
            idx = find(remainingEdges(:,1) == currentEdge | ...
                       remainingEdges(:,2) == currentEdge, 1);
            
            if isempty(idx)
                break;
            end
            
            nextEdge = remainingEdges(idx, :);
            if nextEdge(1) == currentEdge
                currentEdge = nextEdge(2);
            else
                currentEdge = nextEdge(1);
            end
            
            B = [B; currentEdge];
            remainingEdges(idx, :) = [];
        end
    else
        B = [];
    end
end