function c = circle_2d(x, t)
    v1 = x(t(:,1),:) - x(t(:,3),:); v2 = x(t(:,2),:) - x(t(:,3),:); v3 = x(t(:,1),:) - x(t(:,2),:);
    c = sqrt(dot(v1,v1,2)) + sqrt(dot(v2,v2,2)) + sqrt(dot(v3,v3,2));
end