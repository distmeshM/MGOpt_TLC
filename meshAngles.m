function dim3_angle=meshAngles(x,t)
Point1=x(t(:,1),:);
Point2=x(t(:,2),:);
Point3=x(t(:,3),:);
temp1=[Point2-Point1;Point3-Point2;Point1-Point3];
temp2=[Point3-Point1;Point1-Point2;Point2-Point3];
dim3_angle=reshape(acos(dot(temp1,temp2,2)./(vecnorm(temp1,2,2).*vecnorm(temp2,2,2))),size(t,1),3);