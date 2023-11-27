function [recommended_frequencies] = TGC_plotter(allMI,theta,gamma,tlothi,gloghi,res,measName)

meas= size(allMI,3);
% measName={'dPAC','MI','PLV','GLMF','VTK'};
for j = 1:meas
   MI = allMI(:,:,j);
 h(j)=figure ;
theta=mean(theta,1); % makes axis for plots 
gamma=mean(gamma(:,:,1),1);
tlo = find(theta>=tlothi(1),1,'first');
thi = find(theta<=tlothi(2),1,'last');
glo = find(gamma>=gloghi(1),1,'first');
ghi = find(gamma<=gloghi(2),1,'last');

croppedMI=MI(glo:ghi,tlo:thi);
croppedtheta=theta(tlo:thi);
croppedgamma=gamma(glo:ghi);
d=croppedMI;
 thres = (max([min(max(d,[],1))  min(max(d,[],2))])) ; 
 filt = (fspecial('gaussian', 7,1)); 
 edg =2;    
 %res = 1;
[cent]=FastPeakFind(croppedMI, thres, filt , edg,res);
cent = round(cent);
clear z HIGHest HIGH x y 
if ~isempty(cent)
x= cent(1:2:end);
y= cent(2:2:end);
if isempty(y)
    x=cent(1);
    y=cent(2);
end


for i=1:length(x)
z(i)= croppedMI(y(i),x(i));
end
HIGHest= find(z==max(z));
HIGH=HIGHest(1);
rec_theta= croppedtheta(x(HIGH));
rec_gamma= croppedgamma(y(HIGH));

disp('recommended individual peak frequencies =')
disp(measName);disp([rec_theta,rec_gamma]);

else 
rec_theta= 999;
rec_gamma= 999;    
disp('sorry :-(')
disp('sorry :-(')
end
hold on
surf(croppedtheta,croppedgamma,croppedMI); shading interp;
 xlabel('Theta (Hz)');view(2);
 ylabel('Gamma (Hz)');
colormap(jet);


plot3(rec_theta,rec_gamma,100*max(MI([cent(2:2:end)],[cent(1:2:end)])),'k*') 
title([rec_theta;rec_gamma]) 


recommended_frequencies = [rec_theta, rec_gamma];
end