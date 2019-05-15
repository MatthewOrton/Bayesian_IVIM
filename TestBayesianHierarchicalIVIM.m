clearvars
close all force

rng(0)

b = [0 10 20 40 100 200 400 800]';

N = 1000;
D = exp(-6.5 + 0.2*randn(1,N));
Ds = exp(-3 + 0.2*randn(1,N));
f = betarnd(30,100,1,N);
S0 = exp(5 + 0.2*randn(1,N));

data = S0.*(f.*exp(-b*Ds) + (1-f).*(exp(-b*D)));
data = data + 8*randn(size(data));

figure
plot(b,data)

out = BayesianHierarchicalIVIM(b,data);


figure('position',get(0,'ScreenSize'))

subplot(2,3,1)
plot(D,exp(out.logD.lsq),'.')
hold on
axis(max(axis).*[0 1 0 1]); ax = axis;
plot(xlim,xlim,'k')
xlabel('D true')
ylabel('D lsq')
title('Least-squares')

subplot(2,3,4)
plot(D,exp(out.logD.bsp.mean),'.')
hold on
axis(ax)
plot(xlim,xlim,'k')
xlabel('D true')
ylabel('D Bayesian')
title('Bayesian')


subplot(2,3,2)
plot(f,exp(out.sigmoidf.lsq)./(1+exp(out.sigmoidf.lsq)),'.')
hold on
axis(max(axis).*[0 1 0 1]); ax = axis;
plot(xlim,xlim,'k')
xlabel('f true')
ylabel('f lsq')
title('Least-squares')

subplot(2,3,5)
plot(f,exp(out.sigmoidf.bsp.mean)./(1+exp(out.sigmoidf.bsp.mean)),'.')
hold on
axis(ax)
plot(xlim,xlim,'k')
xlabel('f true')
ylabel('f Bayesian')
title('Bayesian')

subplot(2,3,3)
loglog(Ds,exp(out.logDs.lsq),'.')
hold on
axis(max(axis).*[0.0001 1 0.0001 1]); ax = axis;
plot(xlim,xlim,'k')
xlabel('D* true')
ylabel('D* lsq')
title('Least-squares')

subplot(2,3,6)
loglog(Ds,exp(out.logDs.bsp.mean),'.')
hold on
axis(ax)
plot(xlim,xlim,'k')
xlabel('D* true')
ylabel('D* Bayesian')
title('Bayesian')

set(findobj(gcf,'FontSize',get(0,'defaultAxesFontSize')),'FontSize',13)
