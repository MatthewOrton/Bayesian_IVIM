% [stats,raw] = BayesianHierarchicalIVIM(b,data,options)
%
% Function for Bayesian estimation using shrinkage prior with IVIM model,
% see Orton et al. (2013) MRM, DOI: 10.1002/mrm.24649
%
% Requires Optimization and Statistics toolboxes.
%
% b       : array of b-values, should be (# b-values) x 1
%
% data    : array of voxel data, should be (# b-values) x (# voxels)
%
% options : optional input - any or all missing values default to values in brackets:
%
%              options.figFlag         (false) indicates whether to display diagnostic figure to check if algorithm has worked OK.
%              options.nMCMC           (22000) number of iterations for MCMC routine.
%              options.nBurnIn         (2000)  number of initial iterations to discard.
%              options.nUpdateRW       (100)   number of iterations between update of random walk variances.
%              options.nPriorFixed     (200)   number of iterations at start where the prior parameters are not updated.
%              options.noProgress      (false) show progress bar
%              options.priorInit.mu    (empty) 3x1 array with initial values for global mean
%              options.priorInit.Sigma (empty) 3x3 array with initial values for global covariance
%              options.limits          (empty) optional limits on D and D* via options.limits.D = [Dlow Dhigh]; and similar for Ds
%              options.dataSigma       (empty) optional data noise (standard deviation), otherwise will use marginalised likelhood
%
% stats   : structure with posterior mean and std for voxel parameters and population parameters
%
% raw     : structure with the raw MCMC outputs
%
% Example:
%
% load('TestData.mat')
% stats = BayesianHierarchicalIVIM(b,data,struct('figFlag',true,'nMCMC',4000));
%

function [stats,raw] = BayesianHierarchicalIVIM(b,data,options)

% reset random number generator to give repeatable results
rng('default')

% default any missing options
if ~exist('options','var') || ~isfield(options,'figFlag')
    options.figFlag = true;
end
if ~exist('options','var') || ~isfield(options,'nMCMC')
    options.nMCMC = 12000;
end
if ~exist('options','var') || ~isfield(options,'nBurnIn')
    options.nBurnIn = 2000;
end
if ~exist('options','var') || ~isfield(options,'nUpdateRW')
    options.nUpdateRW = 100;
end
if ~exist('options','var') || ~isfield(options,'nPriorFixed')
    options.nPriorFixed = 200;
end
if ~exist('options','var') || ~isfield(options,'noProgress')
    options.noProgress = false;
end
if ~exist('options','var') || ~isfield(options,'priorInit')
    options.priorInit.mu = [];
    options.priorInit.Sigma = [];
end
if ~exist('options','var') || ~isfield(options,'limits')
    options.limits = [];
end
if ~exist('options','var') || ~isfield(options,'dataSigma')
    options.dataSigma = [];
end

if ~isempty(options.limits) && options.nPriorFixed~=options.nMCMC
    options.nPriorFixed = options.nMCMC;
end


% size of data array
[nB,nPixels] = size(data);

% parameters for algorithm
nMCMC = options.nMCMC;
nBurnIn = options.nBurnIn;
nUpdateRW = options.nUpdateRW;

Iattenuate = true(size(data(1,:))); %data(2,:)<data(1,:);

% initialise stats output
stats.sigmoidf.lsq = zeros(1,nPixels);
stats.sigmoidf.bsp.mean = zeros(1,nPixels);
stats.sigmoidf.bsp.std = zeros(1,nPixels);
stats.logDs.lsq = zeros(1,nPixels);
stats.logDs.bsp.mean = zeros(1,nPixels);
stats.logDs.bsp.std = zeros(1,nPixels);
stats.logD.lsq = zeros(1,nPixels);
stats.logD.bsp.mean = zeros(1,nPixels);
stats.logD.bsp.std = zeros(1,nPixels);
stats.S0.lsq = zeros(1,nPixels);
stats.S0.bsp.mean = zeros(1,nPixels);
stats.mu.mean = zeros(3,1);
stats.mu.cov  = zeros(3,3);
stats.Sigma.mean = zeros(3,3);
stats.Sigma.std  = zeros(3,3);

% initialise raw output
raw.sigmoidf = zeros(nMCMC,nPixels);
raw.logDs = zeros(nMCMC,nPixels);
raw.logD = zeros(nMCMC,nPixels);
raw.mu = zeros(3,nMCMC);
raw.Sigma = zeros(3,3,nMCMC);

%%%%%%%%%%%%%%%%%%%%%%
% LSQ initialisation %
%%%%%%%%%%%%%%%%%%%%%%

waitStr = 'IVIM';

if ~options.noProgress
    hWait = waitbar(0,'','Name',['Bayesian ' waitStr ': Least-squares initialisation'],...
        'CreateCancelBtn',...
        'setappdata(gcbf,''canceling'',1)');
end


if isempty(options.limits)
    logDsLim = [-8 log(2)];
    logDLim = [-10 -4];
    sigmoidfLim = [-10 0];
else
    logDsLim = log(options.limits.Ds);
    logDLim = log(options.limits.D);
    if isfield(options.limits,'f')
        sigmoidfLim = sigmoid(options.limits.f,-1);
    else
        sigmoidfLim = [-10 10];
    end
end

if isfield(options','init')
    LSQ.f = options.init.f;
    LSQ.d = options.init.D;
    LSQ.ds = options.init.Ds;
else
    for n = 1:nPixels
        yy = data(:,n); % cheat to get the data into the regression function
        xInit = [mean(logDsLim) mean(logDLim)];
        lb = [logDLim(1) logDsLim(1)];
        ub = [logDLim(2) logDsLim(2)];
        z = lsqcurvefit(@modelFuncIVIM, xInit, b, yy, lb, ub, optimset('Display','off'));
        [~,Sopt,Dopt,DsOpt] = modelFuncIVIM(z,b);
        LSQ.S0(n) = Sopt(1) + Sopt(2);
        LSQ.f(n) = Sopt(2)/LSQ.S0(n);
        LSQ.ds(n) = DsOpt;
        LSQ.d(n) = Dopt;
        if ~options.noProgress && mod(n,50)==0
            waitbar(n/nPixels,hWait,[num2str(n) '/' num2str(nPixels) ' voxels fitted ...'])
        end
        
        % quit if cancelled
        if ~options.noProgress
            if getappdata(hWait,'canceling')
                break
            end
        end

    end
%    opt = IVIMfastFitLsq(b,data,b,struct('DsMin',exp(-7.5)));
% LSQ.f = opt.f;
% LSQ.d = opt.D;
% LSQ.ds = opt.Ds;
end
% for nn = 1:nPixels
%     yy = data(:,nn);
%     gg =  LSQ.f(nn)*exp(-b*LSQ.ds(nn)) + (1-LSQ.f(nn))*exp(-b*LSQ.d(nn));
%     LSQ.S0(nn) = sum(yy.*gg)/sum(gg.^2);
% end
idx = (LSQ.f<sigmoid(sigmoidfLim(1)+0.05)) | (LSQ.f>sigmoid(sigmoidfLim(2)-0.05));
LSQ.f(idx) = median(LSQ.f(~idx));

idx = (LSQ.d<exp(logDLim(1)+0.05)) | (LSQ.d>exp(logDLim(2)-0.05));
LSQ.d(idx) = median(LSQ.d(~idx));

idx = (LSQ.ds<exp(logDsLim(1)+0.05)) | (LSQ.ds>exp(logDsLim(2)-0.05));
LSQ.ds(idx) = median(LSQ.ds(~idx));


if ~options.noProgress
    set(hWait,'Name',['Bayesian ' waitStr ': MCMC progress'])
end


% tweak f so it is strictly >0 and <1
LSQ.f = 0.999*LSQ.f+0.0005;

% replicated b-values for efficient calculations later on
bArr = repmat(b,1,nPixels);



%%%%%%%%%%%%%%%%%%%%%%%
% MCMC initialisation %
%%%%%%%%%%%%%%%%%%%%%%%


% initialise pixel parameters
[sigmoidf,logD,logDs] = deal(NaN(nMCMC,nPixels));
sigmoidf(1,:) = log(LSQ.f./(1-LSQ.f));
logD(1,:) = log(LSQ.d);
logDs(1,:) = log(LSQ.ds);

% initialise population parameters
mu = NaN(3,nMCMC);
Sigma = NaN(3,3,nMCMC);
if ~isempty(options.priorInit.mu)
    mu(:,1) = options.priorInit.mu(:);
else
    mu(:,1) = mean([sigmoidf(1,Iattenuate); logD(1,Iattenuate); logDs(1,Iattenuate)],2);
end
if ~isempty(options.priorInit.Sigma)
    Sigma(:,:,1) = options.priorInit.Sigma;
else
    Sigma(:,:,1) = cov([sigmoidf(1,Iattenuate); logD(1,Iattenuate); logDs(1,Iattenuate)]');
end

% initialise random walk parameters and acceptance counts
sRWf = 0.5*ones(1,nPixels);
sRWd = 0.2*ones(1,nPixels);
sRWds = 0.5*ones(1,nPixels);
accF = zeros(1,nPixels);
accD = zeros(1,nPixels);
accDs = zeros(1,nPixels);

%%%%%%%%%%%%%%%%%%%%%
% Main MCMC section %
%%%%%%%%%%%%%%%%%%%%%

for n = 2:nMCMC
    
    % update sigmoid-f
    sigmoidf(n,:) = sigmoidf(n-1,:) + sRWf.*randn(1,nPixels);
    pOld = posterior(sigmoidf(n-1,:),logD(n-1,:),logDs(n-1,:),mu(:,n-1),Sigma(:,:,n-1));
    pNew = posterior(sigmoidf(n,:),logD(n-1,:),logDs(n-1,:),mu(:,n-1),Sigma(:,:,n-1));
    Imove = rand(1,nPixels)<exp(pNew-pOld);
    accF = accF + Imove;
    sigmoidf(n,~Imove) = sigmoidf(n-1,~Imove);
    
    % update log D
    logD(n,:) = logD(n-1,:) + sRWd.*randn(1,nPixels);
    pOld = posterior(sigmoidf(n,:),logD(n-1,:),logDs(n-1,:),mu(:,n-1),Sigma(:,:,n-1));
    pNew = posterior(sigmoidf(n,:),logD(n,:),logDs(n-1,:),mu(:,n-1),Sigma(:,:,n-1));
    Imove = rand(1,nPixels)<(exp(pNew-pOld).*(logD(n,:)<logDs(n-1,:)));
    accD = accD + Imove;
    logD(n,~Imove) = logD(n-1,~Imove);
    
    % update log Dstar
    logDs(n,:) = logDs(n-1,:) + sRWds.*randn(1,nPixels);
    pOld = posterior(sigmoidf(n,:),logD(n,:),logDs(n-1,:),mu(:,n-1),Sigma(:,:,n-1));
    pNew = posterior(sigmoidf(n,:),logD(n,:),logDs(n,:),mu(:,n-1),Sigma(:,:,n-1));
    Imove = rand(1,nPixels)<(exp(pNew-pOld).*(logD(n,:)<logDs(n,:)));
    accDs = accDs + Imove;
    logDs(n,~Imove) = logDs(n-1,~Imove);
    
    Msigmoidf = mean(sigmoidf(n,Iattenuate));
    MlogD = mean(logD(n,Iattenuate));
    MlogDs = mean(logDs(n,Iattenuate));
    R = [sigmoidf(n,Iattenuate)-Msigmoidf; logD(n,Iattenuate)-MlogD; logDs(n,Iattenuate)-MlogDs];
    
    % only update the population parameters after 200 iterations
    if n>options.nPriorFixed
        dof = length(Iattenuate) - 3;
        beta = 10; 
        alpha = 23;
        Sigma(:,:,n) = iwishrnd(beta*diag([1 1 1]) + R*R',alpha+dof); % hyper-parameters set so with no data the prior has a diagonal covariance = 10/20 = 0.5 i.e. quite broad on f,d,d*
        % cheat way of applying hierarchical distribution independently to
        % each parameter
        %Sigma(:,:,n) = diag(diag(Sigma(:,:,n)));
        mu(:,n) = [Msigmoidf; MlogD; MlogDs] + sqrtm(Sigma(:,:,n)/nPixels)*randn(3,1);
    else
        Sigma(:,:,n) = Sigma(:,:,n-1);
        mu(:,n) = mu(:,n-1);
    end
    
    % update random walk parameters every nUpdateRW iterations during the burn-in period
    if mod(n,nUpdateRW)==0 && n<nBurnIn
        sRWf = 0.5*sRWf.*(nUpdateRW+1)./(nUpdateRW+1-accF);
        accF(:) = 0;
        sRWd = 0.5*sRWd.*(nUpdateRW+1)./(nUpdateRW+1-accD);
        accD(:) = 0;
        sRWds = 0.5*sRWds.*(nUpdateRW+1)./(nUpdateRW+1-accDs);
        accDs(:) = 0;
    end
    
    % update progress bar
    if ~options.noProgress && mod(n,100)==0
        waitbar(n/nMCMC,hWait,[num2str(n) '/' num2str(nMCMC) ' iterations completed ...'])
    end
    
    % quit if cancelled
    if ~options.noProgress
        if getappdata(hWait,'canceling')
            break
        end
    end
    
    
end

if ~options.noProgress
    delete(hWait)
end

% quit if cancel button pressed
if n<nMCMC
    return
end

% compile outputs
stats.sigmoidf.lsq = sigmoidf(1,:);
stats.sigmoidf.bsp.mean = mean(sigmoidf(nBurnIn+1:end,:)); % see Eq [9]
stats.sigmoidf.bsp.std = std(sigmoidf(nBurnIn+1:end,:));  % see Eq [11]

stats.logD.lsq = logD(1,:);
stats.logD.bsp.mean = mean(logD(nBurnIn+1:end,:));
stats.logD.bsp.std = std(logD(nBurnIn+1:end,:));

stats.logDs.lsq = logDs(1,:);
stats.logDs.bsp.mean = mean(logDs(nBurnIn+1:end,:));
stats.logDs.bsp.std = std(logDs(nBurnIn+1:end,:));

stats.mu.mean = mean(mu(:,nBurnIn+1:end),2); % see Eq [10]
stats.mu.cov  = cov(mu(:,nBurnIn+1:end)');
stats.Sigma.mean = mean(Sigma(:,:,nBurnIn+1:end),3);
stats.Sigma.std  = std(Sigma(:,:,nBurnIn+1:end),[],2);

[~,stats.S0.bsp.mean] = posterior(stats.sigmoidf.bsp.mean,stats.logD.bsp.mean,stats.logDs.bsp.mean,stats.mu.mean,stats.Sigma.mean);
[~,stats.S0.lsq] = posterior(sigmoidf(1,:),logD(1,:),logDs(1,:),mu(:,1),Sigma(:,:,1));

raw.sigmoidf = sigmoidf;
raw.logD = logD;
raw.logDs = logDs;

raw.mu = mu;
raw.Sigma = Sigma;


% display some outputs
if options.figFlag
    
    figure('position',get(0,'ScreenSize'))
    
    % display trace of 200 voxels
    skipPix = max([1 floor(nPixels/200)]);
    skipIter = max([1 floor(nMCMC/200)]);
    subplot(5,3,1);
    plot((1:skipIter:nMCMC)',sigmoidf(1:skipIter:end,1:skipPix:end),'b');
    hold on;
    if isempty(options.limits)
        plot((1:skipIter:nMCMC),mu(1,1:skipIter:end),'r');
        plot((1:skipIter:nMCMC),mu(1,1:skipIter:end)+1.96*sqrt(squeeze(Sigma(1,1,1:skipIter:end)))','r');
        plot((1:skipIter:nMCMC),mu(1,1:skipIter:end)-1.96*sqrt(squeeze(Sigma(1,1,1:skipIter:end)))','r');
    end
    axis tight
    xlim([0 nMCMC])
    yL = ylim;
    set(fill3([0 nBurnIn nBurnIn 0 0],[yL(1) yL(1) yL(2) yL(2) yL(1)],[1 1 1 1 1],'w'),'EdgeColor','none','FaceAlpha',0.6)
    title('f (MCMC trace)')
    
    subplot(5,3,2); plot((1:skipIter:nMCMC)',logD(1:skipIter:end,1:skipPix:end),'b'); hold on;
    if isempty(options.limits)
        plot((1:skipIter:nMCMC),mu(2,1:skipIter:end),'r');
        plot((1:skipIter:nMCMC),mu(2,1:skipIter:end)+1.96*sqrt(squeeze(Sigma(2,2,1:skipIter:end)))','r');
        plot((1:skipIter:nMCMC),mu(2,1:skipIter:end)-1.96*sqrt(squeeze(Sigma(2,2,1:skipIter:end)))','r');
    end
    title('log-D (MCMC trace)')
    axis tight
    xlim([0 nMCMC])
    yL = ylim;
    set(fill3([0 nBurnIn nBurnIn 0 0],[yL(1) yL(1) yL(2) yL(2) yL(1)],[1 1 1 1 1],'w'),'EdgeColor','none','FaceAlpha',0.6)
    
    subplot(5,3,3); plot((1:skipIter:nMCMC)',logDs(1:skipIter:end,1:skipPix:end),'b'); hold on;
    if isempty(options.limits)
        plot((1:skipIter:nMCMC),mu(3,1:skipIter:end),'r');
        plot((1:skipIter:nMCMC),mu(3,1:skipIter:end)+1.96*sqrt(squeeze(Sigma(3,3,1:skipIter:end)))','r');
        plot((1:skipIter:nMCMC),mu(3,1:skipIter:end)-1.96*sqrt(squeeze(Sigma(3,3,1:skipIter:end)))','r');
    end
    title('log-D* (MCMC trace)')
    axis tight
    xlim([0 nMCMC])
    yL = ylim;
    set(fill3([0 nBurnIn nBurnIn 0 0],[yL(1) yL(1) yL(2) yL(2) yL(1)],[1 1 1 1 1],'w'),'EdgeColor','none','FaceAlpha',0.6)
    
    subplot(5,3,4); hist(mu(1,nBurnIn:n),100); title('p(\mu_f | data)')
    subplot(5,3,6); hist(mu(3,nBurnIn:n),100); title('p(\mu_{d*} | data)')
    subplot(5,3,5); hist(mu(2,nBurnIn:n),100); title('p(\mu_d | data)')
        
    subplot(5,3,7); hist(sqrt(squeeze(Sigma(1,1,nBurnIn:n))),100); title('p(\Sigma_{ff}^{1/2} | data)')
    subplot(5,3,9); hist(sqrt(squeeze(Sigma(3,3,nBurnIn:n))),100); title('p(\Sigma_{d*d*}^{1/2} | data)')
    subplot(5,3,8); hist(sqrt(squeeze(Sigma(2,2,nBurnIn:n))),100); title('p(\Sigma_{dd}^{1/2} | data)')
    
    subplot(5,3,10); [~,bn] = ksdensity([sigmoidf(1,:) mean(sigmoidf(nBurnIn:n,:))],'NumPoints',200); ksdensity(sigmoidf(1,:),bn); hold on; ksdensity(mean(sigmoidf(nBurnIn:n,:)),bn); title('voxel histogram, f')
    subplot(5,3,12); [~,bn] = ksdensity([logDs(1,:) mean(logDs(nBurnIn:n,:))],'NumPoints',200); ksdensity(logDs(1,:),bn); hold on; ksdensity(mean(logDs(nBurnIn:n,:)),bn); title('voxel histogram, d*')
    subplot(5,3,11); [~,bn] = ksdensity([logD(1,:) mean(logD(nBurnIn:n,:))],'NumPoints',200); ksdensity(logD(1,:),bn); hold on; ksdensity(mean(logD(nBurnIn:n,:)),bn); title('voxel histogram, d'); legend('LSQ','BSP','Location','NorthWest')    
    
    subplot(5,3,13);
    plot(sigmoidf(1,:),logD(1,:),'.',mean(sigmoidf(nBurnIn:n,:)),mean(logD(nBurnIn:n,:)),'r.');
    xlabel('f'); ylabel('d'); title('voxel correlation'); 
    
    subplot(5,3,14);
    plot(logD(1,:),logDs(1,:),'.',mean(logD(nBurnIn:n,:)),mean(logDs(nBurnIn:n,:)),'r.');
    xlabel('d'); ylabel('d*'); title('voxel correlation'); legend('LSQ','BSP','Location','NorthWest')
    
    subplot(5,3,15);
    plot(logDs(1,:),sigmoidf(1,:),'.',mean(logDs(nBurnIn:n,:)),mean(sigmoidf(nBurnIn:n,:)),'r.');
    xlabel('d*'); ylabel('f'); title('voxel correlation'); 
    
    % find axes and change axis labels etc.
    Haxes = findobj(gcf,'Type','axes');
    set(Haxes,'FontSize',14)
    
    for n = 1:length(Haxes)
        % find any text objects in axes
        Htext = findall(Haxes,'Type','text');
        set(Htext,'FontSize',14)
    end
    
%     subplot(5,6,32:35)
%     errorbar(b,mean(data,2)./mean(data(1,:)),std(data./data(1,:),[],2))
%     set(gca,'YScale','log')
%     title(num2str(size(data,2)))
    
    drawnow
end


% function to compute log-posterior for all voxels - includes likelihood
% and prior
    function [logP,S0H] = posterior(f,d,ds,m,E)
        fHere = repmat(exp(f)./(1+exp(f)),nB,1);
        G = fHere.*exp(-bArr.*exp(repmat(ds,nB,1))) + (1-fHere).*exp(-bArr.*exp(repmat(d,nB,1)));
        sumDataG = sum(data.*G);
        S0H = sumDataG./sum(G.^2);
        if isempty(options.dataSigma)
            logLH = -0.5*nB*log(sum(data.^2) - S0H.*sumDataG);
        else
            logLH = -0.5*(sum(data.^2) - S0H.*sumDataG)/options.dataSigma^2;
        end
        r = [f; d; ds]-repmat(m,1,length(f));
        if isempty(options.limits)
            logPr = - 0.5*sum(r.*(E\r)) - 0.5*log(det(E));
        else
            logPr = - 0.5*sum(r.*(E\r)) - 0.5*log(det(E));
            logPr = logPr + log(double(d>logDLim(1) & d<logDLim(2) ...
                & ds>logDsLim(1) & ds<logDsLim(2) ...
                & f>sigmoidfLim(1) & f<sigmoidfLim(2)));
        end
        logP = logLH + logPr;
    end


% function to use in LSQ fitting:
% - this function implicitly optimises over S0 and f, so lsqcurvefit only has to optimise over D and D*
% - D = exp(xIn(1)) and D* = sum(exp(xIn)) which automatically ensures D*>D
    function [curve,S,D,Ds] = modelFuncIVIM(xIn,bIn)
        D = exp(xIn(1));
        Ds = sum(exp(xIn));
        G = [exp(-bIn(:)*D)  exp(-bIn(:)*Ds)];
        % implicit optimisation of S0 and f
        S = (G'*G)\(G'*yy);
        % special bit for implicit optimisation of S0 and f to make sure
        % they are both positive
        if S(1)<0 || S(2)<0
            Sc(:,1) = [sum(G(:,1).*yy)/sum(G(:,1).^2); 0];
            Sc(:,2) = [0 sum(G(:,2).*yy)/sum(G(:,2).^2)];
            Sc(:,3) = [0; 0];
            rss = sum((G*Sc - repmat(yy,1,3)).^2);
            [~,II] = min(rss);
            S = Sc(:,II);
        end
        curve = G*S;
    end

    function y = sigmoid(x,direction)
        
        if nargin==1, direction = 1; end
        
        if direction==1
            y = exp(x)./(1+exp(x));
            y(x>700) = 1;
        elseif direction==-1
            y = log(x) - log(1-x);
        end
    end
end
