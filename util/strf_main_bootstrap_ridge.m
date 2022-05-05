function [strf] = strf_main_bootstrap_ridge(X, Y, time_lag, dataf,regalphas, foldN,fullMod_flag, nboots, TrialInd, foldInd, numFolds)
% ridge regression to predict Y based on X.
% Inputs:
%   X - Matrix of independent variables
%   Y - Matrix of dependent variables - each column of Y is estimated
% separately using the whole matrix X.
%   timelag, dataf - parameters to reformat the columns of X and the
% resulting betas into a 2d-matrix, e.g. if X was made out of spectrograms
%   regalphas - regularization values that are used for ridge regression
%   foldN -  .5*foldN is the size of test and ridge sets, (foldN-1)/foldN is
% the subset of data for training of initial model.
%   fullMod_flag - if 1 (default  is 0) the model is refitted using the whole
% data set with the alpha value from the crossvalidation step
%   numFolds - this is < foldN. If 1 - the model is fitted using
% only one partition of the data; If foldN, it is refitted foldN times with
% all possible partitions of the foldN subsets of the data into
% test/train/ridge data sets
%   TrialInd - Indices of the single trials in X and Y. This variable is used
% to make sure that training/test/ridge sets contain complete trials, so
% that these three sets contain independent data. If TrialInd is not passed
% to the function each line of data will be treated as independent trial.

% result of crossvalidation step is saved in temporary file strf_temp.mat
% Yulia Oganian, June 2017

if nargin < 5,regalphas = [ 0 logspace(2 ,7, 30)];end
if nargin < 6,foldN = 5;end
if nargin < 7,fullMod_flag = 0;end
if nargin < 8,nboots = 10;end
if nargin<11, numFolds = 1; end % always doing just one foldend
if nargin < 9,TrialInd = 1:size(X,1);end
nChan = size(Y,2);

if size(Y, 1) ~= size(X, 1)
    warning('Y and X do not match in sizes.aborting.')
    return;
end


%% 5 - fold cross-validation data partition: train on 80%, test on 20
% set_sep = [0.8 0.1 0.1]; %train ridge, test - fits 5-way cross-validation, btu
trialNums = unique(TrialInd);
nTrials = length(trialNums);
% trialNums = trialNums(randperm(nTrials));
if nargin < 10 % if input does not include split in cross-validation sets
    % set_sep = [0.8 0.1 0.1]
    testfoldInd = logical(kron(eye(foldN), ones(ceil(nTrials/foldN),1)));
    trainfoldInd = logical(ones(size(testfoldInd)) - testfoldInd);
    
    testfoldInd = testfoldInd(1:nTrials,:);
    trainfoldInd = trainfoldInd(1:nTrials,:);
else
    testfoldInd = foldInd.test;
    trainfoldInd = foldInd.train;
end

%%
ridgeCorr = cell(numFolds, 1);
totInd = nan(numFolds,1);
chanInd = nan(numFolds,nChan);
testCorrTotal = nan(numFolds, nChan);
totalBestAlpha = nan(numFolds, 1);
BestBeta = cell(1, numFolds);

byChanBestAlpha = nan(numFolds, nChan);
testCorrByChan = nan(numFolds, nChan);
BestBetaAlpha = cell(1, numFolds);
%% loop through folds
for cfold = 1:numFolds
    
    %     fprintf(1,'Running fold %d/%d\n', cfold, numFolds);
    
    %% subsets of data for current fold
    trainInd = ismember(TrialInd, trialNums(trainfoldInd(:,cfold)));
    testInd = ~trainInd;
    ctrStim = X(trainInd,:);
    ctrResp = Y(trainInd,:);
    % this is the set for testing of the final alpha
    ctsStim = X(testInd,:);
    ctsResp = Y(testInd,:);
    
    %% chunks for bootstrapping of ridge parameter
    % size of chunk
    chunksize  = floor(size(ctrStim,1)/(nboots*5));
    %     chunksize  = round(time_lag*dataf*3);
    
    % number of chunks for ridge testing
    nchunks = ceil(size(ctrStim,1)/chunksize);
    chunkNums= 1:nchunks;
    chunkNums = chunkNums(randperm(nchunks));
    % define chunks
    chunkInd = reshape(repmat(chunkNums, chunksize,1),1,[]);
    chunkInd = chunkInd(1:size(ctrStim,1));
 
    % split ctrStim in data to train and in data to test ridge parameter
    ridgeTestfoldInd = logical(kron(eye(nboots),ones(floor(nchunks/nboots),1)));
    %     ridgeTestfoldInd = ridgeTestfoldInd(1:nchunks,:);
    if size(ridgeTestfoldInd,1)<nchunks
        ridgeTestfoldInd(end+1:nchunks,:) = 0;
    end
    ridgeCorr{cfold} = nan(nChan, length(regalphas), nboots);
    save('ridge_test_reg.mat', 'ridgeTestfoldInd', 'chunksize','chunkInd');
    %% bootstrap the ridge alpha
    nalphas = length(regalphas);
    if nalphas > 1
        for cboot = 1:nboots
%             fprintf(1,'Running %d alpha bootstrap # %d/%d \n', nalphas, cboot, nboots);
            
            cridgeTestInd = ismember(chunkInd, chunkNums(ridgeTestfoldInd(:,cboot)));
            
            % ridge test stimuli
            crStim = ctrStim(cridgeTestInd,:);
            crResp = ctrResp(cridgeTestInd,:);
            
            %ridge train stimuli
            crtrStim =ctrStim(~cridgeTestInd,:);
            crtrResp =ctrResp(~cridgeTestInd,:);
            
            % covariance matrix
            covmattr = crtrStim'*crtrStim;
            % eigenvalue decomposition of covariance matrix of full training set
            [U, S] = eig(covmattr); % U is matrix of eigenvectors; s is diagonal matrix of eigenvalues
%             toc
            eigvals = diag(S);
            Usr = U'*(crtrStim'*crtrResp); % projection of stim on resp
            
            % precalculate a few matrix products for speed
            Usl = crStim*U;
            
            %     cBetas{cfold} = nan(size(covmat,1),nChan, length(regalphas));
            
            calphanum = 1;
            for calpha = regalphas
                % ridge D - tikhonov regularization
                D = diag(1./(eigvals+calpha));
                
                % fast version to compute predicted Y without explicit computation
                % of w - <1 sec
                Usl_D = mat_diag_prod(Usl, D);
                predY = Usl_D * Usr;
                
                % correlation between predicted resp and data
                ccor = nan(nChan,1);
                for cchan = 1:nChan
                    ccor(cchan) = corr(predY(:,cchan), crResp(:,cchan));
                end
                
                % save correlations and beta values for later
                ridgeCorr{cfold}(:,calphanum, cboot) = ccor;
                calphanum = calphanum +1;
            end
        end
        
        %% find best alpha value for single channels and across channels
        % by channel
        [~, chanInd(cfold,:)] = nanmax(nanmean(ridgeCorr{cfold},3),[],2);
        byChanBestAlpha(cfold,1:nChan) = regalphas(chanInd(cfold,:));
        % across channels
        [~, totInd(cfold)] = nanmax(nanmean(nanmean(ridgeCorr{cfold},1),3),[],2);
        totalBestAlpha(cfold) = regalphas(totInd(cfold));
    else
        totalBestAlpha(cfold) = regalphas;
        byChanBestAlpha(cfold,1:nChan) = nan;
    end
    %% ---- test model performance on test data
    
    %% covariance matrix of training stimulus set
%     tic
    covmattr = ctrStim'*ctrStim;
    % eigenvalue decomposition of covariance matrix of full training set
    [U, S] = eig(covmattr); % U is matrix of eigenvectors; s is diagonal matrix of eigenvalues
    eigvals = diag(S);
    Usr = U'*(ctrStim'*ctrResp); % projection of stim on resp
    
    % use one best alpha for all channels
    D = diag(1./(eigvals+totalBestAlpha(cfold)));
    
    wt = U*D*Usr;
    %         % predicted response
    predY = ctsStim*wt;
    
    % correlation of test data and pred. data for each chanel and this fold
    for cchan = 1:nChan
        testCorrTotal(cfold,cchan) = corr(predY(:,cchan), ctsResp(:,cchan));
    end
%     toc
    %% use different alpha for each chanel
%     tic
%     alphawt = nan(size(wt));
%     for cchan = 1:nChan
%         D = diag(1./(eigvals+byChanBestAlpha(cfold, cchan)));
%         
%         alphawt(:,cchan) = U*D*Usr(:,cchan);
%         %         % predicted response
%         alphapredY = ctsStim*alphawt(:,cchan);
%         
%         % correlation of test data and pred. data for each chanel and this fold        
%         testCorrByChan(cfold,cchan) = corr(alphapredY, ctsResp(:,cchan));        
%     end
%     BestBetaAlpha{cfold} = alphawt; % saving best beta only
%     toc
    
    %% results of crossvalidation
    % strf.totalBestAlphaInd = totInd;

    BestBeta{cfold} = wt; % saving best beta only
    strf.testStim{cfold} = ctsStim;
    strf.testY{cfold} = ctsResp;
    strf.predY{cfold} = predY;
    
end
%% save results of crossvalidation
strf.ridgeCorr = ridgeCorr;
strf.totalBestAlpha = totalBestAlpha;
strf.byChanBestAlpha = byChanBestAlpha;

strf.testCorrBestAlpha = testCorrTotal;

strf.testfoldInd = testfoldInd;
strf.trainfoldInd = trainfoldInd;

% strf.ridgefoldInd = ridgefoldInd;
% strf.testCorrByChanAlpha = testCorrByChan;


%% reshape model betas to strf
switch length(time_lag)
    case 1
        nfeat = size(X, 2)/(time_lag*dataf+1);
    case 2
        nfeat = size(X, 2)/(sum(time_lag)*dataf);
end
for cfold = 1:numFolds
    strf.strf{cfold} = reshape(BestBeta{cfold},nfeat,[], nChan);
    strf.strfbyChan{cfold} = reshape(BestBetaAlpha{cfold},nfeat,[], nChan);
end
strf.meanStrf = mean(ecog_norm(cell2mat(reshape(strf.strf, 1, 1, 1,length(strf.strf))),2),4);
strf.meanStrfByChan = mean(ecog_norm(cell2mat(reshape(strf.strfbyChan, 1, 1, 1,length(strf.strf))),2),4);
%% calculate STRFs based on best alpha across all chanels using the whole data set

if fullMod_flag
    % covariance matrix of whole stimulus set
    covmattr = X'*X;
    % eigenvalue decomposition of covariance matrix
    [U, S] = eig(covmattr);
    fprintf(1,'Running total best alpha on whole data set\n');
    fullModelAlpha = nanmean(strf.totalBestAlpha);
    % ridge D
    D = diag(1./(diag(S)+fullModelAlpha));
    % betas
    wt = U*D*U'*(X'*Y);
    % predicted response
    predY = X*wt;
    % correlation between predicted resp and data
    ccor = nan(nChan,1);
    for cchan = 1:nChan
        ccor(cchan) = corr(predY(:,cchan), Y(:,cchan));
    end
    
    strf.fullMod.alpha = fullModelAlpha;
    strf.fullMod.fullCorr = ccor;
    
    strf.fullMod.strf = reshape(wt,nfeat,[], nChan);
end

strf.dataf = dataf;
strf.timeLag = time_lag;
strf.datasize = size(X);
strf.NTrials =  length(unique(TrialInd));

% delete strf_temp.mat

end
%% auxiliary functions
function [m3] = mat_diag_prod(m1, m2)
% calculate product of matrix m1 with diagonal matrix m2 m3 = m1*m2

m3= zeros(size(m1,1), size(m2,2));
for i = 1:size(m1, 2)
    m3(:,i) = m1(:,i)*m2(i,i);
end

end
