%% Feature temporal receptive fields for data in time series format
% this is a wrapper for calculating F-TRFs on data that are organized in a
% time series format with 2 separate timeseries:
% 1) data  - dependent variable el x time
% 2) predictors: feature x time
% 3) predictor names - for parsing model name

%% ridge regression or OLS
ridgeflag =1;
useRate = 1; % slow or regular speech
run plot_def.m;
debug =1;
%% - - - - setup path and load prepared data
addpath(genpath('util/'));
datapath = '..';
megdatapath = fullfile(datapath, 'data');
% output path
outStrfFolder = fullfile(megdatapath, 'strf_v5_reg_filtdata');
  
%% define models
scaleflag = 0; % scale predictors within each sentence.
nFoldsRun = 13; %leave one out with 13 paragraphs.

modelfields = {'SentOns_peakRate'};%,'SentOns_peakRateBins', 'SentOns'};%,  'paraOns', 'paraOns_peakRate', 'SentOns',
binaryModelFields = zeros(1,length(modelfields));
%% load data
ncs = 1;
for i = 1:ncs
    alldata(i) = load(fullfile(megdatapath, sprintf('dataAll_ts_cs%d_broadband.mat', i)));
end
%% prepare data - subsetting etc
for i = 1:ncs
    %% subset data to one speech rate speech
    incldata = alldata(i).pred(strcmpi(alldata(i).predNames, 'speech rate'),:) == useRate & alldata(i).trials~=0;

    % add binary peakRate predictor
    alldata(i).pred(9,:) = sign(alldata(i).pred(4,:));
    alldata(i).predNames{9} = 'peakRateOns';

    % bin peakRate magnitude into 5 bins
    prval = alldata(i).pred(4, incldata);
    prval(prval ==0) = [];
    qs = quantile(prval, 4);
    [~,~,binidx] = histcounts(prval, [0 qs]);

    prbins = zeros(length(alldata(i).pred),1);
    prbins(incldata & alldata(i).pred(4, :)>0) = binidx+1;
    alldata(i).pred(10:14,:) = zeros(5, length(alldata(i).pred));
    for cb = 1:5
        alldata(i).pred(10+cb-1,prbins==cb) = 1;
        alldata(i).predNames{10+cb-1} = ['peakRateBins' num2str(cb)];
    end
    % remove timepoints not included in this model (here - the other speech
    % rate)
    alldata(i).pred(:,~incldata)=[];
    alldata(i).data(:,~incldata)=[];
    alldata(i).dataRaw(:,~incldata)=[];
    alldata(i).paraNum(~incldata)=[];
    alldata(i).trials(~incldata)=[];
    alldata(i).paraend(~incldata)=[];
    alldata(i).tp(~incldata)=[];
end


%% STRF model fitting
disp('%% --------------------- strf model fitting --------------------- %%');
for cs = 1:length(alldata)
    for cmf = 1:length(modelfields)
        cdata = alldata(cs);
        cdata.fs = 400;
        
        %% mark test stim set and create split in nfolds folds
        nTrials = length(unique(alldata(cs).trials));
        trialNums = 1:nTrials;
        testfoldInd = logical(kron(eye(nFoldsRun), ones(floor(nTrials/nFoldsRun),1)));
        addrows = nTrials - size(testfoldInd,1);
        testfoldInd(end+ (1:addrows),:)=0;
        testfoldInd(end-addrows+1:end,1:addrows) = eye(addrows);

        testfoldInd = testfoldInd(randperm(nTrials),:);
        trainfoldInd = logical(ones(size(testfoldInd)) - testfoldInd);
        foldInd = struct('test', testfoldInd, 'train', trainfoldInd);

        % plot folds
        if debug
            figure
            subplot(1,2,1), imagesc(foldInd.train), colorbar;
            subplot(1,2,2), imagesc(foldInd.test), colorbar;
        end
        %% create predictor matrix
        modelname = modelfields{cmf};
        cdata.pred = make_pred_mat(cdata.pred, cdata.predNames, modelname);

        %% run model
        curbinModF=binaryModelFields(cmf);
        curmod = modelfields{cmf};
        disp('starting strf function')
        tic
        strf_main_v2_TSFormat(cdata, curmod, cs, curbinModF, outStrfFolder, scaleflag, nFoldsRun, foldInd, 1:size(alldata(1).data,1), 'data');
        toc
    end
end


%% ------------------ strf create predictor matrix

function [predTS2] = make_pred_mat(predTS, predNames, modelname)
modelPN= textscan(modelname, '%s', 'Delimiter', '_');
modelPN = modelPN{1};

for cpr = 1:length(modelPN)
    switch modelPN{cpr}
        case {'paraOns', 'SentOns'}
            preds{cpr,1} = sign(predTS(strcmpi(modelPN(cpr), predNames),:));

        case {'peakRateOns', 'peakEnvOns'}
            preds{cpr,1} = sign(predTS(strcmpi(modelPN{cpr}(1:end-3), predNames),:));
        case  {'peakRateBins'}
            preds{cpr,1} = predTS(10:end,:);
        otherwise
            if ismember(modelPN(cpr), predNames)
                preds{cpr,1} = predTS(strcmpi(modelPN(cpr), predNames),:);
            else
                error('unknown predictor name');
            end
    end
end
predTS2 = cell2mat(preds);
end

%% ----------------------- strf main function
function [strf] = strf_main_v2_TSFormat(cdata, modelname, cs, binaryModelFields, ...
    strfSaveFolder, scaleflag, nFoldsRun, foldInd, STRFEl, respField)
%% inputs

% binaryModelFields turns all predictors in model into binary 1/0
saveSmallFlag = 1; % save model in small file format, without saving data within strfs - recommended. save data with strf for debugging purposes mainly
if nargin <4
    binaryModelFields = zeros(size(modelname));
end

if nargin < 10 || isempty(respField)
    respField = 'resp';
end
if nargin < 5
    strfSaveFolder=fullfile('out_strf', cs);
end
% within-sentence predictor scaling
if nargin < 6
    scaleflag = zeros(size(modelname)) ;
end
% full cross-validation?  set to 5 for full cross-validation
if nargin <7, nFoldsRun = 5; end

%% estimation parameters

params.onsetflag = 0;
params.inclSentons = 1;
params.zscoreXflag = 1;
params.zscoreYflag =  0;
params.scaleXflag = 1;
params.scaleYflag = 0;
params.highpassflag = 0;
params.sentScale = scaleflag;
cmodname = modelname;

modelnames = sprintf('%s_zX%d_zY%d_scX%d_scY%d_hp%d_SentOns%d_sentScale%d', cmodname,params.zscoreXflag,...
    params.zscoreYflag,params.scaleXflag,params.scaleYflag, params.highpassflag,params.inclSentons,scaleflag);
%% all possible combinations of parameters
% for zx = zscoreXflag
%     for zy = zscoreYflag
%         for sx = scaleXflag
%             for sy = scaleYflag
%                 for hp = highpassflag
%                     %parameters to make stimuli
%                     params(i).onsetflag = 0;
%                     params(i).inclSentons = 1;
%                     params(i).zscoreXflag = zx;%zscoreXflag(i);
%                     params(i).zscoreYflag =  zy;%zscoreYflag(i);
%                     params(i).scaleXflag = sx;%scaleXflag(i);
%                     params(i).scaleYflag = sy;%scaleYflag(i);
%                     params(i).highpassflag = hp;%highpassflag(i);
%                     modelnames{i} = sprintf('%s_onset%d_zX%d_zY%d_scX%d_scY%d_hp%d_SentOns%d_sentScale%d', modelfields{i},params(i).onsetflag,params(i).zscoreXflag,...
%                         params(i).zscoreYflag,params(i).scaleXflag,params(i).scaleYflag, params(i).highpassflag,params(i).inclSentons,scaleflag);
%                     i=i+1;
%                 end
%             end
%         end
%     end
% end
clear i;
%% STRF setup - do not change these parameters
time_lag = [.3 .1]; % past stimulus, future stimulus
% time_lag = [0.75 0.25]; % past stimulus, future stimulus
regalphas = ([logspace(0 ,7, 20)]); %

nfold = 5; % folds
fullmodel_flag = 0; % save model on entire data - really no reason to do that unless to test that model with folds is stable - but that can also be done by comparing feature betas between folds
bootflag = 1; % bootstrap model alpha on training set - keep set to 1
nboots = 10; % number of bootstraps
edgeflag = 1; % keep edges , i.e. transitions between trials. only remove those if many short trials.
dataf = cdata.fs; % frequency of data.
%% run strfs with 80-10-10 division
fprintf(2, '\n ........................ %s: Dataset %d ........................ \n', modelname, cs);
%% get stimulus matrix
shortModelName = modelname;
%% remove empty predictors
mf = 'pred';
a=cdata.(mf);
b=sum(a>0,2);
rmPred = (find(b<10));
cdata.(mf)(rmPred,[]);
clear a b;
%% count predictors
npred = size(cdata.pred, 1);
%% create time-lagged stimuli
[X, Y, trialInd, nfeat] = strf_makeXtimeLag_TSFormat(cdata.(respField), cdata.(mf), cdata.trials, time_lag, dataf, params);
%% binarize model
if binaryModelFields
    X(X>0)=1;
    X(X<0)=-1;
end

%% run strfs
if bootflag
    strf = strf_main_bootstrap_ridge(X, Y, time_lag, dataf ,regalphas, nfold, fullmodel_flag, nboots,trialInd, foldInd, nFoldsRun);
else
    strf = strf_main(X, Y, time_lag, dataf ,regalphas, nfold, fullmodel_flag, nFoldsRun,trialInd);
end
%% add fitting info to strf
strf.Els = STRFEl;
strf.cs = cs;
strf.nfeat = nfeat;
strf.meanTestR = nanmean(strf.testCorrBestAlpha,1);
strf.name = modelnames; %shortnames{cfield};
strf.shortname = shortModelName;
strf.fitParam = params;

%% removed predictors - those that didn't have values ~=0
strf.rmPred = rmPred;

%% add trial indices to strf
for cfold = 1:nFoldsRun
    % trials in this fold:
    curTrials = ismember(trialInd, find(foldInd.test(:,cfold)==1));
    % indexed within data set
    strf.trialIndtest{cfold}= trialInd(curTrials);
    % indexed rel. to all sentences
    strf.sentIndtest{cfold} = [];
end

strf.repSent =[];
strf.repSentName = [];

%% add list of timepoints that were used to fit the model
strf.dataInfo.tp = cdata.tp;
strf.dataInfo.trials = cdata.trials;
%% list of predictors
cpred = textscan(strf.shortname, '%s', 'Delimiter', '_');
strf.featureNames = cpred{1};

%% BIC for model comparisons

bic = @(n,k, sigmaErrSq) n*log(sigmaErrSq)+k*log(n);  % n - # observations, k - # model parameters; sigmaerrSq - error variance
for cfold = 1:length(strf.testY)
    strf.sigmaErrSq(:,cfold) = mean((strf.testY{cfold}-strf.predY{cfold}).^2, 1);
    n = size(strf.testY{cfold}, 1);
    k = size(strf.testStim{cfold},2);
    strf.bic{cfold} = bic(n, k, strf.sigmaErrSq(:,cfold));
    % adjusted bic for model with sparse predictors
    kAdj = max(sum(strf.testStim{cfold}~=0,2));
    if kAdj < k
        kAdj = kAdj*2;
    end
    strf.bicAdj(:,cfold) = bic(n, kAdj, strf.sigmaErrSq(:,cfold));
    strf.kAdj(cfold) = kAdj;
end
%% predictor sparseness
for cp = 1:strf.nfeat
    strf.predsparse(cp) = sum(sum(strf.testStim{1}(:,cp:strf.nfeat:end)~=0))/numel(strf.testStim{1}(:,cp:strf.nfeat:end));
end
%% coefficient of determination
[strf] = strf_coeffDet(strf);

%% by sentence correlations - need to change for time series format
%     strf.sentname = {out.name};
%     edges = [20 20;60 20; 0 0];
%     for cedge = 1:size(edges,1)
%         [strf] = strf_add_sentenceCorr(strf, edges(cedge,1), edges(cedge,2));
%     end

%% remove huge fields in strf that can be recreated/are not used
% this is not necessary if proper trial information is saved in data.
if saveSmallFlag
    strf = rmfield(strf, {'testStim', 'testY', 'predY'});
end

%% save strfs to file
switch length(time_lag)
    case 1
        time_lag_txt = sprintf('past%dms',round(time_lag*1000));

    case 2
        time_lag_txt = sprintf('past%dms_fut%dms',round(time_lag(1)*1000),round(time_lag(2)*1000));
end

switch binaryModelFields
    case 0
        strffilename =  sprintf('cs%d_strf_%s_El%dto%d_%s_edge%d_boot%d.mat',...
            cs,modelnames, strf.Els(1), strf.Els(end),time_lag_txt,edgeflag,bootflag);
    otherwise
        strffilename =  sprintf('%s_strfbin_%s_El%dto%d_%dms_edge%d_boot%d.mat',...
            cs,modelnames, strf.Els(1), strf.Els(end),round(time_lag*1000),edgeflag,bootflag);
        strf.shortname = [strf.shortname '_allbin'];
end

strffolder = strfSaveFolder;%fullfile(strfSaveFolder, sprintf('strf_v%d', sentdetVn));

if ~exist(strffolder, 'dir') , mkdir(strffolder), end
strfOutFile = fullfile(strffolder, strffilename);
save(strfOutFile, '-struct', 'strf', '-v7.3')
fprintf(2,'saved %s to %s. \n', strf.shortname,strfOutFile);
end

%% ----------------------- create time-delayed predictor matrix function
function [X,Y, trialInd, nfeat] = strf_makeXtimeLag_TSFormat(Y,X, trials, time_lag, dataf, parameters)
% function [X,Y, trialInd, nfeat] = strf_makeXtimeLag_TSFormat(Y,X, trials, time_lag, dataf, parameters)
% brings timit/dimex data in a format to calculate strfs using the strf_main
% function based on stimulus features (e.g. spectrograms);
% Inputs:
% Y - dependent var, electrodes x time
% X - predictors - features x time
% trials - trialnumbres, 1 x time
% time_lag - time window from 0 - -time_lag is included in
% dataf - data frequency
% parameters - for mdoel fitting

% Yulia Oganian, Aug 2019
%% inputs
nfeat = size(X, 1);
if nargin < 4, time_lag = .3; end % in sec
if nargin < 2, warning('Need stimulus');return;end

if nargin <6
    parameters.inclSentons = 1;
    parameters.onsetflag = 1;
    parameters.zscoreXflag = 1;
    parameters.zscoreYflag = 1;
    parameters.scaleXflag = 1;
    parameters.scaleYflag = 0;
    parameters.highpassflag =0;
else
    if ~isfield(parameters, 'inclSentons'), parameters.inclSentons = 1; end
    if ~isfield(parameters, 'onsetflag'), parameters.onsetflag = 1; end
    if ~isfield(parameters, 'zscoreXflag'), parameters.zscoreXflag = 1; end
    if ~isfield(parameters, 'zscoreYflag'), parameters.zscoreYflag = 1; end
    if ~isfield(parameters, 'scaleXflag'), parameters.scaleXflag = 1; end
    if ~isfield(parameters, 'scaleYflag'), parameters.scaleYflag = 1; end
    if ~isfield(parameters, 'highpassflag'), parameters.highpassflag = 1; end
end

%% high pass filter the data before running strf if requested
if parameters.highpassflag
    fc=1;
    [b,a] = butter(3, fc/(dataf/2), 'low');
    dataOld = Y;
    trialU = unique(trials);
    for ctr = 1:length(trialU)
        for cel = 1:size(dataOld,1)
            Y(cel,trials == trialU(ctr))  = dataOld(cel,trials == trialU(ctr)) - filtfilt(b,a, dataOld(cel,trials == trialU(ctr)));
        end
    end
end
%%
%% identify binary predictors
binpred = nan(size(X,1),1);
for i =1:size(X,1)
    cval = unique(X(i,:));
    if length(cval)<3
        binpred(i)=1;
    end
end

%% zscore data if requested
% zscore continuous predictors
if parameters.zscoreXflag
    X(isnan(binpred),:) = zscore(X(isnan(binpred),:),0,2);
end

if parameters.zscoreYflag
    Y = zscore(Y,0,2);
end
Y = Y';
%% scale data if requested
if parameters.scaleYflag
    for i = 1:size(Y, 2), Y(:,i) = Y(:,i)/max(abs(Y(:,i)));end
end

if parameters.scaleXflag
    for i = 1:size(X, 1)
        X(i,:) = X(i,:)/max(abs(X(i,:)));
    end
end

trialInd = trials;
nt = size(Y,1);
%% create X with delays
% tic
dstim=cell(2,1);
try
    for i = 1:length(time_lag)
        delaytpn = time_lag(i)*dataf;
        dstim{i,1} = zeros(nfeat*delaytpn, nt);
        switch i
            case 2 % (future stim)
                for cdelay = 1:delaytpn
                    dstim{i,1}((nfeat*(delaytpn-cdelay)+1):nfeat*(delaytpn-cdelay+1),:) = [X(:,(1+cdelay):end), zeros(nfeat, cdelay)];
                end
            case 1 % (past stim)
                for cdelay = 1:delaytpn
                    dstim{i,1}((nfeat*(cdelay-1)+1):nfeat*cdelay,:) = [zeros(nfeat, cdelay-1), X(:,1:(end-cdelay+1))];
                end
        end

    end

    dstim = [dstim{2}; dstim{1}];

catch ME
    disp('problem while making lagged stimulus');
    rethrow(ME);
end
% old version with only past stimulus


% delaytp = 1:time_lag*dataf;
% delaytpn = length(delaytp);
% dstim = zeros(nfeat*delaytpn, nt);
% for cdelay = 1:delaytpn
%     try
%         dstim((nfeat*(cdelay-1)+1):nfeat*cdelay,:) = [zeros(nfeat, cdelay-1), X(:,1:(end-cdelay+1))];
%     catch ME
%         disp('problem while making lagged stimulus');
%         rethrow(ME);
%     end
% end
% toc

X = dstim';
end

%% -----------------------  sentence correlation function

function [cstrf] = strf_add_sentenceCorr(cstrf, onEdge, offEdge)

if nargin < 2
    onEdge = 0; % 20 or 60 used before
end
if nargin < 3
    offEdge=0; % 20 used before
end
edgeFname = sprintf('sentR_ons%d_off%d', onEdge, offEdge);

%% strf sentence by sentence correlations
cstrf.(edgeFname) = cell(size(cstrf.trialIndtest));
for cfold = 1:length(cstrf.trialIndtest)
    ctrialind = cstrf.trialIndtest{cfold};
    trialu = unique(ctrialind);
    cstrf.(edgeFname){cfold} = nan(length(cstrf.Els), length(trialu));

    %% Y
    testY = cstrf.testY{cfold};
    predY = cstrf.predY{cfold};

    %% calculate correlations
    for csent = 1:length(trialu)
        cind = find(ctrialind==trialu(csent));
        cindNoEdge = cind((onEdge+1):end-offEdge);
        cstrf.(edgeFname){cfold}(:,csent) = ...
            diag(corr(testY(cindNoEdge,:),predY(cindNoEdge,:)));
    end
end

end