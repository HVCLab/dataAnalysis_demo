% intial setup: make sure that this folder and the data folder are
% contained in a shared folder one level up
clear all; clc;
set(0,'DefaultFigureWindowStyle','docked')
%% load prepared data
addpath(genpath('util/'));
datapath = '..';
megdatapath = fullfile(datapath, 'data');
ncs = 12; % 12 subjects in data set
run plot_def.m;

for i = 1:ncs
    alldata(i) = load(fullfile(megdatapath,sprintf('dataAll_ts_cs%d_broadband.mat', i)));
end
datafs = alldata(1).datafs;

%% plot data time series for a single subject
figure
subplot(2,1,1)
plotTP = 1:alldata(1).datafs*10;
plot(plotTP/alldata(1).datafs, alldata(1).dataRaw(:,plotTP));
xlabel('time');
ylabel('response');
title('single sensors for single subjects')

subplot(2,1,2)
plot(plotTP/alldata(1).datafs,alldata(1).pred(:,plotTP))
legend(alldata(1).predNames);
set(gca, 'YDir', 'normal')


%% segmentation of data around events
useFeatRow = [4 3] ; 
useFeatName = {'pRate', 'pEnv'};
segmentWindow = -200:200; % in samples
for cs = 1:ncs
    disp(cs);
    for cfeat = 1:2
        %% get ERP around  events
        % event selection
        cev = sign(alldata(cs).pred(useFeatRow(cfeat),:))>0;      
        j=1;
        for crate = [1 3]
            cevr = cev & alldata(cs).pred(7,:)==crate;          
            for i = 1:2
                % raw data
                curSegment = ecog_segment(alldata(cs).dataRaw, cevr,[], segmentWindow,0);
                meanresp.([useFeatName{cfeat} '_raw'])(:,j, cs) = squeeze(mean(mean(curSegment, 3),1));
            end
            j=j+1;
        end
    end
end
%% time axis for plotting
meanresp.timeax = (1:size(meanresp.pEnv_raw,1))/datafs+min(segmentWindow)/datafs;

%% note: there is no baseline correction here, instead we're hoping that the pre-event period will average out across the many event repetitions

% we will look at the effects of baselining at a later stage
%% plot average evoked responses to peakRate and peakEnv
figure
cla; hold on;

cpl(1)=shadedErrorBar(meanresp.timeax, mean(meanresp.pRate_raw(:,1,:), 3),nansem(meanresp.pRate_raw(:,1,:), 3), {'color', 'r'});
cpl(2)=shadedErrorBar(meanresp.timeax, mean(meanresp.pEnv_raw(:,1,:), 3),nansem(meanresp.pEnv_raw(:,1,:), 3), {'color', 'k'});

horzline(0); vertline(0);title('time-domain group average')
legend([cpl.mainLine], {'peakRate', 'peakEnv'})
xlim([min(segmentWindow)/datafs, max(segmentWindow)/datafs])
xlabel('time (s)')
ylabel('neural response a.u.')

%% plot single subjects' erps
figure;
for cs = 1:ncs
    splax(cs) = subplot(3,4,cs);
    hold on;
    plot(meanresp.timeax,meanresp.pRate_raw(:,1,cs));
    plot(meanresp.timeax,meanresp.pEnv_raw(:,1,cs));
end
linkaxes([splax]);

for cs = 1:ncs
    subplot(3,4,cs)
    vertline(0); horzline(0);
end
legend('peakRate', 'peakEnv');

%% - - - - questions - - - - - 
% 1 - how does group ERP change with inclusion of more participants? Plot
% average of 1, 2, .. , 12 participants

%% --- statistics for comparison between peakEnv and peakRate

%% time point by time point t-test

for i = 1:size(meanresp.pRate_raw, 1)
   [~,~,~,tstat(i)] = ttest(squeeze(meanresp.pRate_raw(i,1,:)),...
        squeeze(meanresp.pEnv_raw(i,1,:)));
end
figure, 
hold on; 
plot(meanresp.timeax,abs([tstat.tstat]))
signtp = meanresp.timeax(find(abs([tstat.tstat])> tinv(0.975, tstat(1).df)));
horzline(tinv(0.975, tstat(1).df),[],'r','-')
vertline(0); horzline(0);

legend('tstat', 'sign treshold')



%% cluster-based permutation  
ptres = 0.05;
differenceTrials = [squeeze(meanresp.pRate_raw(:,1,:))-squeeze(meanresp.pEnv_raw(:,1,:))];
[clusters, p_values, t_sums, distribution ] = permutest2tail(differenceTrials, ptres, ...
    100, 2);
 % plot clusters
 
figure, 
hold on; 
plot(meanresp.timeax,abs([tstat.tstat]))
signtp = meanresp.timeax(cell2mat(clusters'));
scatter(signtp, -1*ones(size(signtp)));
horzline(tinv(0.975, tstat(1).df),[],'r','-')
vertline(0); horzline(0);

% plot cluster statistic distributions
figure
for i = 1:length(clusters)
    subplot(1,length(clusters), i)
    hold on 
    histogram(distribution{i})
    vertline(t_sums(i));
end

% - - - - Questions
% 1 - try out different p tresholds
% 2 - try out different numbers of permutaitons


 %% peak detection per subject and condition
figure
for cs = 1:ncs
    subplot(4,4, cs), hold on
    [peaks{cs}, peakloc{cs}]=...
        findpeaks(meanresp.pRate_raw(:,1,cs), meanresp.timeax, 'SortStr','descend');
    plot(meanresp.timeax, meanresp.pRate_raw(:,1,cs));
    scatter(peakloc{cs}(1:2), peaks{cs}(1:2))
% this is pretty noisy... try to limit the number of peaks, peak
% prominence, minimal distance between peaks, to find the larges peak. 
end

%% tasks and questions
% 1 - - how do you find the negative peaks?
% 2 - - extract the following for each subject and peakRate and peakEnv
% conditions separately and run t-tests on them. does it change the
% results? 
% a) absolute response amplitude
% b) trough-to - peak response amplitude 
