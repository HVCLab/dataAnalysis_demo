function [aligndat, L, trialIds] = ecog_segment(data, eventTimes,trialId, segmentWin, meanFlag)
%% function [aligndat, L, trialIds] = ecog_segment(data, eventTimes,trialId, segmentWin, meanFlag)
% creates segmented data, realigned to different types of events, each type
% of event is indicated by a different row in eventTimes. 
% inputs: 
%   data: electrodes x time
%   eventTimes: eventtype x time, binary 0 or 1 for event onsets
%   trialId: if data is concatenated across trials/sentences (as in TIMIT),
%   1 x time, contains trial numbers. 
%   segmentWin - window around event onsets to chooose in sample numbers,
%   e.g. -20:49.
%   meanFlag: output single event occurences (0) or averages for each event
%   (1, default, recommended with large data)
% outputs: 
%   aligndat: segmented data electrodes x event occurences x time
%   L:  events x 1 event types for data in aligndat, values of 1 to numbers
%   of rows in eventTimes
%   trialIds: id of trial from which a segment was taken, based on input
%   trialId
% Yulia Oganian, 15 February 2019

%%
if nargin<3 || isempty(trialId)
    trialId = 1:size(data,2);
end

if nargin<4  || isempty(segmentWin)
    segmentWin = -20:49;
end
 
if nargin< 5 || isempty(meanFlag)
    meanFlag = 1;
end
trialId = trialId(:);
ntp = length(segmentWin);

if size(data,2) ~= size(eventTimes,2)
    error('Size of data and eventTimes do not match');
    return;
end
%% 0-padding
data = [zeros(size(data, 1), length(segmentWin)), data];
eventTimes = [zeros(size(eventTimes, 1), length(segmentWin)), eventTimes];
trialId = [zeros(length(segmentWin),1); trialId];

%% different formants of eventTimes

if size(eventTimes,1) == 1 % check if different events are in eventTimes
    ucond = unique(eventTimes(eventTimes>0));
    if length(ucond)>1
        eventTimes_old = eventTimes;
        eventTimes =zeros(size(eventTimes));
        for i = 1:length(ucond)
            eventTimes(i, eventTimes_old==ucond(i))=1;
        end
    end
else
    eventTimes = eventTimes~=0; % make sure it's binary 
end

%%
if meanFlag
    %% get mean for each eventtype
    
    aligndat=nan(size(data,1), ntp, size(eventTimes,1));
    for j  = 1:size(eventTimes,1)
%         tic
        cloc = find(eventTimes(j,:)==1);
        inclind = cloc(:) + segmentWin;
        
        exclTr = find(inclind(:,end)>size(data,2));
        inclind(exclTr,:)=[];        
        inclind = inclind(:);
        
        cdat2 = data(:,inclind);
        cdat2 = squeeze(nanmean(reshape(cdat2,size(cdat2,1),[],ntp), 2));
        aligndat(:,:,j) = cdat2;        
    end
    % toc
else
    %% get single occurences of each eventtype
%     tic
    [L,fc] = find(eventTimes==1);
    inclind = fc(:) + segmentWin;
    exclTr = find(inclind(:,end)>size(data,2));
    inclind(exclTr,:)=[];
    L(exclTr) = [];
    cdat = data(:,inclind(:));
    aligndat = reshape(cdat,size(cdat,1), int64(size(cdat,2)/ntp),ntp);
    aligndat = permute(aligndat, [1 3 2]);
    trialIds = trialId(fc)';
%     toc
end

