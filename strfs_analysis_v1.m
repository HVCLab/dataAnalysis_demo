%% code for calculating and analysing TRFs on MEG data
%% ---- calculate using strf_by_subj_TSFformat.m
%% path defitions
addpath(genpath('util/'));
datapath = '..';
megdatapath = fullfile(datapath, 'data');
% output path
outStrfFolder = fullfile(megdatapath, 'strf_v5_reg_filtdata');
%% plot defitions
run plot_def.m
%% ---- load all BURSC stimulus details
stimfile = 'sentenceDetails.mat';
load(stimfile);



%% ---- load TRFs
ncs=1;
strfmod = {'SentOns_peakRate'};
strfparamname = '_zX1_zY0_scX1_scY0_hp0_SentOns1_sentScale0_El1to20_past300ms_fut100ms_edge1_boot1.mat';
rates = {'reg'};%, 'slow'};
for cs = 1:ncs
    disp(cs)
    for cm = 1:length(strfmod)
        for cr = 1:length(rates)
            cmname = ['cs' num2str(cs) '_strf_' strfmod{cm} strfparamname];
            strfall.(rates{cr})(cs,cm) = load(fullfile(outStrfFolder, cmname),...
                'meanTestR', 'meanStrf', 'name', 'featureNames'); % %testY, predY
        end
    end
end
dataf=400;
time_lags =[0.1 0.3]*dataf;
%% combine Rsq across models 
allrsq.reg=[];
allrsq.slow=[];
for cm = 1:length(strfmod)
    for cr = 1:length(rates)
        allrsq.(rates{cr})(:,:,cm)=cell2mat({strfall.(rates{cr})(:,cm).meanTestR}').^2;
    end
end
%% compare R2 for trfs - -only after fitting multiple models.
pl=[1 2]; % which models to compare 
figure
for cr = 1:length(rates)
    subplot(2,1,cr)
    cla; hold on;
    for ch = 1:2, scatter(allrsq.(rates{cr})(:,ch,pl(1)), allrsq.(rates{cr})(:,ch,pl(2)));end
    title(rates{cr});
    legend('lh', 'rh');
    refline(1,0);
    xlabel(strfmod{pl(1)});
    ylabel(strfmod{pl(2)});
end
%% plot trfs
figure
prTrf=[];
for cri = 1:length(rates)
    cr = rates{cri};
    for i = 1:length(strfmod)
        alltrf = permute({strfall.(cr)(:,i).meanStrf}, [1 3 4 2]);
        alltrf = cell2mat(alltrf);
        prTrf(cri,:) = mean(mean(alltrf(2,:,:,:), 4), 3)';
        subplot(2,length(strfmod),i + length(strfmod)*(cri-1))
        cla; hold on;
        plot((1:size(alltrf,2))/400-.300, mean(mean(alltrf(2,:,:,:), 4), 3)', 'linewidth' ,2)
        legend(strfall.(cr)(1,i).featureNames);
        title(cr)
%         xlim([-.1 .3])
        ylim([-0.8 0.8])
        horzline(0);
        vertline(0);
        grid on;
    end
end

%% plot peakRate trf only
figure
cla; hold on;
for cri = 1:length(rates)
    cr = rates{cri};
    alltrf = permute({strfall.(cr).meanStrf}, [1 3 4 2]);
    alltrf = cell2mat(alltrf);
    %         subplot(2,length(strfmod),i + length(strfmod)*(cri-1))

    plot((1:size(alltrf,2))/400-.25, mean(mean(alltrf(2,:,:,:), 4), 3)', 'linewidth' ,2)
end

xlim([-.25 .75])
ylim([-0.8 0.8])
horzline(0);
vertline(0);
grid on;
legend(rates);

%% plot average response to peakRate and trf for each subject in regular speech
figure;
for i =1:12
    subplot(3,4,i)
    cla; hold on;    
    plot((1:281)/400 - .2,ecog_norm(mean(alldata(i).aveRespRate,1)),'linewidth', 2);
    plot((1:size(strfall.reg(i,2).meanStrf,2))/400-.1,ecog_norm(squeeze(mean(strfall.reg(i,2).meanStrf(2,:,:),3))), 'linewidth', 2, 'linestyle', ':');
    vertline(0);
    horzline(0);
end
legend('evResp','strf');

%% plot average response to peakRate and trf for each subject in slow speech
figure;
for i =1:12
    subplot(3,4,i)
    cla; hold on;    
    plot((1:281)/400 - .2,ecog_norm(mean(alldata(i).aveRespRateSl,1)),'linewidth', 2);
    plot((1:size(strfall.reg(i,2).meanStrf,2))/400-.1,ecog_norm(squeeze(mean(strfall.slow(i,2).meanStrf(2,:,:),3))), 'linewidth', 2, 'linestyle', ':');
    vertline(0);
    horzline(0);
end
legend('evResp','strf');

%% plot predicted responses for single sentences using the trf model  - - using cstrf.predY
cm=2;
figure
cla; hold on;
plot(ecog_norm(strfall.slow(1,cm).testY{1}(:,1)));
plot(ecog_norm(strfall.slow(1,cm).predY{1}(:,1)));
xlim([0 1600]);


%% calculate residuals from model
for cr = 1:2
    for cs = 1:12
        for cm = 1:length(strfmod)
            for cf = 1:length(strfall.(rates{cr})(cs,cm).testY)
                strfall.(rates{cr})(cs,cm).Yres{cf} = ...
                    ecog_norm(strfall.(rates{cr})(cs,cm).testY{cf},2) - ecog_norm(strfall.(rates{cr})(cs,cm).predY{cf},2);
                %             strfall.reg(cs,cm).Yres{cf} = strfall.reg(cs,cm).testY{cf} - strfall.reg(cs,cm).predY{cf};
            end
        end
    end
end

%% plot average response in residuals
cs = 1; cm = 2;
figure
cla; hold
plot((strfall.reg(cs,cm).predY{1}(:,1)));
plot((strfall.reg(cs,cm).testY{1}(:,1)));
plot((strfall.reg(cs,cm).Yres{1}(:,1)));
legend('pred', 'data', 'res');

%% average peakRate response in test data
respf = {'testY', 'predY', 'Yres'};
for cr = 1:2
    for cs =1:12
        for cm  = 1:2
            % within strf
            timewind = [-.2*fs: .5*fs];
            for cf = 1:13
                b = strfall.(rates{cr})(cs,cm).testStim{cf}(:,2)';
                for cfn = 1:3
                    strfall.(rates{cr})(cs,cm).RespRate.testY{cf} = ...
                        ecog_segment(strfall.(rates{cr})(cs,cm).(respf{cfn}){cf}', sign(b), [],timewind, 0);
                end
            end
            %         figure,
            %         cla; hold on;
            %         for cf = 1:3
            %             cresp = cell2mat(permute(strfall.reg(cs,cm).RespRate.(respf{cf}), [1 3 2]));
            %             plot(timewind/400, mean(cresp(1,:,:),3)');
            %         end
            %         vertline(0);
            %         legend(respf);
        end
    end
end
