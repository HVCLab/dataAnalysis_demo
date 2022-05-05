%% plot defitions
onsetColor =  [85 184, 72]/256;
rampColor = [180, 71, 143]/256;
hgColor = [232 167 105]/256; % yellow
rates = {'reg', 'slow'};
ratefreqs = [5.7 0 1.9];
conditionNames = rates;
condColors = [106 189 69; 46 53 143]/256;
nColors=100;
rbcm = [[repmat(linspace(0,1, nColors)',1,2),ones(nColors, 1)] ; [ones(nColors, 1), repmat(linspace(1,0, nColors)',1,2)]];
