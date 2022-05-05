function [clusters, p_values, t_sums, distribution ] = permutest2tail(differenceTrials, pThreshold, ...
    nPermutations, nClusters)
% Permutation test for dependent measures for 1-D or 2-D data. Based on 
% Maris & Oostenveld 2007 for 1-D and 2-D vectors. The test statistic is 
% T-Sum - the total of t-values within a cluster of contingent
% above-threshold data points. 
% 
% 
% Inputs: 
% differenceTrials - a 2-D or 3-D matrix, holding the difference at each
% trial between two conditions (dependent samples). The last dimension has
% to be the trials. 
% pThreshold - the p-value threshold below which a data point is part of an
% active cluster. Lower values mean that the test is sensitive to
% narrower, stronger effects. The p-value is translated to a t-value for
% the purpose of this test. 
% nPermutations - number of desired permutations. The lowest possible
% p-value will be 1 / (nPermutations+1). 
% nClusters (optional) - the number of concurrent clusters to be checked 
% for significance. The second-largest cluster will be compared against the
% permutation distribution of second-largest clusters, etc. Default: 3
% 
% Outputs:
% clusters - a cell array with each cell corresponding to a cluster (up to
% nClusters), holding the vector/matrix indexes corresponding to the 
% current cluster. 
% p_values - the permutation p-value corresponding to each cluster. 
% t_sums - the sum of t-values of all data points comprising each cluster
% distributions - a cell array holding the T-Sum permutation distributions
% for each cluster up to nClusters. 
% 
% Written by Edden M. Gerber, lab of Leon Y. Deouell, 2014
% Send bug reports and requests to edden.gerber@gmail.com


% Change Log: 
% 
% 11.12.2014 Edden: Now using the bwconncomp function from the image
% processing toolbox if it is supported, instead of GetClusters. 
% 11.12.2014 Edden: Fixed a serious bug - instead of the Tsum, the sum 
% which was computed before was just the number of pixels in the cluster. 
% 11.12.2014 Edden: Output variable clusters now holds the actual indexes 
% from the original vector/matrix belonging to of each cluster, not a 
% logical map. 
% 22.12.2015 Edden: corrected p-value calculation to treat null 
% distribution values equal to the primary value as "equal or higher". 
% 8.8.2016 Edden: changed input argument from t-threshold to p-threshold;
% the t-threshold is computed using tinv from the p-threshold and the
% degrees of freedom. 
% 28.9.2016 Edden: Fixed a critical bug caused by the previous change - the
% computed t-value threshold was negative, leading to very large
% above-threshold clusters being found. 

% 2/1/19: Added 2 tailed ttest. Threshold is adjusted. KK


%% INITIALIZE
% Set optional arguments:
%if nargin < 5
%    nClusters = 3;
%end

%% Check if clustering function is supported:
if exist('bwconncomp','file')% Use this function if it is supported (image processing toolbox)
    use_bwconncomp = true;
else
    use_bwconncomp = false;
end

%% Input matrix dimensions: 
if ismatrix(differenceTrials)
    [nDataPoints1, nTrials] = size(differenceTrials);
    nDataPoints2 = 1;
elseif ndims(differenceTrials) == 3
    [nDataPoints1, nDataPoints2, nTrials] = size(differenceTrials);
else
    error('Input variable "differenceTrials" needs to be 2D or 3D');
end

% Check if number of requested permutations is possible:
if 2^nTrials < nPermutations
    error('Impossible number of permutations requested (larger than 2^(number of trials)).');
end

% Initialize output variables
clusters = cell(nClusters,1);
p_values = ones(nClusters,1);
distribution = cell(nClusters,1);

% Compute tThreshold
tThreshold = abs(tinv(pThreshold/2, nTrials-1)); % pThreshold/2 because 2 tailed now (KK)

% PRODUCE PERMUTATION VECTORS
if nTrials > log2(1000*nPermutations) % there are at least 1000 times more possible permutations 
                                      % than the requested number, so randomly selected permutations 
                                      % will not be repeated
    Perm = round(rand(nTrials,nPermutations)) * 2 - 1; % Randomly generate 1's and -1's .
    
else                                  % The number of requested permutations is close to the maximum 
                                      % which can be produced by the number of trials, so permutation 
                                      % sequences are drawn without repetition. 
    Perm = NaN(nTrials,nPermutations);
    % generate a non-repeating list of permutations by taking a list of
    % non-repating numbers (up to nPermutations), and translating them 
    % into binary (0's and 1's): 
    rndB = dec2bin(randperm(2^nTrials,nPermutations)-1);
    % if the leading bit of all the numbers is 0, it will be truncated, so
    % we need to fill it in:
    nBits = size(rndB,2);
    if nBits < nTrials
        rndB(:,(nTrials-nBits+1):nTrials) = rndB;
        rndB(:,1:nTrials-nBits) = '0';
    end
    % translate the bits into -1 and 1:
    for ii=1:numel(Perm)
        Perm(ii) = str2double(rndB(ii)) * 2 - 1;
    end
end

% RUN PRIMARY PERMUTATION
tValues = zeros(nDataPoints1,nDataPoints2);
if nDataPoints2 > 1
    for t = 1:nDataPoints2
        tValues(:,t) = simpleTTest(squeeze(differenceTrials(:,t,:))',0);
    end
else
    tValues = simpleTTest(differenceTrials',0);
end
% Find the above-threshold clusters: 
if use_bwconncomp
    CC = bwconncomp(abs(tValues) > tThreshold,4); % added abs (KK), connected component > 4
    cMapPrimary = zeros(size(tValues));
    tSumPrimary = zeros(CC.NumObjects,1);
    for i=1:CC.NumObjects
        cMapPrimary(CC.PixelIdxList{i}) = i;
        tSumPrimary(i) = sum(abs(tValues(CC.PixelIdxList{i}))); % add abs
    end
else % Otherwise use this custom function, which is slower for large matrices
    T = tValues;
    T(tValues <= tThreshold) = 0;
    [cMapPrimary, tSumPrimary] = GetClusters(T);
end

% Sort clusters:
[tSumPrimary, tSumIdx] = sort(tSumPrimary,'descend');


% RUN PERMUTATIONS
for p = 1:nPermutations
    if nDataPoints2 > 1
        for t = 1:nDataPoints2
            D = squeeze(differenceTrials(:,t,:));
            D = bsxfun(@times,D',Perm(:,p));
            tValues(:,t) = simpleTTest(D,0);
        end
    else
        D = bsxfun(@times,differenceTrials',Perm(:,p));
        tValues = simpleTTest(D,0);
    end
    if use_bwconncomp
        CC = bwconncomp(abs(tValues) > tThreshold,4); % add abs
        tSum = zeros(CC.NumObjects,1);
        for i=1:CC.NumObjects
            tSum(i) = sum(abs(tValues(CC.PixelIdxList{i}))); % add abs
        end
    else
        T = zeros(size(tValues));
        T(tValues <= tThreshold) = 0;
        [~, tSum] = GetClusters(T);
    end
    tSum = sort(tSum,'descend');
    for clustIdx = 1:min(nClusters,length(tSum))
        distribution{clustIdx}(p) = tSum(clustIdx);
    end
end

%% DETERIMNE SIGNIFICANCE

for clustIdx = 1:min(nClusters,length(tSumPrimary))
    ii = sum(distribution{clustIdx} >= tSumPrimary(clustIdx));
    clusters{clustIdx} = find(cMapPrimary == tSumIdx(clustIdx));
    p_values(clustIdx) = (ii+1) / (nPermutations+1);
end

% return regular arrays if only one cluster is requested
t_sums = tSumPrimary;
if nClusters ==1
    clusters = clusters{1};
    distribution = distribution{1};
end

end

function [cmap, sum] = GetClusters(data)
% Identifies contiguous clusters of non-zero pixels in a 2D map. Returns a
% same-size map with each pixel coding its cluster ID, and a cell array
% holding the sum of each cluster's member pixels. 
% This function is called only if the built-in function "bwconncomp" is 
% not supported. 

[Ny, Nx] = size(data);               % map dimensions
Nc = 0;                              % number of clusters
cmap = zeros(size(data));            % cluster map
sum = [];                            % cluster statistics

% Run over each pixel in the map
for i=1:Ny 
    for j=1:Nx 
        if ( data(i,j) > 0 ) % If this pixel is non-zero...
            if ( j>1 ) && ( cmap(i,j-1) ~= 0 )  % if  the pixel on the left is mapped to a cluster:
                c = cmap(i,j-1); % copy that pixel's cluster ID...
                cmap(i,j) = c; % ... to this point on the map... 
                sum(c) = sum(c) + data(i,j); % ...and add the current pixel value to the cluster's sum
                if ( i>1 ) && ( cmap(i-1,j) ~= 0 ) && ( cmap(i-1,j) ~= c ) % now if the pixel above is also mapped to a (different) cluster
                    old_c = cmap(i-1,j); % then the current and previous clusters need to be merged.
                    ReplaceClusters(c,old_c);% This function replaces the ID of all pixels in the current cluster with the previous ID
                end
            elseif ( i>1 ) && ( cmap(i-1,j) ~= 0 ) % If the pixel on the left was not mapped but the one above is
                c = cmap(i-1,j);% copy that pixel's cluster ID...
                cmap(i,j) = c;% ... to this point on the map... 
                sum(c) = sum(c) + data(i-1,j);% ...and add the current pixel value to the cluster's sum
            else % if this pixel has no immediate neighbors 
                Nc = Nc+1;% Increase the cluster count
                cmap(i,j) = Nc; % Give this pixel the new ID
                sum(Nc) = data(i,j); % And add its value to the cluster sum.
            end
        end
    end
end

    function ReplaceClusters(c1,c2)
        cmap(cmap==c1) = c2; % Replace one value with another...
        for cc=(c1+1):Nc % ...and shift down all the higher IDs. 
            cmap(cmap==cc)=cc-1;
        end
        sum(c2) = sum(c2) + sum(c1); % Combine the two cluster values
        sum(c1) = []; % Truncate the invalid index out 
        Nc = length(sum); % and update the current number of clusters. 
    end
end

function [t, df] = simpleTTest(x,m)
    %TTEST  Hypothesis test: Compares the sample average to a constant.
    %   [STATS] = TTEST(X,M) performs a T-test to determine
    %   if a sample from a normal distribution (in X) could have mean M.
    %  Modified from ttest function in statistical toolbox of Matlab
    % The  modification is that it returns only t value and df. 
    %  The reason is that calculating the critical value that
    % passes the threshold via the tinv function takes ages in the original
    % function and therefore it slows down functions with many
    % iterations.
    % Written by Leon Deouell. 
    % 
    % Modified by Edden Gerber 19.11.2013: Added support for x being a matrix, where columns are
    % observations and rows are variables (output is a vector). 

    if nargin < 1, 
        error('Requires at least one input argument.'); 
    end

    if nargin < 2
        m = 0;
    end

    samplesize  = size(x,1);
    xmean = sum(x)/samplesize; % works faster then mean

    % compute std  (based on the std function, but without unnecessary stages
    % which make that function general, but slow (especially using repmat)
    xc = bsxfun(@minus,x,xmean);  % Remove mean
    xstd = sqrt(sum(conj(xc).*xc,1)/(samplesize-1));

    ser = xstd ./ sqrt(samplesize);
    tval = (xmean - m) ./ ser;
    % stats = struct('tstat', tval, 'df', samplesize-1);
    t = tval;
    df = samplesize-1;

end

