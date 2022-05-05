function [p_vals, f_sums, f_vals] = permclusterANOVA(data, factor1, factor2, num_perm, balance, num_clusters)
%   PERMCLUSTERANOVA is a cluster-based permutation for effects of 2D data (paired var).
%   Based on Bullmore & Suckling 2004 for 1-D and 2-D vectors.
%   Dependencies: permBetweenFactorANOVA, cluster/v2
%   
%   Use as
%      [p_val, tsum, cluster] = permclusterANOVA(data, factor1, factor2, timedim, num_perm, balance)
%   where data is in matrix form [time by trial], num-perm is the number of
%   permutations to test for the randomized clusters, num-clusters are the
%   number of clusters that will be reported on. Factor 1 & factor 2 are in
%   format [trial by 1] and factor2 must have at most 2 levels. Balance is
%   specified as 'yes' or 'no', if 'yes' then unequal sample sizes per
%   treatment group will be corrected for by using a random/first subset of
%   treatment group trials. The number of clusters specified ****
%   
%   Output
%   p_vals are the p_value calculated from the randomized permutations
%   (i.e. the proportion of randomized permutation f-statistics that are
%   greater than that of the observed f-statistic), f_sums are the maximum
%   num_clusters cluster sums, f_vals are all of the f-values produced for
%   the observed data
%   
%   Created by Ilina Bhaya-Grossman, ibg@berkeley.edu 
%   Modified by Lingyun Zhao, 2019.05

addpath(genpath('../../ecog_scripts'))
time_points = size(data, 1);
if nargin < 3, factor2 = []; end
if nargin < 4, num_perm = 1000; end
if nargin < 5, balance = 'no'; end
if nargin < 6, num_clusters = 1; end

% two sided testing (f-stats)
p_thresh = 1-0.05;
p_cutoff = 0.1;

% check for treatment levels per group
if length(unique(factor2)) > 2
    error("Too many levels per factor");
end

% number of levels in each factor is 2 (4 treatment groups)
% A_df = j, B_df = k
k = length(unique(factor1)); n = size(data, 2);
if length(unique(factor2)) ~= 1
    j = 2; 
    threshold = finv(p_thresh, (k-1)*(j-1), n-k*j);   
else
    threshold = finv(p_thresh, (k-1), n-k);
end

% calculate observed f-value & permuted f-values per time point (num_perms)
data_wlabels = cell(time_points, 1);
for t = 1:time_points
    temp = [factor1 factor2 data(t, :)'];
    if strcmp(balance,'yes')
        % format data such that it is safe to input to helper function
        % fixed s.t. first trials will be selected for balancing
        data_wlabels(t) = {balanceFactors(temp, 'yes')};
    else
        data_wlabels(t) = {temp};
    end
end
tic
[fObservedValues, ~, fPermutedValues] = permBetweenFactorANOVA(data_wlabels, num_perm);
toc

%% find observed clusters of f (where the f-value true surpasses f-threshold)
observed_fstat = nan(num_clusters, 3);
f_locs= cell(num_clusters, 3);
for i =1:3
    [fstat, f_loc] = cluster_fstat(fObservedValues(:, i), threshold, num_clusters);
    observed_fstat(1:length(fstat), i) = fstat;
    f_locs(1:length(f_loc), i) = f_loc;
end

%% find permuted clusters of f (where the f-value surpasses f-threshold) 
perm_fstat = nan(num_clusters, 3, num_perm);
for j = 1:num_perm
    for i = 1:3 % 2 main effects, 1 interaction
        % clustering & switching dimensionality of permuted f-values
        fstat = cluster_fstat(squeeze(fPermutedValues(:, j, i)), threshold, num_clusters);
        perm_fstat(1:length(fstat), i, j) = fstat;
    end
end

%% calculate the p-value based on the histogram of f-statistics
% keep all significant first order clusters and discard any significant
% clusters that are produced after non-significant clusters to avoid small
% noisy spikes
p_vals = ones(size(data,1), 3);
prev_ps = zeros(1, 3);
for j = 1:num_clusters
    for i = 1:3 % 2 main effects, 1 interaction
        if prev_ps(i) < p_cutoff && ~isempty(cell2mat(f_locs(j, i)))
            p_vals_temp = (nansum(squeeze(perm_fstat(j, i, :))>observed_fstat(j, i)')+1)/(num_perm+1);
            p_vals(cell2mat(f_locs(j, i))',i) = p_vals_temp;
            prev_ps(i) = p_vals_temp;
        end
    end
end

% filter out insignificant p-values
% p_vals(p_vals > p_cutoff) = 1;

% format output
f_sums = observed_fstat;
f_vals = fObservedValues;

end

function [lfstat, lfinds] = cluster_fstat(fvals, threshold, num_clusts)  
    
    if nargin < 3, num_clusts = 1; end
    fval_sums = [0];
    fval_clusters = {0};
    indices = find(fvals > threshold);

    % cluster adjacent samples (temporal) & sum t-vals in a cluster
    if ~isempty(indices)
        [fval_sums, fval_clusters] = cluster(indices, fvals);
    end
    % take largest cluster t-statistic
    [lfstat, max_ind] = maxk(abs(fval_sums), num_clusts);
    lfinds = cell(num_clusts, 1);
    if lfstat ~= 0, lfinds(1:length(max_ind), 1) = fval_clusters(max_ind); end
end

function [balancedData] = balanceFactors(data, first)
    % first refers to whether or not the balanced indices will be randomly
    % selected or be the consecutively selected (i.e. the first x trials)
    if nargin < 2; first = 'no'; end
    dataWithLabels=sortrows(data,[1 2]);

    labelsA=unique(dataWithLabels(:,1));
    labelsB=unique(dataWithLabels(:,2));
    nlabelsA=length(labelsA);
    nlabelsB=length(labelsB);

    % assumes that number of levels in A are equal to number of levels in B
    % produces the nlabelsA * nlabelsB treatment groups
    combinations = combvec(labelsA', labelsB')';
    withinCellN=grpstats(dataWithLabels(:,3), dataWithLabels(:,[1 2]), 'numel');

    if length(unique(withinCellN))>1
        min_trls = min(withinCellN);
        balancedData = nan(min_trls*nlabelsA*nlabelsB, 3);
        cnt = 1;
        for i = 1:size(combinations, 1)
            inds = find(sum(dataWithLabels(:, 1:2)==combinations(i, :), 2) == 2);
            % IMPORTANT: This will re-order the trials so if anything is
            % reliant on ordering of trials be weary of this function
            if ~isempty(inds)
                if strcmp(first, 'yes')
                    balancedInds = inds(1:min_trls);
                else
                    balancedInds = randsample(inds, min_trls);
                end
                balancedData(cnt:cnt+min_trls-1, :) = dataWithLabels(balancedInds, :);
                cnt = cnt + min_trls;
            end
        end
    end
end

function [sums, clusters] = cluster(indices, fvals)
    % function [sums, clusters] = find_clusters2(indices,fvals)
    % inputs: 
    %   indices - list of indices to be clustered
    % outputs:
    %   sums - sum of fvals for each cluster
    %   clusters - cell of indices within each cluster
    % Yulia Oganian, Feb 2019 (modified by Ilina Bhaya-Grossman)
    
    %% if no fvals, sums will return size of single clusters.
    if nargin<2
        fvals = [];
        fvals(indices')=1;
    end

    %% find cluster onsets - indices that are not consecutive
    b = diff(indices');
    ClustOns = find(b>1)+1; % correct for the first index in indices, because b has 1 entry less than indices

    if isempty(ClustOns) || ClustOns(1) ~=1 % if the very first index is a 1-item cluster
        ClustOns = [1, ClustOns];
    end
    ClustSize = diff([ClustOns length(indices)+1]); % add last item to get length of very last cluster, because it isn't followed by a cluster onset

    %% get outputs
    sums = nan(length(ClustOns),1);
    clusters = cell(length(ClustOns),1);
    for cl = 1:length(ClustOns)
        clusters{cl} = indices(ClustOns(cl):(ClustOns(cl)+ClustSize(cl)-1));
        sums(cl) = sum(fvals( clusters{cl} ));
    end
end

