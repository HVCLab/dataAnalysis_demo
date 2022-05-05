function [datapath, prcsdpath] = setDatapath()
a = fileparts(mfilename('fullpath'));
disp(a);
if ~isempty(strfind(a, 'Users'))
    datapath = '/Users/yuliao/Dropbox/data/';
    prcsdpath = '/Users/yuliao/Dropbox/prcsd_data/';
    addpath(genpath('/Users/yuliao/software/matlab/'))
elseif ~isempty(strfind(a, 'home'))
    datapath = '/userdata/yuliao/data/'; 
    prcsdpath = '/data_store1/human/prcsd_data/';
else
    warning('path not located');
    
    datapath = '';
    prcsdpath ='';
end
addpath(genpath('../nansuite'));
end

