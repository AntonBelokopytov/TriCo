Site-packages and data: https://drive.google.com/drive/u/0/folders/1uxSsBCCkTNMPzyf7fSFK3O87xk00QWFL


It is better to import Fieldtrip by strings because it has some problems with uploading:

close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end