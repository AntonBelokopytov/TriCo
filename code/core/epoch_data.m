function X_epo = epoch_data(X, Fs, Ws, Ss)
% EPOCH_DATA  Segment continuous data into overlapping epochs.
%
%   X_epo = epoch_data(X, Fs, Ws, Ss)
%
%   INPUT:
%       X   - continuous data matrix [T x D],
%             where T = number of time samples,
%                   D = number of channels (sensors)
%
%       Fs  - sampling frequency (Hz)
%
%       Ws  - epoch window length in seconds
%
%       Ss  - step (shift) between consecutive epochs in seconds
%             (if Ss < Ws → overlapping epochs)
%
%   OUTPUT:
%       X_epo - epoched data array [W x D x E],
%               where W = samples per epoch,
%                     D = number of channels,
%                     E = number of epochs
%
%   The function creates sliding windows over time with
%   fixed window length and fixed step.

% Convert window length from seconds to samples
W = fix(Ws * Fs);   % number of samples per epoch

% Convert step size from seconds to samples
S = fix(Ss * Fs);   % shift between epochs in samples

% Initial time index range for the first epoch
range = 1:W;

% Epoch counter
ep = 1;

% Preallocate empty array (will grow dynamically)
X_epo = [];

% Slide window over data until the last sample fits inside X
while range(end) <= size(X,1)
    
    % Extract epoch: [W x D]
    X_epo(:,:,ep) = X(range,:);
    
    % Shift window forward
    range = range + S;
    
    % Increment epoch counter
    ep = ep + 1;
end

% Remove singleton dimension if only one epoch
X_epo = squeeze(X_epo);

end
