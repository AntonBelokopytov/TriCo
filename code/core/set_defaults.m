function [opt, isdefault] = set_defaults(opt, varargin)
%[opt, isdefault]= set_defaults(opt, field/value list)
%
%Description:
% This functions fills in the given struct opt some new fields with
% default values, but only when these fields DO NOT exist before in opt.
% Existing fields are kept with their original values.

isdefault = [];
if ~isempty(opt)
    for Fld = fieldnames(opt)'
        isdefault.(Fld{1}) = 0;
    end
end
defopt = propertylist2struct(varargin{:});
for Fld = fieldnames(defopt)'
    fld = Fld{1};
    if ~isfield(opt, fld)
        [opt.(fld)] = deal(defopt.(fld));
        isdefault.(fld) = 1;
    end
end
end
