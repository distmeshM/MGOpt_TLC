function unionResult = cellUnion(cellArray)
%CELLUNION Compute the union of numeric arrays stored in a cell array.
%
%   unionResult = CELLUNION(cellArray) returns the sorted union of all
%   numeric vectors contained in the input cell array. Non-numeric
%   elements are ignored with a warning.
%
%   Input:
%       cellArray - Cell array containing numeric arrays
%
%   Output:
%       unionResult - Row vector containing the sorted unique union
%
%   Example:
%       C = {[1 2 3], [3 4 5], [10 11]};
%       U = cellUnion(C);
%       % U = [1 2 3 4 5 10 11]

if nargin < 1
    error('Input argument is required.');
end

if ~iscell(cellArray)
    error('Input must be a cell array.');
end

unionResult = [];

try
    for i = 1:numel(cellArray)
        currentElement = cellArray{i};

        if isnumeric(currentElement)
            % Compute sorted union with previous results
            unionResult = union(unionResult, currentElement(:).');
        else
            warning('Ignoring non-numeric element at cell index %d.', i);
        end
    end

    % Ensure row vector output
    unionResult = unionResult(:).';

catch ME
    error('Error while computing union: %s', ME.message);
end
end