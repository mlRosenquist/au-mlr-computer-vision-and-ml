function [model] = ovrtrain(y, train_data, cmd)

labelSet = unique(y);
labelSetSize = length(labelSet);
models = cell(labelSetSize,1);

for i=1:labelSetSize
    models{i} = svmtrain(double(y == labelSet(i)), train_data, cmd);
end

model = struct('models', {models}, 'labelSet', labelSet);