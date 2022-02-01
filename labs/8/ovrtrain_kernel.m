function [model] = ovrtrain_kernel(y, K, cmd)

labelSet = unique(y);
labelSetSize = length(labelSet);
models = cell(labelSetSize,1);

for i=1:labelSetSize
    models{i} = svmtrain(double(y == labelSet(i)), K, cmd);
    model = svmtrain(double(y == labelSet(i)), K, cmd);
end
model = struct('models', {models}, 'labelSet', labelSet);