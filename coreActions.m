% core action of neural network training
function [cost, grad] = coreActions(theta, inputSize, hiddenSizes, outputSize, ...
                                             lambda, images, labels)
    % the first weight matrix
	weightsMatrixFirst = reshape(theta(1:hiddenSizes(1)*inputSize), hiddenSizes(1), inputSize);
    wMidStartIdx = hiddenSizes(1)*inputSize + 1;
    midWeightsSize = 0;
    for i = 1:length(hiddenSizes)-1
        % need to generate weight matrix dynamically
        nextMatrix = reshape(theta(wMidStartIdx:wMidStartIdx+(hiddenSizes(i)*hiddenSizes(i+1))-1), hiddenSizes(i+1), hiddenSizes(i));
        midWeightsSize = midWeightsSize + 0; % to be done
    end
    
    wRightStartIdx = wMidStartIdx + midWeightsSize;
    % the last weight matrix
	weightsMatrixLast = reshape(theta(wRightStartIdx:wRightStartIdx+hiddenSizes(end)*outputSize-1), outputSize, hiddenSizes(end));
    
    % the first bias vector
    bFirstStartIdx = wRightStartIdx + hiddenSizes(end)*outputSize;
    biasFirst = theta(bFirstStartIdx:bFirstStartIdx+hiddenSizes(1)-1);
    bMidStartIdx = bFirstStartIdx + length(biasFirst);
    % add code to init the mid bias vectors
    
    % the last bias vector
    biasLast = theta(bMidStartIdx:bMidStartIdx+outputSize-1);
    
    cost = 0;
    weightsFirstGrad = zeros(size(weightsMatrixFirst));
    weightsLastGrad = zeros(size(weightsMatrixLast));
    biasFirstGrad = zeros(size(biasFirst));
    biasLastGrad = zeros(size(biasLast));
    %fprintf('start of loop: %s\n', datestr(now, 'dd-mm-yyyy HH:MM:SS FFF'));
    parfor idx = 1:size(images, 2)
        % start of feedforward
        label = labels(:,idx); %given
        zFirst = weightsMatrixFirst * images(:,idx) +  biasFirst;
        activationFirst = sigmoid(zFirst);
        zLast = weightsMatrixLast * activationFirst + biasLast;
        output = sigmoid(zLast);
        % end of feedforward
        
        costTmp =(0.5)*(sum((output - label).^2));     %cost for this sample
        cost = cost + costTmp; %given
        
        % start of backprop
        deltaLast = (sigmoid(zLast).*(1-sigmoid(zLast))) .* ((label - output) * (-1));
        deltaFirst = zeros(hiddenSizes,1);
        for i=1:hiddenSizes
            entry = (sigmoid(zFirst(i))*(1-sigmoid(zFirst(i)))) * (sum(weightsMatrixLast(:,i).* deltaLast));
            deltaFirst(i,1)=entry;
        end
        % end of backprop
        
        %update grads
        weightsLastGrad = weightsLastGrad + deltaLast*(activationFirst');
        weightsFirstGrad = weightsFirstGrad + deltaFirst*(images(:,idx)');
        biasLastGrad = biasLastGrad + deltaLast;
        biasFirstGrad = biasFirstGrad + deltaFirst;
        
    end
    %fprintf('end of loop: %s\n', datestr(now, 'dd-mm-yyyy HH:MM:SS FFF'));
    grad = [weightsFirstGrad(:) ; weightsLastGrad(:) ; biasFirstGrad(:) ; biasLastGrad(:)];
    
end









