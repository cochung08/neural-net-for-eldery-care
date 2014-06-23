function output = computeOutput(theta, inputSize, hiddenSizes, outputSize, ...
                                             lambda, imageSample)
    % the first weight matrix
	weightsMatrixFirst = reshape(theta(1:hiddenSizes(1)*inputSize), hiddenSizes(1), inputSize);
    wMidStartIdx = hiddenSizes(1)*inputSize + 1;
    midWeightsSize = 0;
    parfor i = 1:length(hiddenSizes)-1
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
    
    % start of feedforward    
    zFirst = weightsMatrixFirst * imageSample + biasFirst;
    activationFirst = sigmoid(zFirst);
    zLast = weightsMatrixLast * activationFirst + biasLast;
    output = sigmoid(zLast);    
end




