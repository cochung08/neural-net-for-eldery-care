function theta = initParams(inputSize, hiddenSizes, outputSize)    
    hiddenOverallSize = 0;
    for i=1:length(hiddenSizes)
        hiddenOverallSize = hiddenOverallSize + hiddenSizes(i);
    end    
    r  = sqrt(6) / sqrt(hiddenOverallSize+inputSize+outputSize);   % init weights uniformly between [-r, r]
    weightsFirst = rand(inputSize*hiddenSizes(1), 1) * 2 * r - r;
    weightsLast = rand(hiddenSizes(end)*outputSize, 1) * 2 * r - r;
    biasFirst = zeros(hiddenSizes(1), 1);
    biasLast = zeros(outputSize, 1);
    weightsMid = [];
    biasMid = [];
    if(length(hiddenSizes) > 1)
        midWeightsSums = zeros(length(hiddenSizes)-1, 1);
        midBiasSums = zeros(length(hiddenSizes)-1, 1);
        for i=1:length(midWeightsSums)
            midWeightsSums(i) = hiddenSizes(i) * hiddenSizes(i+1);
            midBiasSums(i) = hiddenSizes(i+1);
        end
        midWeightsSum = sum(midWeightsSums);
        midBiasSum = sum(midBiasSums);
        weightsMid = rand(midWeightsSum, 1) * 2 * r - r;
        biasMid = rand(midBiasSum, 1) * 2 * r - r;
    end
    theta = [weightsFirst(:) ; weightsMid(:) ; weightsLast(:) ; biasFirst(:) ; biasMid(:) ; biasLast(:)];    
end


