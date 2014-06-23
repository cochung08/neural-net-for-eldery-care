function numgrad = computeNumericalGradient(paraVectorT, inputSize, hiddenSizes, outputSize, lambda, ...
                                                images_padded, labelData, estimatedGrad)
                                            
    epsilon = 0.0001;
    % Initialize numgrad with zeros
    numgrad = zeros(size(paraVectorT));
    loopNum = size(paraVectorT, 1);
    
    % % % % % % % % % % % % % % % %
    N = loopNum;
    try % Initialization
       ppm = ParforProgressStarter2('test', N, 0.1);
    catch me % make sure "ParforProgressStarter2" didn't get moved to a different directory
       if strcmp(me.message, 'Undefined function or method ''ParforProgressStarter2'' for input arguments of type ''char''.')
           error('ParforProgressStarter2 not in path.');
       else
           % this should NEVER EVER happen.
           msg{1} = 'Unknown error while initializing "ParforProgressStarter2":';
           msg{2} = me.message;
           print_error_red(msg);
           % backup solution so that we can still continue.
           ppm.increment = nan(1, nbr_files);
       end
    end
    % % % % % % % % % % % % % % % %
    
    for i = 1:loopNum
        paraVector = paraVectorT;
        paraVector(i) = paraVector(i) + epsilon;
        [cost1, grad] = coreActions(paraVector, inputSize, hiddenSizes, outputSize, lambda, ...
                                            images_padded, labelData);
        paraVector(i) = paraVector(i) - (2*epsilon);
        [cost2, grad] = coreActions(paraVector, inputSize, hiddenSizes, outputSize, lambda, ...
                                            images_padded, labelData);

        % compute the numerical 
        temp = (cost1 - cost2) / (2 * epsilon);
        numgrad(i) = temp;
        
        diff = norm(temp-estimatedGrad(i));
        threshold = 10^-10;
        if(diff > threshold)
            fprintf('numerical: %i, ', temp);
            fprintf('estimated: %i. \n', estimatedGrad(i));
            fprintf('loop No: %i, \n', i);
            fprintf('diff too large! diff: %i.\n', diff);
        end
        ppm.increment(i);
    end
    
    try % use try / catch here, since delete(struct) will raise an error.
       delete(ppm);
    catch me %#ok<NASGU>
    end
end



