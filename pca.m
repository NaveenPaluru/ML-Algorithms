function [U] = pca(N, D, traindata)

    % This code  perfoms Principal Component Analysis
    % Ref: Pattern Recognition by ChristopherM.Bishop

    % Input
    % N -Number of training Samples
    % D -Feature Vectors Dimensions
    % traindata - datamatrix in D X N Format (Mean Subtracted Data)

    % Output
    % Principal Components : U
    
    % High Dimensional PCA

    if N <= D  

            
            [V,l] = eig(traindata'*traindata / N);         % EV's
            [l,idx]= sort(abs(diag(l)),'descend');
            V = V(:,idx);    
            U   = traindata*V;
            k   = N;            
            for i = 1:k
              U(:,i) = U(:,i)/sqrt(N*l(i)); % Normalizing
            end   

%           
    
    % Low Dimensional PCA
    
    else
        
            [V,l] = eig(traindata*traindata' / N);         % EV's
            [l,idx]= sort(abs(diag(l)),'descend');
            V = V(:,idx);    
            U   = V;
            k   = N;            
             
    
    end
 
end