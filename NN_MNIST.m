% This code implements MLNN from scratch on MNIST Data

% mnist_train.mat : should have mnist data in the format of N X 785 (First Column contains the label for corresponding row)
% N - Number of samples
% mnist_test.mat  : Follows the same as mnist_train.mat


data = load('mnist_train.mat');

data = data.traindata;       Traindata = data(:,2:end);      TrainLabe = data(:,1); 

X = Traindata';
X = X - mean(X,2);
X = X./255;

j=1;
for batch =1:100
    minibat(:,:,batch) = X(:,j:j+49);
    j = j + 50;
end

Y = zeros(10,size(X,2));

for i=1:size(X,2)
    idx = TrainLabe(i);  Y(idx+1,i)=1;  % One hot encoded labels
end

j =1;
for batch =1:100
    minilab(:,:,batch) = Y(:,j:j+49);
    j = j + 50;
end

W1 = rand(392,784);
b1 = rand(392,1);
W2 = rand(10,392);
b2 = rand(10,1);


start=1;
epochs=200;
epoch =1;
while(epochs)
    
    for batch =1:100
        
            x = minibat(:,:,batch);
            y = minilab(:,:,batch);

            Z1=W1*x+b1;   A1=relu(Z1);      Z2=W2*A1+b2;    A2=relu(Z2);   loss(batch) = 1/size(x,2) * (norm(A2 - y))^2;   
            
           
            da2  = 2/length(Traindata)*(A2 - y);
            
           
            dz2 = da2.* sigmoid(Z2).*sigmoid(1-Z2);
            
            db2  = mean(dz2,2);
            
            da1  = W2.'* dz2;
            
            dw2  = dz2 * A1.';
            
            dz1 = da1.* double(Z1>0);
            
            db1 = mean(dz1,2);
            
            dw1 = dz1 * x.';            
            
                   
            W1=W1-0.1*dw1;
            b1=b1-0.1*db1;            
            W2=W2-0.1*dw2;
            b2=b2-0.1*db2;
            
            vec=['Epoch =',num2str(epoch),  '  Iterat =  ',num2str(batch),'   Loss = ' num2str(loss(batch))];
            disp(vec);
            
    end
            disp(['']);
            lossepoch(epoch) = mean(loss);
            disp(['Avg Los for Epoch : ', num2str(epoch), ' is ',num2str(lossepoch(epoch) )]);
            disp(['___________________________________________________________________________________________']);
            clear loss;
            epoch = epoch+1;
            epochs=epochs-1;
            
end
 

%Testing

data = load('mnist_test.mat');

data = data.testdata;       Testdata = data(:,2:end);      TestLabe = data(:,1); 



y = zeros(10,500);

for i=1:500
    idx = TestLabe(i);  y(idx+1,i)=1;  % One hot encoded labels
end

x = Testdata';
x = x - mean(x,2);
x = x./255;

Z1=W1*x+b1;   A1=relu(Z1);      Z2=W2*A1+b2;    A2=relu(Z2);
losstest = 1/size(Testdata,2) * (norm(A2 - y))^2;   