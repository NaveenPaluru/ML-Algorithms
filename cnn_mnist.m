% This code implements CNN on MNIST Data from scratch

% Initializing weights by Xavier initialization

w1 = randn(5, 5, 6 ).*sqrt(2/784);   
b1 = randn(1,     6).*sqrt(2/784);     
w2 = randn(5,5,6,16).*sqrt(2/196);
b2 = randn(1    ,16).*sqrt(2/196);     
w3 = randn(120, 400).*sqrt(2/400);
b3 = randn(120,   1).*sqrt(2/400);   
w4 = randn(84,  120).*sqrt(2/120);
b4 = randn(84,    1).*sqrt(2/120);
w5 = randn(10,   84).*sqrt(2/084); 
b5 = randn(10,    1).*sqrt(2/084);

%Loaddata

%Subset of data (1000 training samples from MNIST Data)

data = load('./data/mnistdata.mat');
train_img = (data.TRI(:,:,1:10000)./255);
train_labels = data.TRL(1:10000);


epochs = 1; eeta = 0.01;
while(epochs<=20)

        for i=1:1000
            
            % STOCHASTIC GRADIENT DESCENT (Batch Size = 1)
            % FORWARD PROPAGATION
            
            % getting maps of first cnn layer
            for i1 = 1:6
                c1(:,:,i1)=conv2(train_img(:,:,i),w1(:,:,i1),'same')+b1(i1);
            end
            % nonlinearity
            r1 = max(0,c1);
            % Average pool
            for i1 = 1:6
                tmp = r1(:,:,i1);
                x=1;
                for j1 = 1:2:27
                    y=1;
                    for k1 = 1:2:27
                        box = tmp(j1:j1+1,k1:k1+1);
                        avg(x,y)= mean(mean(box));
                        y=y+1;
                    end
                    x=x+1;
                end
                a1(:,:,i1)=avg;
                clear avg tmp x y;
            end
            % getting maps of second cnn layer
            for i2 = 1:16
                for i22 = 1:6
                     c22(:,:,i22)=conv2(a1(:,:,i22),w2(:,:,i22,i1),'valid');
                end
                c2(:,:,i2)=sum(c22,3)+b2(i2);
            end
             % nonlinearity
            r2 = max(0,c2);
            % Average pool
            for i2 = 1:16
                tmp = r2(:,:,i2);
                x=1;
                for j2 = 1:2:9
                    y=1;
                    for k2 = 1:2:9
                        box = tmp(j2:j2+1,k2:k2+1);
                        avg(x,y)= mean(mean(box));
                        y=y+1;
                    end
                    x=x+1;
                end
                a2(:,:,i2)=avg;
                clear avg tmp x y;
            end
            % vectorizing the feature maps to make FC connections
            %FC1
            a22 = reshape(a2,400,1);
            c3 = w3*a22 + b3;    r3 = max(0,c3);
            %FC2
            c4 = w4*r3 + b4;    r4 = max(0,c4);
            %FC3
            c5 = w5*r4 + b5;
            %SOFTMAX
            ycap = exp(c5)./sum(exp(c5));
            
            %LOSS (Cross Entropy)
            loss(i) = -1 * log(ycap(train_labels(i)+1));
            
            %BACKWARD PROPAGATION
            e = zeros(10,1); 
            
            %gradients after crossing softmax layer            
            e(train_labels(i)+1)=1;       grad_c5 =  -1*(e - ycap);
            
            %gradients after crossing FC3
            grad_r4 = w5.' * grad_c5;    grad_w5 = grad_c5 * r4.';  grad_b5 = grad_c5;
            
            %gradients after crossing FC2
            grad_c4 = grad_r4.*(c4>0);  grad_w4 = grad_c4 * r3.';  grad_b4 = grad_c4;   grad_r3 = w4.' * grad_c4; 
            
            %gradients after crossing FC1
            grad_c3 = grad_r3.*(c3>0);  grad_w3 = grad_c3 * a22.';  grad_b3 = grad_c3;  grad_a22 = w3.' * grad_c3;
                   
            %reshaping gradients for a2
            grad_a2 = reshape(grad_a22,5,5,16);
            
            %gradients after crossing 2nd pooling Layer
            for i2 = 1:16
                tmp = grad_a2(:,:,i2);
                avg  = zeros(10,10);
                x=1;
                for j2 = 1:5
                    y=1;
                    for k2 = 1:5
                        avg(x:x+1,y:y+1) = 0.25*tmp(j2,k2);
                        y = y+2;
                    end
                    x = x+2;
                end
                grad_r2(:,:,i2) = avg;
                clear avg tmp x y ;
            end
            
            %gradients after crossing non linearity
            grad_c2 = grad_r2.* (c2>0);            
            
            %gradients after crossing conv2 layer            
            for i2 = 1:16
                for i22 = 1:6
                    grad_w2(:,:,i22,i2) = conv2(flip(flip(a1(:,:,i22),1)),grad_c2(:,:,i2),'valid');
                end
                grad_b2(i2) = mean(mean(grad_c2(:,:,i2)));
            end
            for i22 = 1:6
                for i2 = 1:16
                    tmp(:,:,i2) = conv2(grad_c2(:,:,i2),flip(flip(w2(:,:,i22,i2))),'full');
                end
                grad_a1(:,:,i22)= sum(tmp,3);
                clear tmp;
            end
            
            %gradients after crossing 1st pooling Layer
            for i1 = 1:6
                tmp = grad_a1(:,:,i1);
                avg  = zeros(28,28);
                x=1;
                for j1 = 1:14
                    y=1;
                    for k1 = 1:14
                        avg(x:x+1,y:y+1) = 0.25*tmp(j1,k1);
                        y = y+2;
                    end
                    x = x+2;
                end
                grad_r1(:,:,i1) = avg;
                clear avg tmp x y ;
            end
            
            %gradients after crossing non linearity
            grad_c1 = grad_r1.* (c1>0);  
            
            %gradients after crossing conv1 layer             
            for i1 = 1:6
                 tmp = zeros(32,32);
                 tmp(3:30,3:30)  = train_img(:,:,i);
                 grad_w1(:,:,i1) = conv2(flip(flip(tmp,1)),grad_c1(:,:,i1),'valid');
                 grad_b1(i1) = mean(mean(grad_c1(:,:,1)));
                 clear tmp;
            end
            
              % Uncomment this to verify the dimensions of gradients
            
%             disp(['w1  ' num2str(size(w1)) '  grad_w1  ' num2str(size(grad_w1))])
%             sprintf('\n')
%             disp(['b1  ' num2str(size(b1)) '  grad_b1  ' num2str(size(grad_b1))])
%             sprintf('\n')
%             disp(['w2  ' num2str(size(w2)) '  grad_w2  ' num2str(size(grad_w2))])
%             sprintf('\n')
%             disp(['b2  ' num2str(size(b2)) '  grad_b2  ' num2str(size(grad_b2))])
%             sprintf('\n')
%             disp(['w3  ' num2str(size(w3)) '  grad_w3  ' num2str(size(grad_w3))])
%             sprintf('\n')
%             disp(['b3  ' num2str(size(b3)) '  grad_b3  ' num2str(size(grad_b3))])
%             sprintf('\n')
%             disp(['w4  ' num2str(size(w4)) '  grad_w4  ' num2str(size(grad_w4))])
%             sprintf('\n')
%             disp(['b4  ' num2str(size(b4)) '  grad_b4  ' num2str(size(grad_b4))])
%             sprintf('\n')
%             disp(['w5  ' num2str(size(w5)) '  grad_w5  ' num2str(size(grad_w5))])
%             sprintf('\n')
%             disp(['b5  ' num2str(size(b5)) '  grad_b5  ' num2str(size(grad_b5))])


              % UPDATING PARAMETERS
              
              w1 = w1 - eeta*grad_w1;
              b1 = b1 - eeta*grad_b1;
              w2 = w2 - eeta*grad_w2;
              b2 = b2 - eeta*grad_b2;
              w3 = w3 - eeta*grad_w3;
              b3 = b3 - eeta*grad_b3;
              w4 = w4 - eeta*grad_w4;
              b4 = b4 - eeta*grad_b4;
              w5 = w5 - eeta*grad_w5;
              b5 = b5 - eeta*grad_b5;
              
              if (rem(i,100) == 0)
                  disp(['Epoch:  ',  num2str(epochs), ' Iteration:  ',  num2str(i) '  Loss :  ', num2str(mean(loss))])
              end
        end
        lossdata(epochs)=mean(loss);
        disp(['Average Loss For Epoch : ',num2str(epochs), ' is  ' , num2str(lossdata(epochs))])      
       
        epochs = epochs+1;
end
