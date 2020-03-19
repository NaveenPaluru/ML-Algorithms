function [alpha, mu, sigma, LOGLKHD,iter] = mygmm(data, numberofgaussians,flag)

        
	% This code implements EM Algorithm for building GMMs
	% Ref Pattern Recognition - by -Christopher M. Bishop

	% Input
	% data in D X N Format (data should contain training samples from same class)
	% D - Feature VEctor Dimension
	% N - Number of training Samples
	%flag = 1, for full Covariance and flag = 0, for diagonal Covariance (Naive)
	% numberofgaussians -number of gaussians to fit (2,3,4 etc.)

	% Output
	% alpha  - weights of each gaussian
	% mu     - means of each gaussain    % sigma  - cov of each gaussian     
        %LOGLKHD - The complete logliklihood
	%iter    - NUmber of iterations took for convergence

        c1 = data';
	[idx]=kmeans(c1,numberofgaussians);
        
        %Initialising Parameters For EM Algorithm
        for i=1:numberofgaussians
                meanc1 (:,:,i) = mean(c1(idx==i,:),1);  
                if flag
                    sigmac1(:,:,i) = cov(c1(idx==i,:));
                else
                    sigmac1(:,:,i) = diag(diag(cov(c1(idx==i,:))));      
                end
                pic1(i)=length(c1(idx==i))/size(c1,1);
        end
        %----------------------------------------------------------------------------------------------------------------------------------------------------
        %EM Algorithm
        start=1;f=1;
        temp=0;iter=1;checkpoints=0;
        while(start)
            
                   
                %Making Denominator For Finding Responsibilities
                
                for i=1:size(c1,1)
                    sumc1=0;
                    for j=1:numberofgaussians
                             mu1=meanc1(:,:,j);
                             k1=1/((2*pi)^(0.5*size(c1,2))*(det(sigmac1(:,:,j)))^0.5);
                             vec = c1(i,:)-mu1;
                             G1(j)=pic1(j)*k1*exp(-0.5*( vec *inv(sigmac1(:,:,j))* vec')); 
                             sumc1=sumc1+G1(j);
                    end
                    denr(i)=sumc1;                    
                end
                
                %Finding Ynk(Xn)
                for j=1:numberofgaussians
                    mu1=meanc1(:,:,j);
                    k1=1/((2*pi)^(0.5*size(c1,2))*(det(sigmac1(:,:,j)))^0.5);
                    for i=1:size(c1,1)
                             vec = c1(i,:)-mu1;
                             G11=pic1(j)*k1*exp(-0.5*( vec *inv(sigmac1(:,:,j))* vec')); 
                             res(i)=G11/denr(i);
                    end
                    res1(:,:,j)=res;
                end           
                                           
                           
                 %Finding New Parameters
                 for j=1:numberofgaussians
                        
                        rs=res1(:,:,j);
                        s=sum(rs);
                        Nks1(j)=s;
                        e=0;
                        for i=1:size(c1,1)
                            e=e+rs(i).*data(:,i);
                        end
                        meanc1(:,:,j)=e'./Nks1(j);
                        %Cov For every Gaussian
                        d=0;
                        rs=res1(:,:,j);
                        mu1=meanc1(:,:,j);
                        for i=1:size(c1,1)
                                vec = c1(i,:)-mu1;
                                if flag
                                    sg(:,:,i)=rs(i).* vec'*vec;
                                else
                                    sg(:,:,i)=rs(i).* diag(diag(vec'*vec));
                                end
                                d=d+sg(:,:,i);
                        end
                        sigmac1(:,:,j)=d./Nks1(j);
                        %pik For every Gaussian
                        pic1(j)=Nks1(j)/size(c1,1);
                 end
                 
                 %Checking the convergence..
                 for i=1:size(c1,1)
                        T1=0;      
                        for j=1:numberofgaussians
                                 k1=1/((2*pi)^(0.5*size(c1,2))*(det(sigmac1(:,:,j)))^0.5);
                                 mu1=meanc1(:,:,j);
                                 vec = c1(i,:)-mu1;
                                 G1(j)=pic1(j)*k1*exp(-0.5*( vec *inv(sigmac1(:,:,j))* vec')); 
                                 T1=T1+G1(j);
                        end
                        llkhd(i)=log(T1);         
                 end
                 
                 LOGLKHD(iter) = sum(llkhd);
                 
                 if iter==1
                     start=1;
                     iter=iter+1;
                 else
                     if  LOGLKHD(iter)< LOGLKHD(iter-1) || checkpoints==10
                         start = 0;
                         alpha = pic1; mu = meanc1; sigma = sigmac1;                         
                     else
                         
                         start = 1;
                         if round(abs(LOGLKHD(iter)-LOGLKHD(iter-1))) <= 10
                             checkpoints=checkpoints+1
                         end
                         iter = iter+1;
                     end
                 end
                 
                 
                
%                  if sum(abs(llkhd))>temp   
%                         start=1;
%                  else
%                         start=0;
%                  end
%                  temp=sum(abs(llkhd));
                
                
                 
        end
end