function [w,projc1,projc2] = lda(trainc1, trainc2)

	% This code implements 2 class LDA     
	% Ref - Pattern Recognition Christopher M. Bishop

	% Input
	% trainc1 -class 1 train data 	in D X N Format
	% trainc2 -class 2 train data 	in D X N Format
	% N - NUmber of training samples
	% D - Feature Vector Dimension

	% Output
	% w - LDA projection
	% projc1 - projected data for class 1
	% projc2 - projected data for class 2

        m1 = mean(trainc1,2);
        m2 = mean(trainc2,2);     
        
        sw = cov(trainc1')+ cov(trainc2');
        
        w  = inv(sw)*(m1-m2);  
        
        projc1 = w'*trainc1;
        projc2 = w'*trainc2;



end