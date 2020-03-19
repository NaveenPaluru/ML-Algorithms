function Ans = relu(Temp) 
           Ans=0;
           [i,j]=size(Temp);
           for m=1:i  
               for n=1:j
                    Ans(i,j)=max(0,Temp(i,j));                   
               end
           end
	% Simply we can have
	% Ans = Temp(Temp>0);
end