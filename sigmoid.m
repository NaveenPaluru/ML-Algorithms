
function Ans = sigmoid(Temp) 
           clear Ans;
           [i,j]=size(Temp);
           for m=1:i  
               for n=1:j
                    Ans(i,j)= 1/(1+exp(-1*Temp(i,j)));                   
               end
           end
end
               