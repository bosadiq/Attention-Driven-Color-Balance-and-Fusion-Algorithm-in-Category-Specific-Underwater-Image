function Psnr = psnrxx(data2,output)
[m0,n0]=size(output);
[m1,n1]=size(data2);
 m=min(m0,m1);
 n=min(n0,n1);
% x=0;
 z=m*n;
for h=1:m
 for j=1:n
   %x=x+(double(data2(h,j))-double(output(h,j)))^2;
   x=(double(data2(h,j))-double(output(h,j)))^2;
 end
end
mse=x/z;
maxium=255*255;
Psnr=10*log10(maxium/mse);