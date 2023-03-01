function y = SLRM_1D_gradRank(xt,Pr,fftPcVw)
% note: unitary normalization for gradients of norms dependent upon fft
% operators is very deceiving!

%% initialize routine parameters
    Nx = size(xt,1);%size of image
    Nf = size(fftPcVw,2);%size of filter
    y = zeros(Nx,1);%gradient vector to be returned
    xw = fft(xt);%calculate spectrum (unitary normalization handled below...)
%% cycle over each column vector of Vw...
    for i=1:Nf
        hi = fftPcVw(:,i);%pull out the i'th column of Vw
        y = y + conj(hi).*fft(Pr*ifft(hi.*xw));%compute i'th term of gradient and add to running sum
    end
%% and finally compute the ifft
    y = ifft(y)*Nx;%the extra factor of Nx accounts for each factor of sqrt(Nx) from "fft(Pr*ifft(" in line 13
end