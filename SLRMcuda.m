close all;
clear all;
set(0,'DefaultFigureWindowStyle','docked');

%% initialize SLR parameters
    dims = [512];%signal size (pixels)
    N = prod(dims);
    Ndims = numel(dims);
    N_SLR_its = 1;%number of SLR iterations
    fDims = [12];%filter size
    Nf = prod(fDims);
    SNR = 50;%signal-to-noise ratio
    lambda = 0.3;%lagrange multiplier (use ~0.75 for 2X US, ~0.25 for 10X US)
    US = 10;%undersample factor

%% initialize exponential parameters (currently using a 3-exp model)
    a1 = 10.01/dims(1) - 0.461*2*pi*1i;%exp argument 1
    a2 = 8.31/dims(1) - 0.441*2*pi*1i;%exp argument 2
    a3 = 12.22/dims(1) - 0.242*2*pi*1i;%exp argument 3
    A1 = -1.2;%exp coefficient 1
    A2 = 1.8;%exp coefficient 2
    A3 = 0.9;%exp coefficient 3

%% calculate true response and spectrum
    x = linspace(0,dims(1)-1,dims(1))';%I prefer column vectors
    dx = x/dims(1);
    R_true = A1*exp(-a1*x)+A2*exp(-a2*x)+A3*exp(-a3*x);
    Spec_true = fft(R_true);

%% synthesize noise and add itN_SLR to response
    %load('noise1.mat');
    noise = (max(R_true)/SNR)*randn(size(R_true)) + (max(R_true)/SNR)*1i*randn(size(R_true));
    R_noisy = R_true + noise;
    Spec_noisy = fft(R_noisy);

%% generate a sampling mask and make undersampled response
    %load('maskGood.mat');
    mask = makeMask(N,US);    
    %load('mask1.mat');
    R_US = R_noisy.*mask;
    USstringName = sprintf("Noisy + %iX US",US);

% %% create row- and column-selection matrices
%     d = zeros(1,N)';
%     d(1:Fsz) = 1;
%     Pc = diag(d);%column selection matrix (not currently in use)
%     d = ones(1,N)';
%     d(1:Fsz-1) = 0;
%     Pr = diag(d);%row selection matrix
    
%     R_US(dims(1)-10:dims(1)) = 0;
%     mask(dims(1)-10:dims(1)) = 1;

%% calculate b vector and initialzie time domain recon vector (xt)
    b = R_US;
    xt = R_US;

%% ***START OF SLR ALGORITHM***
%% build Pc
    Pc = ones(dims,1);
    for n=1:N
        IV = CI2IV1(n,dims);
        for dim=1:Ndims
            if(IV(dim)>fDims(dim))
                Pc(n) = 0;
            end
        end
    end
    Pc = diag(Pc);
%% build Pr
    Pr = ones(dims,1);
    for n=1:N
        IV = CI2IV1(n,dims);
        for dim=1:Ndims
            if(IV(dim) < fDims(dim))
                Pr(n) = 0;
            end
        end
    end
    Pr = diag(Pr);
%% initialize parameters for Toeplitz and SVD
    winDims = dims - fDims + 1;
    ToepRowSize = prod(winDims);
    ToepColSize = prod(fDims);
    Toep = zeros(ToepRowSize,ToepColSize);
    windowMap = int32(zeros(winDims));
    G = complex(zeros(ToepColSize,ToepColSize),zeros(ToepColSize,ToepColSize));
    
%% calculate window map
    for n=1:numel(windowMap)
        IV = CI2IV1(n,winDims);
        windowMap(n) = IV2CI1(IV,dims);
    end
    filterCI = int32(IV2CI1(fDims,dims));
    
%% prep GPU arrays
    cu_b = gpuArray(b);
    cu_xt = gpuArray(xt);
    cu_G = gpuArray(G);
    cu_windowMap = gpuArray(windowMap);
    cu_Pr = gpuArray(Pr);
    cu_AF = gpuArray(complex(zeros(Nf,1),zeros(Nf,1)));
    
%% for-loop over SLR iterations
    for i=1:N_SLR_its
        fprintf("SLR iteration: %i\n",i);
        
        %% build toeplitz
%             c = xt(Fsz:dims(1));
%             r = flipud(xt(1:fDims(1)));
%             T = toeplitz(c,r);
            
%             for col=1:ToepColSize
%                 shiftIV = CI2IV1(col,fDims);
%                 shiftCI = IV2CI1(shiftIV,dims)-1;% -1 due to 1-based indexing
%                 for row=1:ToepRowSize
%                     toep(row,col) = xt(filterCI + (windowMap(row)-1) - shiftCI);% -1 due to 1-based indexing
%                 end
%             end
            
        %% compute SVD of toeplitz
            %[U,S,V] = svd(T);
            %Vtrue = Toep'*Toep;
            %G = Toep'*Toep
            G = complex(zeros(ToepColSize,ToepColSize),zeros(ToepColSize,ToepColSize));
            for col=1:ToepColSize
                colShiftIV = CI2IV1(col,fDims);
                colShiftCI = IV2CI1(colShiftIV,dims)-1;% -1 due to 1-based indexing
                for row=1:ToepColSize
                    rowShiftIV = CI2IV1(row,fDims);
                    rowShiftCI = IV2CI1(rowShiftIV,dims)-1;% -1 due to 1-based indexing
                    for n=1:ToepRowSize
                        G(row,col) = G(row,col) + conj(xt(filterCI + (windowMap(n)-1) - rowShiftCI))*xt(filterCI + (windowMap(n)-1) - colShiftCI);
                    end
                end
            end
            [V,D] = eig(G);
            S = sqrt(D);
            maxS = max(diag(S));
            minS = min(diag(S));
            
            mexcudaSLRM(cu_xt,cu_G,cu_AF,cu_windowMap,filterCI-1,ToepRowSize,ToepColSize);
            
        %% compute weighted V space
            eps = sqrt(maxS*(minS+((1e-12)*maxS)))/(1+sqrt(2));%calculate epsilon in rank function
                %note: the term "+((1e-12)*S(1,1)))" is added just in case
                %min(S) is so small that eps calculation would be poorly
                %conditioned
            Nullw = diag(S);
            Nullw = 1./(Nullw+eps);
            Vw = V*diag(Nullw);%weighted V space

        %% project Vw onto image space, then compute fft of column vectors
            fftPcVw = fft(Vw,N,1)/sqrt(N);%unitary tranform

        %% compute rank
            rank = xt'*SLR_1D_gradRank(xt,Pr,fftPcVw);
            fprintf("    rank = %i\n",rank);

        %% calculate mean annihilation filter and extract exponentials
            meanAF = zeros(Nf,1);
            for k=1:Nf
                meanAF = meanAF + Vw(:,k);
            end
            expRoots = -log(roots(meanAF));
            expBasis = zeros(Nf-1,Nf-1);
            for n=1:Nf-1
                for k=1:Nf-1
                    expBasis(n,k) = exp(-expRoots(k)*x(n));
                end
            end
            xt_lnls = xt(1:Nf-1);
            expCoeffs = lsqlin(expBasis,xt_lnls,[],[]);
%             b_short = zeros(Fsz-1,1);
%             m = 0;
%             for n=1:Fsz-1
%                 m = m+1;
%                 while(b(m) == 0)
%                     m = m+1;
%                 end
%                 b_short(n) = b(m);
%                 for k=1:Fsz-1
%                     expBasis(n,k) = mask(m).*exp(-expRoots(k)*x(m));
%                 end
%             end
%             expCoeffs = lsqlin(expBasis,b_short,[],[]);
            
        %% calculate CG_init
            cg_init = zeros(N,1);
            for n=1:N
                for k=1:Nf-1
                     cg_init(n) = cg_init(n) +  expCoeffs(k)*exp(-expRoots(k)*x(n));
                end
            end
            
        %% solve least squares
            lhs = @(x) (x.*mask + lambda*SLR_1D_gradRank(x,Pr,fftPcVw));%gradRank is the gradient of the rank function
            [xt,flag,relres,iter,resvec] = cgs(lhs,b,1e-8,100,[],[],xt);%solve least-squares by CG
            if(flag ~= 0)%if CG exited unsuccessfully, return CG information
                disp("        CG routine failed to exit successfully!");
                fprintf("        error flag = %i, last good CG cycle = %i, RELRES = %i\n",flag,iter,relres);
            end
            xw = fft(xt);%recon spectrum
            xw_its(:,i) = xw;%save recon of each SLR iteration
    end
%% ***END OF ALGORITHM***

%% plot recon iterations
    figure('Name','Recon Iterates');
    for i=1:size(xw_its,2)
        displayName = sprintf("Recon Iter %i",i);
        plot(dx,real(xw_its(:,i)),'DisplayName',displayName,'LineWidth',1.5);
        hold on;
    end
    ylabel('Spectrum (Arb. Units)'); xlabel('Frequency (2\pirad/s)'); set(gca,'fontsize',15);
    legend('show');
    
%% plot response comparison
    figure('Name','Response');
    plot(dx,real(R_true),'DisplayName','True','LineWidth',1.5);
    hold on;
    plot(dx,real(R_noisy),'DisplayName','Noisy','LineWidth',1.5);
    hold on;
    plot(dx,real(R_US),'DisplayName',USstringName,'LineWidth',1.5);
    hold on;
    plot(dx,real(xt),'DisplayName','Final Recon','LineWidth',1.5);
    hold off;
    ylabel('Response (Arb. Units)'); xlabel('Time (s)'); set(gca,'fontsize',15);
    legend('show');
    
%% plot spectra comparison
    figure('Name','Spectra Comparison');
    plot(dx,real(Spec_true),'DisplayName','True','LineWidth',1.5);
    hold on;
    plot(dx,real(Spec_noisy),'DisplayName','Noisy','LineWidth',1.5);
    hold on;
    plot(dx,real(fft(R_US)),'DisplayName',USstringName,'LineWidth',1.5);
    hold on;
    plot(dx,real(xw),'DisplayName','Final Recon','LineWidth',1.5);
    hold off;
    ylabel('Intensity (Arb. Units)'); xlabel('Frequency (2\pirad/s)'); set(gca,'fontsize',15);
    legend('show');

%% extraneous code....
%% create circulant matrix from fft eigen basis CircFFT = zeros(N,N);
    % diagS = diag(Spec_true);
    % CircFFT = zeros(N,N);
    % for i=1:N
    %    v = zeros(N,1);
    %    v(i) = 1;
    %    CircFFT(:,i) = ifft(diagS*fft(v));
    % end