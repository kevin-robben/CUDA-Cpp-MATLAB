close all;
clear all;
set(0,'DefaultFigureWindowStyle','docked');

%% initialize SLR parameters
    dims = [512];%signal size (pixels)
    N = prod(dims);
    Ndims = numel(dims);
    N_SLR_its = 30;%number of SLR iterations
    fDims = [12];%filter size
    Nf = prod(fDims);
    SNR = 50;%signal-to-noise ratio
    lambda = 0.2;%lagrange multiplier (use ~1 for 2X US, ~0.2 for 20X US)
    US = 20;%undersample factor

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
    R_true = R_true;
    Spec_true = fft(R_true);

%% synthesize noise and add itN_SLR to response
    %load('noise1.mat');
    noise = (max(R_true)/SNR)*randn(size(R_true)) + (max(R_true)/SNR)*1i*randn(size(R_true));
    R_noisy = R_true + noise;
    Spec_noisy = fft(R_noisy);

%% generate a sampling mask and make undersampled response
    %load('mask1.mat');
    mask = makeMask(N,US);    
    R_US = R_noisy.*mask;
    USstringName = sprintf("Noisy + %iX US",US);

%% pinch time domain
%     R_US(dims(1)-10:dims(1)) = 0;
%     mask(dims(1)-10:dims(1)) = 1;

%% calculate b vector and initialzie time domain recon vector (xt)
    b = complex(R_US);
    xt = complex(R_US);

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
%% initialize parameters for Teoplitz and SVD
    filterCI = IV2CI1(fDims,dims);
    winDims = dims - fDims + 1;
    TeopRowSize = prod(winDims);
    TeopColSize = prod(fDims);
    windowMap = zeros(winDims);
%% calculate window map
    for n=1:numel(windowMap)
        IV = CI2IV1(n,winDims);
        windowMap(n) = IV2CI1(IV,dims);
    end
%% for-loop over SLR iterations
    for i=1:N_SLR_its
        fprintf("SLR iteration: %i\n",i);
        %% compute right-sided SVD of toeplitz (i.e. V'*S^2*V = V'*D*V = H = Teop'*Teop)
            H = zeros(TeopColSize,TeopColSize);%H = Teop'*Teop
            for col=1:TeopColSize
                colShiftIV = CI2IV1(col,fDims);
                colShiftCI = IV2CI1(colShiftIV,dims)-1;% -1 due to 1-based indexing
                for row=1:TeopColSize
                    rowShiftIV = CI2IV1(row,fDims);
                    rowShiftCI = IV2CI1(rowShiftIV,dims)-1;% -1 due to 1-based indexing
                    for n=1:TeopRowSize
                        H(row,col) = H(row,col) + conj(xt(filterCI + (windowMap(n)-1) - rowShiftCI))*xt(filterCI + (windowMap(n)-1) - colShiftCI);
                    end
                end
            end
            [V,D] = eig(H);
            S = sqrt(D);
            maxS = max(diag(S));
            minS = min(diag(S));
            
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
%             meanAF = zeros(Nf,1);
%             for k=1:Nf
%                 meanAF = meanAF + Vw(:,k);
%             end
%             expRoots = -log(roots(meanAF));
%             expBasis = zeros(Nf-1,Nf-1);
%             for n=1:Nf-1
%                 for k=1:Nf-1
%                     expBasis(n,k) = exp(-expRoots(k)*x(n));
%                 end
%             end
%             xt_lnls = xt(1:Nf-1);
%             expCoeffs = lsqlin(expBasis,xt_lnls,[],[]);
            
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
    
    
%% compute right-sided SVD of toeplitz (i.e. V'*S^2*V = V'*D*V = H = Teop'*Teop)
    H = zeros(TeopColSize,TeopColSize);%H = Teop'*Teop
    for col=1:TeopColSize
        colShiftIV = CI2IV1(col,fDims);
        colShiftCI = IV2CI1(colShiftIV,dims)-1;% -1 due to 1-based indexing
        for row=1:TeopColSize
            rowShiftIV = CI2IV1(row,fDims);
            rowShiftCI = IV2CI1(rowShiftIV,dims)-1;% -1 due to 1-based indexing
            for n=1:TeopRowSize
                H(row,col) = H(row,col) + conj(xt(filterCI + (windowMap(n)-1) - rowShiftCI))*xt(filterCI + (windowMap(n)-1) - colShiftCI);
            end
        end
    end
    [V,D] = eig(H);
    S = sqrt(D);
    maxS = max(diag(S));
    minS = min(diag(S));

%% compute weighted V space
    eps = sqrt(maxS*(minS+((1e-12)*maxS)))/(1+sqrt(2));%calculate epsilon in rank function
        %note: the term "+((1e-12)*S(1,1)))" is added just in case
        %min(S) is so small that eps calculation would be poorly
        %conditioned
    Nullw = diag(S);
    Nullw = 1./(Nullw+eps);
    Vw = V*diag(Nullw);%weighted V space
            
    meanAF = zeros(Nf,1);
    for k=1:Nf
        meanAF = meanAF + Vw(:,k);
    end
    expRoots = -log(roots(meanAF));
    expBasis = zeros(N,Nf-1);
    for n=1:N
        for k=1:Nf-1
            expBasis(n,k) = exp(-expRoots(k)*x(n));
            if(real(-expRoots(k)) > 0 )
                expBasis(n,k) = 1e-5;
            end
        end
    end
    Jacobian = expBasis'*expBasis;
    expNorms = real(sqrt(diag(Jacobian)));
    expBasis = expBasis*diag(1./expNorms);
    Jacobian = expBasis'*expBasis;
    [E,L2] = eig(Jacobian);
    l2 = diag(L2);
    l1 = real(sqrt(l2));
    l1_inv = diag(1./(l1+(1e-10)*max(l1)));
    W = expBasis*E*l1_inv;
    c = W'*b;
    A_recon = E*diag(l1)*c;
    %expCoeffs = lsqlin(expBasis,xt_lnls,[],[]);

%% extraneous code....
