%% GIRAF CUDA CODE
%by Robben and Humston

% To compile matlabGIRAF.cu, run the following command:
% mexcuda matlabGIRAF.cu -lcublas -lcufft -lcusolver -L"/opt/cuda/lib64" -v NVCC_FLAGS="-ccbin /opt/rh/devtoolset-3/root/bin"
% Note that the cuda library link path (-L"...") and the c++ compiler path (NVCC_FLAGS="-ccbin ...") are both platform dependent.
% If the default c++ compiler on the computer is depreciated enough to be compatible with cuda, then the
% NVCC_FLAGS="-ccbin ..." command is unnecessary. Be sure that matlab's working directory is under the same
% folder as matlabGIRAF.cu when compiling.

clear all;close all;
%% Things to Change
lambda0 = 1.0e-2; %Larger lambda0 puts more weight on the data and less on the model
del = 10;
LSsolver = 'CG';%'CG' or 'ADMM'
USSet = [2]; %Undersampling factor
tic;
for us = 1 : length(USSet); %Loop over multiple undersampling factors
    US = USSet(us);
    
    topFolder = '/mnt/cifs/rdss_ccheatum/humstonj/CS-2DIR/GIRAF/Paper2/FDH_phil/Data';
    bottomFolder = sprintf('FDH_phil_comp%i',US); %Folder containing tvsw files
    subcodesFolder = './SubCodes/';
    addpath('./SubCodes');
    %     cudaGIRAFfolder = '/space/robben/git/CBIGR_cuda/CBIGR_cuda/CBIGR_cuda/';
    
    %% Loading Parameters
    numCycles = 20;%number of iterations to run in GIRAF routine
    filterDims = [21,21]; %filter dimensions (use odd nums)
    Nw3 = 1024; Nw1 = 1024; Nw = 1; %Size of final spectrum
    N = 167; %Number of tau times without compressing
    t = 24; %Time step in fs
    % Truncate in spectral domain to remove noise from ends
    w3_Trunc_low=280;w3_Trunc_high=535; %Result must be an even length vector, factor of 2 best
    t3_lowpass_size = length(w3_Trunc_low:w3_Trunc_high);
    
    %% Load directory of data
    dataFolder = fullfile(topFolder,bottomFolder);%Full path to the folder containing data
    tvswFilePattern = fullfile(dataFolder,'tvsw*'); %Select the tvsw data from the foler
    thetvswFiles = dir(tvswFilePattern); %Structure containing list of all the files
    
    %% Load Mask
    maskName = sprintf('mask%i_%i.mat',N,US); %Mask should be in same folder as the data
    maskFolder = fullfile(dataFolder,maskName);
    %s2=load(maskFolder);
    load(maskFolder);
    tau_ind = find(s2); %Find the indices of the tau times used
    tau = (tau_ind-1) .* t; %Converts from index to actual time
    Numtau = length(tau); %The number of tau times measured
    
    mask_line = zeros(1,Nw1); mask_line(1:N) = s2; %Create full mask for zeropadded data
    
    %% Load data
    for x = 1:1%length(thetvswFiles)
        str = sprintf('%d / %d',x,length(thetvswFiles));
        disp(str);
        tvswBaseFileName = thetvswFiles(x).name;
        tvswFullFileName = fullfile(dataFolder,tvswBaseFileName);
        
        place = findstr('.',tvswBaseFileName);
        AlphaTw = tvswBaseFileName(5:place-1); %Extract basename (trial and waiting time) from filename
        
        data= load(tvswFullFileName); %variable is named 'data' and has correct orientation
        data = data';
        load(tvswFullFileName);
        data(isnan(data)) = 0; data(isinf(data)) = 0;
        
        %% Convert from nonlinear w3 to linearized w3 data
        calibPath = fullfile(dataFolder,'ArrayCalibConsts_Shiftfreq_Interval.calib');
        n = load(calibPath);
        
        run(strcat(subcodesFolder,'CreateAxesAll.m'));
        run(strcat(subcodesFolder,'LinearGriddedData.m'));
        %% Place CS data into matrix the size of full spectrum grid size
        datatw = zeros(Nw3,Nw1);
        for a = 1: Numtau
            datatw(:,tau_ind(a)) = data(:,a);
        end
        %% Convert to time-time data
        data_trunc = datatw(w3_Trunc_low:w3_Trunc_high,:);
        datatt = ifft(data_trunc,[],1); %invserse fft from w3 to t3 to get t-t data
        dim = size(datatt);
        datatt_lowpass = datatt;
        datatt_lowpass(t3_lowpass_size:(dim(1)-t3_lowpass_size),:) = 0;
        
        %% Repmat mask
        mask = repmat(mask_line,dim(1),1);
        mask(t3_lowpass_size:dim(1)-t3_lowpass_size,:) = 0;

        %% Pre-Process Data by Padding
        pad_size = 1;
        datatt_padded = zeros(dim(1)*pad_size,dim(2)*pad_size);
        datatt_padded(1:dim(1)/2,1:dim(2)/2)=datatt_lowpass(1:dim(1)/2,1:dim(2)/2);
        datatt_padded(dim(1)*pad_size-dim(1)/2+1:dim(1)*pad_size,1:dim(2)/2)=datatt_lowpass(dim(1)/2+1:dim(1),1:dim(2)/2);
        %% Upload GPU arrays
        dataGPU = gpuArray(datatt_padded);
        filterGPU = complex(zeros(filterDims,'gpuArray'),zeros(filterDims,'gpuArray'));
        maskGPU = complex(gpuArray(mask),zeros(size(mask),'gpuArray'));
        
        %% Run GIRAF reconstruction
        cost = matlabGIRAF(dataGPU,filterGPU,maskGPU,numCycles,lambda0,LSsolver,del);
        clear ans i a
        
        %% Download GPU arrays
        Recon_tt = gather(dataGPU);
        filter = gather(filterGPU);
        
        %% Begin post-GIRAF reconstruction analysis
        Recon_tt(dim(1)/2+1:dim(1)*pad_size-dim(1)/2,:) = []; %Remove pad along t3
        Recon_tt(:,dim(2)/2+1:dim(2)*pad_size-dim(2)/2) = []; %Remove pad along t1
        Recon_ww = fft2(Recon_tt);
        Recon_ww = Recon_ww(:,1:512);
        
        figure; contourf(real(Recon_ww(:,1:512)),20);
        %figure; plot(Recon_tt_r(1,:))
        
        %% Analysis Calculate the centerline slope
        w3_low=110; w3_high=190; w1_low=170; w1_high=200;
        zoom = real(Recon_ww(w3_low:w3_high,w1_low:w1_high)); %w3 axis comes first because it is rows then columns. The range inputs comes from CreateAxes.m
        
        figure; contourf(zoom,20);
        %figure; plot(Recon_tt_r(1,:))
        
        [M,I] = min(zoom); %Taking minimum for each w1 slice (minimum value from each column matrix)
        [m,i] = min(M); %i is a scalar, the location (along w1) of the center of the peak
        
        centpeak = i + w1_low;
        centpeakcm = n(6)+((centpeak)/(3*1e-5*1024*n(7))); %Check position of the 'peak' to make sure it is the desired
        
        for l = i - 2: i + 2
            temp = zoom(:,l); %A slice along w1
            temp = movingmean(temp,15);% Taking a moving mean to smooth the daClusterta
            [peakloc,peakmag] = peakfinder(temp,(max(temp)-min(temp))/2,0,-1,true,true); %finds location (index) in w3 of peak for the lth w1 slice
            [y,Y] = min(peakmag); truepeak = peakloc(Y); %Finds the largest peak
            slope(l-i+3,1) = l; slope(l-i+3,2) = truepeak; %Saves cls data, column1 is w1 (in data pt) and column 2 is w3 peak location (in data pt)
        end
        
        % Convert point number to wavenumber
        w1 = slope(:,1) + w1_low; %w1 is column 1, and adding in the w1 offset, in order to convert from data pt to cm-1
        w1 = n(6)+((w1)/(3*1e-5*1024*n(7))); %Converts w1 values from data pt to cm-1.
        
        w3 = slope(:,2) + w3_low + w3_Trunc_low; %w3 is column 2, and adding in w3 offset in order to convert from data pt to cm-1
        w3_linefit = polyfit(1:length(w3_axisL),w3_axisL,1); %convert w3 linear axis to a function
        w3 = w3*w3_linefit(1) + w3_linefit(2);
        %Fit array is now two columns, w1 and corresponding peak location along w3.
        
        clsfit = polyfit(w1,w3,1); %Fit to a line
        cls = clsfit(1);
        
        %% Save
        Recon = './Recon_Saved/';
        USstring = num2str(US);
        Specname = strcat(Recon,'US',USstring,'_',AlphaTw,'.mat');
        save(Specname,'Recon_ww','AlphaTw','US','cls');
    end
end
toc
