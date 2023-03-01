#ifndef __CUDATA_CU__
#define __CUDATA_CU__
#include "cudata.cuh"
/***cuda kernels***/
__global__ void cuDoubleLinear(unsigned long numPixels, double *buffer, double x0, double dx)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPixels)
	{
		buffer[i] = x0 + i*dx;
	}
}
__global__ void cuewMultiply(unsigned long numPixels, cuDoubleComplex* out, cuDoubleComplex a, cuDoubleComplex* in1,cuDoubleComplex* in2)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	cuDoubleComplex w;
	if(i<numPixels)
	{
		w.x = in1[i].x*in2[i].x - in1[i].y*in2[i].y;
		w.y = in1[i].x*in2[i].y + in1[i].y*in2[i].x;
		out[i].x = w.x*a.x - w.y*a.y;
		out[i].y = w.x*a.y + w.y*a.x;
	}
}
__global__ void cuAbsSq(unsigned long numPixels, cuDoubleComplex *bufferIn, cuDoubleComplex *bufferOut)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPixels)
	{
		bufferOut[i].x = bufferIn[i].x*bufferIn[i].x + bufferIn[i].y*bufferIn[i].y;
		bufferOut[i].y = 0.0;
	}
}
__global__ void cu_sqrt(unsigned long numPixels, cuDoubleComplex *in, cuDoubleComplex *out)
{
	//returns root in 1st or 4th quadrant
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPixels)
	{
		double r = sqrt(in[i].x*in[i].x+in[i].y*in[i].y);
		out[i].x = sqrt( ( r + in[i].x ) / 2 );
		if(out[i].y >= 0)
			out[i].y = sqrt( ( r - in[i].x ) / 2 );
		else
			out[i].y = -sqrt( ( r - in[i].x ) / 2 );
	}
}
__global__ void cuLorentzian(unsigned long numPixels, unsigned long numDims, unsigned long *dims, double *FWHM, double *center, double *amplitude, cuDoubleComplex *L)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPixels)
	{
		unsigned long *v = new unsigned long[numDims];
		v = RI2IV(i,v,dims,numDims);
		L[i].x = 1.0;
		L[i].y = 0.0;
		for(int k=0; k<numDims; k++)
		{
			v[i] -= v[i]/2;
			L[i].x *= amplitude[i] / (3.1415928*(FWHM[k]/2.0)*(1.0 + pow((v[k] - center[k])/(FWHM[k]/2.0),2.0)));
		}
		delete[] v;
	}
}
__global__ void cu_AZ_add_BZ(unsigned long size, cuDoubleComplex *out, cuDoubleComplex *A, cuDoubleComplex *B)
{
	/*out = A - B*/
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<size)
	{
		out[i].x = A[i].x + B[i].x;
		out[i].y = A[i].y + B[i].y;
	}
}
__global__ void cu_AZ_add_bd_BZ(unsigned long size, cuDoubleComplex *out, cuDoubleComplex *A, double b, cuDoubleComplex *B)
{
	/*out = A - B*/
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<size)
	{
		out[i].x = A[i].x + b*B[i].x;
		out[i].y = A[i].y + b*B[i].y;
	}
}
__global__ void cuDoubleTocuComplex(unsigned long numPixels, double *in, cuDoubleComplex *out)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPixels)
	{
		out[i].x = in[i];
		out[i].y = 0;
	}
}
__global__ void cu_AZ_minus_BZ(unsigned long size, cuDoubleComplex *out, cuDoubleComplex *A, cuDoubleComplex *B)
{
	/*out = A - B*/
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<size)
	{
		out[i].x = A[i].x - B[i].x;
		out[i].y = A[i].y - B[i].y;
	}
}
__global__ void cu_AZ_minus_bZ_BZ(unsigned long size, cuDoubleComplex *out, cuDoubleComplex *A, cuDoubleComplex b, cuDoubleComplex *B)
{
	/*out = A - b * B*/
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<size)
	{
		out[i].x = A[i].x - b.x * B[i].x + b.y * B[i].y;
		out[i].y = A[i].y - b.x * B[i].y - b.y * B[i].x;
	}
}
__global__ void cu_AZ_multiply_BZ(unsigned long size, cuDoubleComplex *out, cuDoubleComplex *A, cuDoubleComplex *B)
{
	/*out = A * B*/
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<size)
	{
		out[i].x = A[i].x * B[i].x - A[i].y * B[i].y;
		out[i].y = A[i].x * B[i].y + A[i].y * B[i].x;
	}
}
__global__ void cu_this_multiply_BZ(unsigned long size, cuDoubleComplex *out, cuDoubleComplex *B)
{
	/*out = out * A*/
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<size)
	{
		out[i].x = out[i].x * B[i].x - out[i].y * B[i].y;
		out[i].y = out[i].x * B[i].y + out[i].y * B[i].x;
	}
}
__global__ void cu_this_dscaled(unsigned long size, cuDoubleComplex *out, double d)
{
	/*out *= C*/
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<size)
	{
		out[i].x *= d;
		out[i].y *= d;
	}
}
/****The following kernels are outdated and need to be rewritten to use the current dynamic dimensions array****/
__global__ void cuCanvasCutComplex(unsigned long numCutPixels, unsigned long numDims, unsigned long *cutDims, cuDoubleComplex *cutBuffer, unsigned long *cutIndxStart, unsigned long *dims, cuDoubleComplex *buffer)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numCutPixels)
	{
		unsigned long *IV = new unsigned long[numDims];
		unsigned long *cutIV = new unsigned long[numDims];
		cutIV = RI2IV(i,cutIV,cutDims,numDims);
		for(int k=0; k<numDims; k++)
			IV[k] = cutIndxStart[k] + cutIV[k];
		cutBuffer[i].x = buffer[IV2RI(IV,dims,numDims)].x;
		cutBuffer[i].y = buffer[IV2RI(IV,dims,numDims)].y;
		delete[] IV;
		delete[] cutIV;
	}
}
__global__ void cuCanvasCutDouble(unsigned long numCutPixels, unsigned long numDims, unsigned long *cutDims, double *cutBuffer, unsigned long *cutIndxStart, unsigned long *dims, double *buffer)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numCutPixels)
	{
		unsigned long *IV = new unsigned long[numDims];
		unsigned long *cutIV = new unsigned long[numDims];
		cutIV = RI2IV(i,cutIV,cutDims,numDims);
		for(int k=0; k<numDims; k++)
			IV[k] = cutIndxStart[k] + cutIV[k];
		cutBuffer[i] = buffer[IV2RI(IV,dims,numDims)];
		delete[] IV;
		delete[] cutIV;
	}
}
__global__ void cuContiguousCut(unsigned long numCutPixels, unsigned long numDims, unsigned long *cutDims, cuDoubleComplex *cutBuffer, unsigned long *cutIndxStart, unsigned long *dims, cuDoubleComplex *buffer)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numCutPixels)
	{
		unsigned long *IV = new unsigned long[numDims];
		unsigned long *cutIV = new unsigned long[numDims];
		cutIV = RI2IV(i,cutIV,cutDims,numDims);
		for(int k=0; k<numDims; k++)
		{
			if(cutIV[k] < cutIndxStart[k])//if before cut
				IV[k] = cutIV[k];
			else//else, must be after cut
				IV[k] = cutIV[k] + (dims[k] - cutDims[k]);
		}
		cutBuffer[i].x = buffer[IV2RI(IV,dims,numDims)].x;
		cutBuffer[i].y = buffer[IV2RI(IV,dims,numDims)].y;
		delete[] IV;
		delete[] cutIV;
	}
}
__global__ void cuContiguousPad(unsigned long paddedNumPixels, unsigned long numDims, cuDoubleComplex *unpaddedBuffer, cuDoubleComplex *paddedBuffer, cuDoubleComplex padVal, unsigned long *startPadIndx, unsigned long *unpaddedDims, unsigned long *paddedDims, unsigned long *padSize)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<paddedNumPixels)
	{
		unsigned long *unpaddedIV = new unsigned long[numDims];
		unsigned long *paddedIV = new unsigned long[numDims];
		paddedIV = RI2IV(i,paddedIV,paddedDims,numDims);
		/* for-loop over each dimension*/
		for(int k=0; k<numDims; k++)
		{
			if(paddedIV[k]<startPadIndx[k])//if before padding zone
				unpaddedIV[k] = paddedIV[k];
			else if(paddedIV[k]<(startPadIndx[k]+padSize[k]))//else, if in padding zone
			{
				paddedBuffer[i].x = padVal.x;
				paddedBuffer[i].y = padVal.y;
				delete[] unpaddedIV;
				delete[] paddedIV;
				return;
			}
			else//else, we must be past the padding zone
				unpaddedIV[k] = paddedIV[k] - padSize[k];
		}
		paddedBuffer[i].x = unpaddedBuffer[IV2RI(unpaddedIV,unpaddedDims,numDims)].x;
		paddedBuffer[i].y = unpaddedBuffer[IV2RI(unpaddedIV,unpaddedDims,numDims)].y;
		delete[] unpaddedIV;
		delete[] paddedIV;
	}
}
__global__ void cuCanvasPad(unsigned long numPaddedPixels, unsigned long numDims, cuDoubleComplex *unpaddedBuffer, cuDoubleComplex *paddedBuffer, cuDoubleComplex padVal, unsigned long *padIndxStart, unsigned long *unpaddedDims, unsigned long *paddedDims, unsigned long *padSize)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPaddedPixels)
	{
		unsigned long *unpaddedIV = new unsigned long[numDims];
		unsigned long *paddedIV = new unsigned long[numDims];
		paddedIV = RI2IV(i,paddedIV,paddedDims,numDims);
		/* for-loop over each dimension*/
		for(int k=0; k<numDims; k++)
		{
			if(paddedIV[k]<padIndxStart[k])//if in the lower padding zone
			{
				paddedBuffer[i].x = padVal.x;
				paddedBuffer[i].y = padVal.y;
				delete[] unpaddedIV;
				delete[] paddedIV;
				return;
			}
			else if(paddedIV[k]<(padIndxStart[k]+padSize[k]))//else, if in the pasting zone
			{
				unpaddedIV[k] = paddedIV[k] - padSize[k];
			}
			else//else, we must be in the higher padding zone
			{
				paddedBuffer[i].x = padVal.x;
				paddedBuffer[i].y = padVal.y;
				delete[] unpaddedIV;
				delete[] paddedIV;
				return;
			}
		}
		paddedBuffer[i].x = unpaddedBuffer[IV2RI(unpaddedIV,unpaddedDims,numDims)].x;
		paddedBuffer[i].y = unpaddedBuffer[IV2RI(unpaddedIV,unpaddedDims,numDims)].y;
		delete[] unpaddedIV;
		delete[] paddedIV;
	}
}
__global__ void cufft_shift(unsigned long numPixels, cuDoubleComplex *unShiftedBuffer, cuDoubleComplex *shiftedBuffer, unsigned long *dims, bool forward)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPixels)
	{

	}
}
__global__ void cuSetConst(unsigned long numPixels, cuDoubleComplex *buffer, double real, double imag)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPixels)
	{
		buffer[i].x = real;
		buffer[i].y = imag;
	}
}
/***************************************************************************************************************/



/****CUFFT****/
 cufft_struct::~cufft_struct()
{
	delete[] n;
	delete[] inembed;
	delete[] onembed;
	cufftDestroy(cufft_handle);
}
cufft_struct::cufft_struct(unsigned long *dims, int numDims)
{
	idist = 1;
	odist = 1;
	batch = 1;
	rank = numDims;
	n = new int[rank];
	for(int i=0; i<rank; i++)
	{
		n[i] = dims[i];
		idist *= dims[i];
		odist *= dims[i];
	}
}
cufft_struct::cufft_struct(unsigned long *dims, int numDims, int batchDim)
{
	/*refer to Section 2.6 (Advanced Data Layout) & 3.2.4 (Function cufftPlanMany()) of cuFFT Library Guide*/
	if(batchDim == 0 && numDims<=4)//batch dim is least consecutive, making this the easiest format for cufftPlanMany()
	{
		idist = 1;
		odist = 1;
		batch = dims[0];
		rank = (int)(numDims-1);
		n = new int[rank];
		for(int i=1; i<numDims; i++)
		{
			n[i-1] = dims[i];
			idist *= dims[i];
			odist *= dims[i];
		}
	}
	else if(numDims==2  && batchDim==1)//batch dim is most consecutive, and cufftPlanMany() becomes more complicated
	{
		batch = dims[1];
		idist = 1;
		odist = 1;
		istride = dims[1];
		ostride = dims[1];
		rank = (int)(numDims-1);
		n = new int[rank];
		inembed = new int[rank];
		onembed = new int[rank];
		inembed[0] = 1;//dims[0];//this is just for the sake of lower dimension example... Section 2.6 of cuFFT shows how inembed[0] is actually replaced by idist for indexing
		onembed[0] = 1;//dims[0];//this is just for the sake of lower dimension example... Section 2.6 of cuFFT shows how inembed[0] is actually replaced by odist for indexing
		n[0] = dims[0];
	}
	else
		throw runtime_error("cufft_struct not configured to have this combination of numDimsPlus1 and batchDim");
}
void cufft_struct::plan()
{
	cudaCheck("cufftPlanMany",(cufft_result = cufftPlanMany(&(cufft_handle),rank,n,inembed,istride,idist,onembed,ostride,odist,CUFFT_Z2Z,batch)));
}

/****CUSOLVER****/
cusolver_struct::~cusolver_struct()
{
	cudaFree(cuW);
	cudaFree(cuWork);
	cudaFree(cuInfo);
	cusolverDnDestroy(cusolver_DnHandle);
}

/*****cudata functions*****/
//dgmm inplace
template <>
void cudata<cuDoubleComplex>::dgmm(cublasHandle_t cublas_handle, cuDoubleComplex* diagonal, cudata &out)
{
	cudaCheck("cudata::dgmm",cublasZdgmm(cublas_handle,CUBLAS_SIDE_RIGHT,dims[0],dims[0],this->buffer,dims[0],diagonal,1,out.buffer,dims[0]));
}
/****    CONSTRUCTORS    ****/
//read from file constructor - specialized for cuDoubleComplex
template <>
cudata<cuDoubleComplex>::cudata(string filename)
{
	/*because numPixels is defined to be the product of all dimensions (including those unused), all elements of dims must be of size 1, at the very least*/

	/*open file and read in first line of file (data dimensions) into a string*/
		ifstream ifs(filename);//initiate input file stream
		if(ifs.fail()==true)
			throw runtime_error("A logical error occurred while trying to open data file - this file may not exist in the current working directory, or the working directory may have unexpectedly changed");
		string str;
		getline(ifs,str);//scan in first line of file (which should be the data dimensions) into str
		ifs >> ws;//move file stream pointer past any white space
		if(ifs.bad()==true)
			throw runtime_error("A read/writing error occurred while trying to scan in dimensions from data file");
		else if(ifs.fail()==true)
			throw runtime_error("A logical error occurred while trying to scan in dimensions from data file");
		else if(ifs.eof()==true)
			throw runtime_error("End of file occurred after trying to scan in dimensions of the input file - this file appears to be empty");
	/*FIRST ROW OF FILE: convert data dimensions string into (numeric) dims array*/
		string tempStr;
		unsigned long temp;
		/*first determine number of dimensions and pixels*/
			stringstream ss(str);//copy str (a string) into ss (a stringstream)
			int i=0;
			while(!ss.eof())//scan in dimensions until the end of the string is reached
			{
				if(i >= MAX_DIM)
					throw runtime_error("Number of dimensions of data file exceeds program ability");
				/*read in next dimension into dims array*/
					ss >> tempStr;
					temp = stod(tempStr);
				if(temp > 1)
					numDims++;
				i++;
				ss >> ws;
			}
			ss.clear();//must clear to reuse
		/*now allocate memory for dims and scan in dimensions*/
			dims = new unsigned long [numDims];
			ss.str(str);
			numPixels = 1;
			for(int i=0; i<numDims; i++)//scan in dimensions until the end of the string is reached
			{
				ss >> tempStr;
				dims[i] = stod(tempStr);
				if(dims[i] > 1)
					numPixels *= dims[i];
				else if(dims == 0)
					dims[i] = 1;//user is an idiot
				ss >> ws;
			}
			ss.clear();//must clear to reuse
	/*REMAINING ROWS OF FILE: read in data from file and save to cudata buffer*/
		complex<double> *hostBuffer = new complex<double> [numPixels];//hold data on CPU for a moment - will transfer to GPU shortly
		ifs >> ws;
		unsigned long CI=0, RI;//indexing variables
		bool nanFlag = false, infFlag = false;
		while(ifs.good()==true)
		{
			getline(ifs,str);//scan next line of data into string (str)
			ss.str(str);//transfer string of data to a stringstream (ss)
			ss >> ws;
			while(ss.eof()==false)//continue reading stringstream until the end of stream is reached
			{
				if(ss.good()==false)//make sure stringstream is good to read
				{
					throw runtime_error("An error occurred while reading data file at string stream ss(str)");
				}
				if(CI >= numPixels)//make sure data in file isn't longer than expected
				{
					throw runtime_error("Input data file has more pixels than dimensions initially defined!");
				}
				ss >> tempStr;//read to next white-space character
				/*data is assumed column major in file, but we need to store data as row major in cudata buffers for optimal performance with cuda libraries*/
					/*convert column-major index to row-major index*/
						RI = CI2RI(CI,dims,numDims);
					/*read in datum from file into host buffer using string to double function and row-major index*/
						hostBuffer[RI] = stod(tempStr);
				if(std::isnan(hostBuffer[RI].real()))
				{
					hostBuffer[RI] = 0;
					if(nanFlag == false)
					{
						nanFlag = true;
						//perror("At least one instance of NAN was encountered in data file - this was interpreted as a zero-valued pixel(s)");
					}
				}
				else if(std::isinf(hostBuffer[RI].real()))
				{
					hostBuffer[RI] = 0;
					if(infFlag == false)
					{
						infFlag = true;
						//perror("At least one instance of (+/-)INF was encountered in data file - this was interpreted as a zero-valued pixel(s)");
					}
				}
				CI++;
				ss >> ws;
			}
			ss.clear();//clear to reuse
			ifs >> ws;
		}
		if(ifs.fail()==true)//make sure nothing bad happened while reading from input file
			throw runtime_error("An error occurred while reading the data file");
		else if(ifs.eof() && CI < numPixels)
			throw runtime_error("Data file has fewer pixels than dimensions initially defined!");
	//allocate GPU memory for device buffer
		cudaCheck("constructor cudaMalloc in cudata(filename) for cudata.buffer",cudaMalloc((void**)&buffer,numPixels*sizeof(cuDoubleComplex)));
	//transfer buffer data from host to device
		cudaCheck("constructor cudaMemcpy in cudata(filename) for cudata.buffer",cudaMemcpy(buffer,hostBuffer,numPixels*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice));
	//allocate GPU memory for device version of dims array
		cudaCheck("constructor cudaMalloc in cudata(filename) for cudata.cuDims",cudaMalloc((void**)&cuDims,numDims*sizeof(unsigned long)));
		updateCuDims();
	delete[] hostBuffer;
}
//MATLAB mexcuda constructor
template <>
cudata<cuDoubleComplex>::cudata(mxArray const * const mp)
{
	//initialize mxGPUArray object from mp
		mxGPUArray *mexIn = (mxGPUArray *)mxGPUCreateFromMxArray(mp);
	//copy GPUArray pointer to buffer pointer
		buffer = (cuDoubleComplex*)(mxGPUGetData(mexIn));
	//copy number of pixels from mxGPUArray object
		numPixels = (unsigned long)mxGPUGetNumberOfElements(mexIn);
	//copy number of dimensions from mxGPUArray object
		numDims = (unsigned long)mxGPUGetNumberOfDimensions(mexIn);
	//initialize dims array from mxGPUArray object
		mwSize const *mexDims = mxGPUGetDimensions(mexIn);
	//initialize dims array
		dims = new unsigned long [numDims];
	//copy over dims array
		for(int i=0; i<numDims; i++)
			dims[i] = (unsigned long)mexDims[numDims - 1 - i];
		cudaCheck("cudaMalloc in MATLAB mexcuda constructor for cuDims",cudaMalloc((void**)&cuDims,numDims*sizeof(cuDoubleComplex)));
		updateCuDims();
		mxFree((void *)mexDims);
	//set mxCuda flag
		isMexCuda = true;
	//destroy 
		mxGPUDestroyGPUArray(mexIn);
}
//deep copy constructor
template <>
cudata<cuDoubleComplex>::cudata(const cudata &obj)
{
	//copy dimensions
		numPixels = obj.numPixels;
		numDims = obj.numDims;
		if(obj.buffer != nullptr)//make sure nullptr doesn't get copied
		{
			cudaCheck("cudaMalloc in deep copy constructor for buffer",cudaMalloc((void**)&buffer,obj.numPixels*sizeof(cuDoubleComplex)));
			cudaCheck("cudaMemcpy buffer in deep copy constructor",cudaMemcpy(buffer,obj.buffer,obj.numPixels*sizeof(cuDoubleComplex),cudaMemcpyDeviceToDevice));
		}
		if(obj.cuDims != nullptr)//make sure nullptr doesn't get copied
			cudaCheck("cudaMalloc in deep copy constructor for cuDims",cudaMalloc((void**)&cuDims,obj.numDims*sizeof(cuDoubleComplex)));
		dims = new unsigned long [numDims];
		for(int i=0; i<obj.numDims; i++)
			dims[i] = obj.dims[i];
		updateCuDims();
	//copy bools and cublas handle
		this->boolCopy(obj);
}

/****RECONSTRUCTORS****/
template <>
void cudata<cuDoubleComplex>::import_real(string filename)
{
	/*open file and read in first line of file (data dimensions) into a string*/
		ifstream ifs(filename);//initiate input file stream
		if(ifs.fail()==true)
			throw runtime_error("A logical error occurred while trying to open data file - this file may not exist in the current working directory, or the working directory may have unexpectedly changed.");
	/*REMAINING ROWS OF FILE: read in data from file and save to cudata buffer*/
		string str, tempStr;
		stringstream ss;
		complex<double> *hostBuffer = new complex<double> [numPixels];//hold data on CPU for a moment - will transfer to GPU shortly
	//transfer buffer data to host from device
		cudaCheck("reconstructor cudaMemcpy in import_real(filename) for hostBuffer",cudaMemcpy(hostBuffer,buffer,numPixels*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost));
		ifs >> ws;
		unsigned long CI=0, RI;//indexing variables
		while(ifs.good()==true)
		{
			getline(ifs,str);//scan next line of data into string (str)
			ss.str(str);//transfer string of data to a stringstream (ss)
			ss >> ws;
			while(ss.eof()==false)//continue reading stringstream until the end of stream is reached
			{
				if(ss.good()==false)//make sure stringstream is good to read
					throw runtime_error("An error occurred while reading data file at string stream ss(str)");
				if(CI >= numPixels)//make sure data in file isn't longer than expected
					throw runtime_error("Input data file has more pixels than dimensions initially defined!");
				ss >> tempStr;//read in next number from string stream
				/*data is assumed column major in file, but we need to store data as row major in cudata buffers for optimal performance with cuda libraries*/
					/*convert column-major index to row-major index*/
						RI = CI2RI(CI,dims,numDims);
					/*read in datum from file into host buffer using string to double function and row-major index*/
						hostBuffer[RI].real(stod(tempStr));
				CI++;
				ss >> ws;
			}
			ss.clear();//clear to reuse
			ifs >> ws;
		}
		if(ifs.fail()==true)//make sure nothing bad happened while reading from input file
			throw runtime_error("An error occurred while reading the data file");
		else if(ifs.eof() && CI < numPixels)
			throw runtime_error("Data file has fewer pixels than dimensions initially defined!");
	//transfer buffer data from host to device
		cudaCheck("reconstructor cudaMemcpy in import_real(filename) for cudata.buffer",cudaMemcpy(buffer,hostBuffer,numPixels*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice));
	delete[] hostBuffer;
}
template <>
void cudata<cuDoubleComplex>::import_imag(string filename)
{
	/*open file and read in first line of file (data dimensions) into a string*/
		ifstream ifs(filename);//initiate input file stream
		if(ifs.fail()==true)
			throw runtime_error("A logical error occurred while trying to open data file - this file may not exist in the current working directory, or the working directory may have unexpectedly changed.");
	/*REMAINING ROWS OF FILE: read in data from file and save to cudata buffer*/
		string str, tempStr;
		stringstream ss;
		complex<double> *hostBuffer = new complex<double> [numPixels];//hold data on CPU for a moment - will transfer to GPU shortly
	//transfer buffer data to host from device
		cudaCheck("reconstructor cudaMemcpy in import_imag(filename) for hostBuffer",cudaMemcpy(hostBuffer,buffer,numPixels*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost));
		ifs >> ws;
		unsigned long CI=0, RI;//indexing variables
		while(ifs.good()==true)
		{
			getline(ifs,str);//scan next line of data into string (str)
			ss.str(str);//transfer string of data to a stringstream (ss)
			ss >> ws;
			while(ss.eof()==false)//continue reading stringstream until the end of stream is reached
			{
				if(ss.good()==false)//make sure stringstream is good to read
					throw runtime_error("An error occurred while reading data file at string stream ss(str)");
				if(CI >= numPixels)//make sure data in file isn't longer than expected
					throw runtime_error("Input data file has more pixels than dimensions initially defined!");
				ss >> tempStr;//read in next number from string stream
				/*data is assumed column major in file, but we need to store data as row major in cudata buffers for optimal performance with cuda libraries*/
					/*convert column-major index to row-major index*/
						RI = CI2RI(CI,dims,numDims);
					/*read in datum from file into host buffer using string to double function and row-major index*/
						hostBuffer[RI].imag(stod(tempStr));
				CI++;
				ss >> ws;
			}
			ss.clear();//clear to reuse
			ifs >> ws;
		}
		if(ifs.fail()==true)//make sure nothing bad happened while reading from input file
			throw runtime_error("An error occurred while reading the data file");
		else if(ifs.eof() && CI < numPixels)
			throw runtime_error("Data file has fewer pixels than dimensions initially defined!");
	//transfer buffer data from host to device
		cudaCheck("reconstructor cudaMemcpy in import_imag(filename) for cudata.buffer",cudaMemcpy(buffer,hostBuffer,numPixels*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice));
	delete[] hostBuffer;
}
/*import matlab complex data*/
template <>
void cudata<cuDoubleComplex>::import_matlab_complex(string filename)
{
	/*because numPixels is defined to be the product of all dimensions (including those unused), all elements of dims must be of size 1, at the very least*/

	/*open file and read in first line of file (data dimensions) into a string*/
		ifstream ifs(filename);//initiate input file stream
		if(ifs.fail()==true)
			throw runtime_error("A logical error occurred while trying to open data file - this file may not exist in the current working directory, or the working directory may have unexpectedly changed.");
		string str;
		getline(ifs,str);//scan in first line of file (which should be the data dimensions) into str
		ifs >> ws;//move file stream pointer past any white space
		if(ifs.bad()==true)
			throw runtime_error("A read/writing error occurred while trying to scan in dimensions from data file");
		else if(ifs.fail()==true)
			throw runtime_error("A logical error occurred while trying to scan in dimensions from data file");
		else if(ifs.eof()==true)
			throw runtime_error("End of file occurred after trying to scan in dimensions of the input file - this file appears to be empty");
	/*FIRST ROW OF FILE: convert data dimensions string into (numeric) dims array*/
		string tempStr;
		unsigned long temp;
		/*first determine number of dimensions and pixels*/
			stringstream ss(str);//copy str (a string) into ss (a stringstream)
			int i=0;
			while(!ss.eof())//scan in dimensions until the end of the string is reached
			{
				if(i >= MAX_DIM)
					throw runtime_error("Number of dimensions of data file exceeds program ability");
				/*read in next dimension into dims array*/
					ss >> tempStr;
					temp = stod(tempStr);
				if(temp > 1)
					numDims++;
				i++;
				ss >> ws;
			}
			ss.clear();//must clear to reuse
		/*now allocate memory for dims and scan in dimensions*/
			dims = new unsigned long [numDims];
			ss.str(str);
			numPixels = 1;
			for(int i=0; i<numDims; i++)//scan in dimensions until the end of the string is reached
			{
				ss >> tempStr;
				dims[i] = stod(tempStr);
				if(dims[i] > 1)
					numPixels *= dims[i];
				else if(dims == 0)
					dims[i] = 1;//user is an idiot
				ss >> ws;
			}
			ss.clear();//must clear to reuse
	/*REMAINING ROWS OF FILE: read in data from file and save to cudata buffer*/
		complex<double> *hostBuffer = new complex<double> [numPixels];//hold data on CPU for a moment - will transfer to GPU shortly
		ifs >> ws;
		unsigned long CI=0, RI;//indexing variables
		string realStr, imagStr;
		while(ifs.good()==true)
		{
			getline(ifs,str);//scan next line of data into string (str)
			ss.str(str);//transfer string of data to a stringstream (ss)
			ss >> ws;
			while(ss.eof()==false)//continue reading stringstream until the end of stream is reached
			{
				if(ss.good()==false)//make sure stringstream is good to read
					throw runtime_error("An error occurred while reading data file at string stream ss(str)");
				if(CI >= numPixels)//make sure data in file isn't longer than expected
					throw runtime_error("Input data file has more pixels than dimensions initially defined!");
				ss >> realStr;//read in real part
				ss >> ws;
				ss >> imagStr;
				ss >> ws;
				ss >> tempStr;//read in imaginary part
				imagStr += tempStr;
				/*data is assumed column major in file, but we need to store data as row major in cudata buffers for optimal performance with cuda libraries*/
					/*convert column-major index to row-major index*/
						RI = CI2RI(CI,dims,numDims);
					/*read in datum from file into host buffer using string to double function and row-major index*/
						hostBuffer[RI].real(stod(realStr));
						hostBuffer[RI].imag(stod(imagStr));
				CI++;
				ss >> ws;
			}
			ss.clear();//clear to reuse
			ifs >> ws;
		}
		if(ifs.fail()==true)//make sure nothing bad happened while reading from input file
			throw runtime_error("An error occurred while reading the data file");
		else if(ifs.eof() && CI < numPixels)
			throw runtime_error("Data file has fewer pixels than dimensions initially defined!");
	//allocate GPU memory for device buffer
		cudaCheck("constructor cudaMalloc in cudata(filename) for cudata.buffer",cudaMalloc((void**)&buffer,numPixels*sizeof(cuDoubleComplex)));
	//transfer buffer data from host to device
		cudaCheck("constructor cudaMemcpy in cudata(filename) for cudata.buffer",cudaMemcpy(buffer,hostBuffer,numPixels*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice));
	//allocate GPU memory for device version of dims array
		cudaCheck("constructor cudaMemcpy in cudata(filename) for cudata.cuDims",cudaMalloc((void**)&cuDims,MAX_DIM*sizeof(unsigned long)));
		updateCuDims();
	delete[] hostBuffer;
}

/*deep copy assignment operator*/
template <>
cudata<cuDoubleComplex>& cudata<cuDoubleComplex>::operator=(const cudata &obj)
{
	//copy dimensions
		numPixels = obj.numPixels;
		numDims = obj.numDims;
		cudaFree(buffer);//free existing cuda memory in this->buffer
		if(obj.buffer != nullptr)//make sure nullptr doesn't get copied
		{
			cudaCheck("cudaMalloc in deep assignment operator for buffer",cudaMalloc((void**)&buffer,obj.numPixels*sizeof(cuDoubleComplex)));
			cudaCheck("cudaMemcpy buffer in deep assignment operator",cudaMemcpy(buffer,obj.buffer,obj.numPixels*sizeof(cuDoubleComplex),cudaMemcpyDeviceToDevice));
		}
		else
			buffer = nullptr;
		cudaFree(cuDims);//free existing cuda memory in this->cuDims
		if(obj.cuDims!= nullptr)//make sure nullptr doesn't get copied
			cudaCheck("cudaMalloc in deep assignment operator for cuDims",cudaMalloc((void**)&cuDims,obj.numDims*sizeof(cuDoubleComplex)));
		else
			cuDims = nullptr;
		delete[] dims;
		dims = new unsigned long [numDims];
		for(int i=0; i<numDims; i++)
			dims[i] = obj.dims[i];
		updateCuDims();
	//copy bools and cublas handle
		this->boolCopy(obj);
	return *this;
}
/*make sure buffer is in column major format*/
template <>
void cudata<cuDoubleComplex>::colMajor()
{
	if(isRowMajor)
		throw runtime_error("colMajor member function incomplete");
}

/*element-wise addition operator*/
template <>
void cudata<cuDoubleComplex>::add(cudata &A, cudata &B)
{
	if( (this->numPixels == A.numPixels) && (this->numPixels == B.numPixels) )
	{
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in cudata::cu_AZ_add_BZ()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cu_AZ_add_BZ,0,0));
		gridSize = (this->numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cu_AZ_add_BZ<<<gridSize,blockSize>>>(this->numPixels,this->buffer,A.buffer,B.buffer);
		cudaDeviceSynchronize();
	}
	else
		throw runtime_error("cudata::cu_AZ_add_BZ() is not defined for current instances of objects");
}
template <>
void cudata<cuDoubleComplex>::add(cudata &A, double b, cudata &B)
{
	if( (this->numPixels == A.numPixels) && (this->numPixels == B.numPixels) )
	{
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in cudata:cu_AZ_add_bd_BZ()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cu_AZ_add_bd_BZ,0,0));
		gridSize = (this->numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cu_AZ_add_bd_BZ<<<gridSize,blockSize>>>(this->numPixels,this->buffer,A.buffer,b,B.buffer);
		cudaDeviceSynchronize();
	}
	else
		throw runtime_error("cudata::cu_AZ_add_bd_BZ() is not defined for current instances of objects");
}
template <>
void cudata<cuDoubleComplex>::axpy(cuDoubleComplex a, cudata &x)
{
	if(this->numPixels == x.numPixels)
	{
		cudaCheck("cublasZaxpy for numerator in cudata::operator+=(cuDoubleComplex a, cudata &x)",cublasZaxpy(*(cublas_handle),(int)(numPixels),&a,x.buffer,1,this->buffer,1));
	}
	else
		throw runtime_error("cudata::operator+=(cuDoubleComplex a, cudata &x) is not defined for current instances of objects");
}
template <>
void cudata<cuDoubleComplex>::axmy(cuDoubleComplex a, cudata &x)
{
	if(this->numPixels == x.numPixels)
	{
		a.x = -a.x;
		a.y = -a.y;
		cudaCheck("cublasZaxpy for numerator in cudata::operator-=(cuDoubleComplex a, cudata &x)",cublasZaxpy(*(cublas_handle),(int)(numPixels),&a,x.buffer,1,this->buffer,1));
	}
	else
		throw runtime_error("cudata::operator-=(cuDoubleComplex a, cudata &x) is not defined for current instances of objects");
}
/*dot product*/
template <>
cuDoubleComplex cudata<cuDoubleComplex>::dot(cudata &A)
{
	cuDoubleComplex product;
	if(this->numPixels == A.numPixels)
	{
		cudaCheck("cublasZdotc for numerator in cudata::dot(cudata &A)",cublasZdotc(*(cublas_handle),(int)(numPixels),buffer,1,A.buffer,1,&product));
	}
	else
		throw runtime_error("cudata::dot(cudata &A) is not defined for current instances of objects");
	return product;
}
/*element-wise subtraction operator*/
template <>
void cudata<cuDoubleComplex>::subtract(cudata &A, cudata &B)
{
	if( (this->numPixels == A.numPixels) && (this->numPixels == B.numPixels) )
	{
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in cudata::cu_AZ_minus_BZ()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cu_AZ_minus_BZ,0,0));
		gridSize = (this->numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cu_AZ_minus_BZ<<<gridSize,blockSize>>>(this->numPixels,this->buffer,A.buffer,B.buffer);
		cudaDeviceSynchronize();
	}
	else
		throw runtime_error("cudata::cu_AZ_minus_BZ() is not defined for current instances of objects");
}
template <>
void cudata<cuDoubleComplex>::subtract(cudata &A, cuDoubleComplex b, cudata &B)
{
	if( (this->numPixels == A.numPixels) && (this->numPixels == B.numPixels) )
	{
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in cudata::cu_AZ_minus_bZ_BZ()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cu_AZ_minus_bZ_BZ,0,0));
		gridSize = (this->numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cu_AZ_minus_bZ_BZ<<<gridSize,blockSize>>>(this->numPixels,this->buffer,A.buffer,b,B.buffer);
		cudaDeviceSynchronize();
	}
	else
		throw runtime_error("cudata::cu_AZ_minus_bZ_BZ() is not defined for current instances of objects");
}
/*element-wise multiplication operator*/
template <>
cudata<cuDoubleComplex>& cudata<cuDoubleComplex>::operator*=(double d)
{
	cudaCheck("cublasZdscal for numerator in cudata::operator*=(double d)",cublasZdscal(*(cublas_handle),(int)(numPixels),&d,buffer,1));
	return *this;
}
template <>
cudata<cuDoubleComplex>& cudata<cuDoubleComplex>::operator/=(double d)
{
	double dinv = 1.0/d;
	cudaCheck("cublasZdscal for numerator in cudata::operator*=(double d)",cublasZdscal(*(cublas_handle),(int)(numPixels),&dinv,buffer,1));
	return *this;
}
template <>
cudata<cuDoubleComplex>& cudata<cuDoubleComplex>::operator*(cudata &obj)
{
	if(this->isIdentity)
	{
		return obj;
	}
	else if(this->isDiagonal)
		throw runtime_error("cudata overloaded operator* not defined for diagonal obj");
	else
		throw runtime_error("cudata overloaded operator* not defined for non-diagonal obj");
}
template <>
void cudata<cuDoubleComplex>::multiply(cudata &A, cudata &B)
{
	if(A.isIdentity == true)
	{
		*this = B;
	}
	else if(B.isIdentity == true)
	{
		*this = A;
	}
	else if( (this->numPixels == A.numPixels) && (this->numPixels == B.numPixels) )
	{
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in cudata::multiply()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cu_AZ_multiply_BZ,0,0));
		gridSize = (this->numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cu_AZ_multiply_BZ<<<gridSize,blockSize>>>(this->numPixels,this->buffer,A.buffer,B.buffer);
		cudaDeviceSynchronize();
	}
	else
		throw runtime_error("cudata::multiply() is not defined for current instances of objects");
}
template <>
void cudata<cuDoubleComplex>::multiply_(cudata &B)
{
	if(B.isIdentity == true)
	{
		return;
	}
	else if( this->numPixels == B.numPixels )
	{
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in cudata::multiply()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cu_this_multiply_BZ,0,0));
		gridSize = (this->numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cu_this_multiply_BZ<<<gridSize,blockSize>>>(this->numPixels,this->buffer,B.buffer);
		cudaDeviceSynchronize();
	}
	else
		throw runtime_error("cudata::multiply() is not defined for current instances of objects");
}
//write linear array to buffer
template <>
void cudata<double>::linear(double x0, double dx)
{
	int gridSize, blockSize, minGridSize;
	cudaCheck("Block size calculator in cudata<double>::linear()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuDoubleLinear,0,0));
	gridSize = (this->numPixels+blockSize-1)/blockSize;
	cudaDeviceSynchronize();
	cuDoubleLinear<<<gridSize,blockSize>>>(this->numPixels,this->buffer,x0,dx);
	cudaDeviceSynchronize();
}
/*return maximum magnitude of elements of cudata array*/
template <>
double cudata<cuDoubleComplex>::max_magnitude()
{
	int maxIndx;
	complex<double> maxZ;
	cudaCheck("cublasIzamax",cublasIzamax(*cublas_handle,numPixels,buffer,1,&maxIndx));//note maxIndx is returned in 1-based indexing
	cudaMemcpy(&maxZ,&(buffer[maxIndx-1]),sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);//subtract 1 to account for 1-based indexing
	return sqrt(maxZ.real()*maxZ.real()+maxZ.imag()*maxZ.imag());
}
/*print buffer to file*/
template <>
void cudata<cuDoubleComplex>::printb2f(string filename, char CorR)
{
	unsigned long *IV = new unsigned long[numDims];
	//allocate host memory and transfer buffer
		complex<double> *hostBuffer = new complex<double>[numPixels];
		cudaCheck("cudaMemcpy in printb2f()",cudaMemcpy(hostBuffer,buffer,numPixels*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost));
	//print dimensions on first line
		ofstream ofs(filename);
		/*for(int i=0; i<numDims; i++)
			ofs << dims[i] << "\t";
		ofs << endl;*/
	//reset ofs formatting
		ofs.precision(3);
		ofs << std::scientific;
	unsigned long CI, indx;
	for(CI=0; CI<numPixels; CI++)
	{
		//determine datum index as column-major or row-major
			if(isRowMajor)
				indx = CI2RI(CI,dims,numDims);
			else
				indx = CI;
		//print datum to file
			if(CorR == 'C' || CorR == 'c')//print complex
			{
				//ofs << hostBuffer[indx] << '\t';
				ofs << hostBuffer[indx].real();
				if( hostBuffer[indx].imag() >= 0)
				{
					ofs << " + " << hostBuffer[indx].imag() << "i\t";
				}
				else
				{
					ofs << " - " << abs(hostBuffer[indx].imag()) << "i\t";
				}
			}
			else if(CorR == 'R' || CorR == 'r')//print real
				ofs << hostBuffer[indx].real() << '\t';
			else if(CorR == 'I' || CorR == 'i')//print imaginary
				ofs << hostBuffer[indx].imag() << '\t';
			else if(CorR == 'M' || CorR == 'm')//print magnitude
				ofs << sqrt(hostBuffer[indx].real()*hostBuffer[indx].real()+hostBuffer[indx].imag()*hostBuffer[indx].imag()) << '\t';
			else
				throw runtime_error("in printb2f an invalid char was entered as an arguement for CorR");
		//print delimitating character
			IV = CI2IV(CI,IV,dims,numDims);
			bool endlFlag=true;
			for(int i=0; i<numDims; i++)
			{
				for(int k=0; k<=i; k++)
				{
					if(IV[k] != (dims[k]-1))
						endlFlag = false;
				}
				if(endlFlag)
					ofs << endl;
			}
	}
	ofs.close();
	delete[] IV;
	delete[] hostBuffer;
}
template <>
void cudata<cuDoubleComplex>::printb2f(string filename, int n, char CorR)
{
	stringstream ss;
	ss << n << "-" << filename;
	printb2f(ss.str(),CorR);
}
template <>
void cudata<cuDoubleComplex>::printmap2f(string filename, unsigned long *map)
{
	unsigned long *IV = new unsigned long[numDims];
	//allocate host memory and transfer map
		unsigned long *hostMap = new unsigned long[numPixels];
		cudaCheck("cudaMemcpy in printb2f()",cudaMemcpy(hostMap,map,numPixels*sizeof(unsigned long),cudaMemcpyDeviceToHost));
	//print dimensions on first line
		ofstream ofs(filename);
		/*for(int i=0; i<numDims; i++)
			ofs << dims[i] << "\t";
		ofs << endl;*/
	//reset ofs formatting
	unsigned long CI, indx;
	for(CI=0; CI<numPixels; CI++)
	{
		//determine datum index as column-major or row-major
			if(isRowMajor)
				indx = CI2RI(CI,dims,numDims);
			else
				indx = CI;
		//print datum to file
			ofs << hostMap[indx] << '\t';
			/*if(CorR == 'C' || CorR == 'c')//print map element

			else if(CorR == 'R' || CorR == 'r')//print real
				ofs << hostBuffer[indx].real() << '\t';
			else if(CorR == 'M' || CorR == 'm')//print magnitude
				ofs << hostBuffer[indx].real()*hostBuffer[indx].real()+hostBuffer[indx].imag()*hostBuffer[indx].imag() << '\t';
			else
				throw runtime_error("in printmap2f an invalid char was entered as an arguement for CorR");*/
		//print delimitating character
			IV = CI2IV(CI,IV,dims,numDims);
			bool endlFlag=true;
			for(int i=0; i<numDims; i++)
			{
				for(int k=0; k<=i; k++)
				{
					if(IV[k] != (dims[k]-1))
						endlFlag = false;
				}
				if(endlFlag)
					ofs << endl;
			}
	}
	ofs.close();
	delete[] IV;
	delete[] hostMap;
}
/*rearranges index in ascending dimension order*/
template <>
void cudata<cuDoubleComplex>::calcAscendingMap(unsigned long *map)
{
	for(int i=0; i<numDims; i++)
	{
		map[i] = i;
	}
	for(unsigned long i=0; i<numDims; i++)
	{
		map[i] = 0;
		for(unsigned long k=0; k<i; k++)
		{
			if(dims[i] >= dims[k])
			{
				map[i]++;
			}
		}
		for(int k=i; k<numDims; k++)
		{
			if(dims[i] > dims[k])
			{
				map[i]++;
			}
		}
	}
}
/*returns euclidean norm of buffer (the sqrt of the absolute magnitude)*/
template <>
double cudata<cuDoubleComplex>::nrm2()
{
	double nrm2;
	cudaCheck("cublasDznrm2 for numerator in cudata:nrm2()",cublasDznrm2(*(cublas_handle),(int)(numPixels),buffer,1,&nrm2));
	return nrm2;
}
/*returns p-norm of buffer (||x||_p = (sum_i |x_i|^p)^(1/p))*/
template<>
void cudata<cuDoubleComplex>::cuSqrt(cudata &x_in)
{
	int gridSize, blockSize, minGridSize;
	cudaCheck("Block size calculator in sqrt()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cu_sqrt,0,0));
	gridSize = (numPixels+blockSize-1)/blockSize;
	cudaDeviceSynchronize();
	cu_sqrt<<<gridSize,blockSize>>>(x_in.numPixels,x_in.buffer,this->buffer);
	cudaDeviceSynchronize();
}
/*computes the element-wise absolute square of cudata object and returns the answer in-pace*/
template <>
void cudata<cuDoubleComplex>::ewAbsSq_()
{
	int gridSize, blockSize, minGridSize;
	cudaCheck("Block size calculator in ewAbsSq_inplace()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuAbsSq,0,0));
	gridSize = (numPixels+blockSize-1)/blockSize;
	cudaDeviceSynchronize();
	cuAbsSq<<<gridSize,blockSize>>>(numPixels,this->buffer,this->buffer);
	cudaDeviceSynchronize();
}
/*computes the element-wise absolute square of cudata object and returns the answer*/
template <>
void cudata<cuDoubleComplex>::ewAbsSq(cudata &out)
{
	int gridSize, blockSize, minGridSize;
	cudaCheck("Block size calculator in ewAbsSq()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuAbsSq,0,0));
	gridSize = (this->numPixels+blockSize-1)/blockSize;
	cudaDeviceSynchronize();
	cuAbsSq<<<gridSize,blockSize>>>(this->numPixels,this->buffer,out.buffer);
	cudaDeviceSynchronize();
}
/*computes the multidimensional fourier transform of cudata buffer in-place*/
template <>
void cudata<cuDoubleComplex>::fft_(cufft_struct *cufft, int direction)
{
	/*Description: Computes the fft transform of the member function's buffer.*/
	/* Parameters:
	 * int *dim: This is dimension over which to do the fft. E.g. pass dim = MAX_DIM if full multidimensional
	 * 		transform desired, or pass dim = i if 1D transform across the i'th dimension desired.
	 * int direction: Pass either CUFFT_FORWARD or CUFFT_INVERSE to specify direction of fft to be computed.*/
	/*make sure direction parameter is valid*/
		if(direction != CUFFT_FORWARD && direction != CUFFT_INVERSE)
			throw runtime_error("dimension parameter invalid in fft_inplace()");
	/*make sure cudata is in row-major format, as required by cuFFT library*/
		this->rowMajor();
	/*proceed with fourier transform under appropriate dimension(s)*/
		if(numDims <= 3)
			cudaCheck("cufft_inplace", (cufft->cufft_result = cufftExecZ2Z(cufft->cufft_handle, buffer, buffer, direction) ));
		else
			throw runtime_error("numDims > 3 case of cufftExecZ2Z in fft_inplace() is not supported yet");
}
/*computes the multidimensional fourier transform of cudata buffer out-of-place*/
template <>
void cudata<cuDoubleComplex>::fft(cufft_struct *cufft, int direction, cudata &out)
{
	/*Description: Computes the fft transform of the member function's buffer and returns object in place of out.*/
	/* Parameters:
	 * int *dim: This is dimension over which to do the fft. E.g. pass dim = MAX_DIM if full multidimensional
	 * 		transform desired, or pass dim = i if 1D transform across the i'th dimension desired.
	 * int direction: Pass either CUFFT_FORWARD or CUFFT_INVERSE to specify direction of fft to be computed.*/
	/*make sure direction parameter is valid*/
		if(direction != CUFFT_FORWARD && direction != CUFFT_INVERSE)
			throw runtime_error("dimension parameter invalid in fft_inplace()");
	/*make sure cudata is in row-major format, as required by cuFFT library*/
		this->rowMajor();
	/*proceed with fourier transform under appropriate dimension(s)*/
		if(numDims <= 3)
			cudaCheck("cufft_inplace", (cufft->cufft_result = cufftExecZ2Z(cufft->cufft_handle, buffer, out.buffer, direction) ));
		else
			throw runtime_error("numDims > 3 case of cufftExecZ2Z in fft_inplace() is not supported yet");
}
/*canvas-cut buffer*/
template <>
void cudata<cuDoubleComplex>::canvasCut(cudata &cutObj, unsigned long *cutDims, unsigned long *cutIndxStart)
{
	this->rowMajor();
	cout << "missing dimensions check in cudata:canvasCut()" << endl;
	cout << "missing general object transfer in cudata:canvasCut()" << endl;
	int gridSize, blockSize, minGridSize;
	cudaCheck("Block size calculator in cudata:canvasCut()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuCanvasCutComplex,0,0));
	gridSize = (cutObj.numPixels+blockSize-1)/blockSize;
	unsigned long *cuCutIndxStart;
	cudaCheck("cudaMalloc of cuCutIndxStart in cudata:canvasCut()",cudaMalloc((void**)&cuCutIndxStart,numDims*sizeof(unsigned long)));
	cudaCheck("cudaMemcpy of cuCutIndxStart in cudata:canvasCut()",cudaMemcpy(cuCutIndxStart,cutIndxStart,numDims*sizeof(unsigned long),cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	cuCanvasCutComplex<<<gridSize,blockSize>>>(cutObj.numPixels,cutObj.numDims,cutObj.cuDims,cutObj.buffer,cuCutIndxStart,cuDims,buffer);
	cudaDeviceSynchronize();
	cudaFree(cuCutIndxStart);
}
template <>
void cudata<double>::canvasCut(cudata &cutObj, unsigned long *cutDims, unsigned long *cutIndxStart)
{
	this->rowMajor();
	cout << "missing dimensions check in cudata:canvasCut()" << endl;
	cout << "missing general object transfer in cudata:canvasCut()" << endl;
	int gridSize, blockSize, minGridSize;
	cudaCheck("Block size calculator in cudata:canvasCut()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuCanvasCutDouble,0,0));
	gridSize = (cutObj.numPixels+blockSize-1)/blockSize;
	unsigned long *cuCutIndxStart;
	cudaCheck("cudaMalloc of cuCutIndxStart in cudata:canvasCut()",cudaMalloc((void**)&cuCutIndxStart,numDims*sizeof(unsigned long)));
	cudaCheck("cudaMemcpy of cuCutIndxStart in cudata:canvasCut()",cudaMemcpy(cuCutIndxStart,cutIndxStart,numDims*sizeof(unsigned long),cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	cuCanvasCutDouble<<<gridSize,blockSize>>>(cutObj.numPixels,cutObj.numDims,cutObj.cuDims,cutObj.buffer,cuCutIndxStart,cuDims,buffer);
	cudaDeviceSynchronize();
	cudaFree(cuCutIndxStart);
}
/*contiguous-cut to buffer*/
template <>
void cudata<cuDoubleComplex>::contiguousCut(cudata &cutObj, unsigned long *cutDims, unsigned long *cutIndxStart)
{
	this->rowMajor();
	int gridSize, blockSize, minGridSize;
	cudaCheck("Block size calculator in cudata:contiguousCut()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuContiguousCut,0,0));
	gridSize = (cutObj.numPixels+blockSize-1)/blockSize;
	unsigned long *cuCutIndxStart;
	cudaCheck("cudaMalloc of cuCutIndxStart in cudata:contiguousCut()",cudaMalloc((void**)&cuCutIndxStart,numDims*sizeof(unsigned long)));
	cudaCheck("cudaMemcpy of cuCutIndxStart in cudata:contiguousCut()",cudaMemcpy(cuCutIndxStart,cutIndxStart,numDims*sizeof(unsigned long),cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	cuContiguousCut<<<gridSize,blockSize>>>(cutObj.numPixels,cutObj.numDims,cutObj.cuDims,cutObj.buffer,cuCutIndxStart,cuDims,buffer);
	cudaDeviceSynchronize();
	cudaFree(cuCutIndxStart);
}
/*contiguous-pad to buffer*/
template <>
void cudata<cuDoubleComplex>::contiguousPad(cudata &obj, unsigned long *padIndxStart, unsigned long *padSize, cuDoubleComplex padVal)
{
	this->boolCopy(obj);
	this->rowMajor();
	obj.numPixels = 1;
	obj.numDims = numDims;
	for(int i=0; i<numDims; i++)
	{
		obj.dims[i] = dims[i]+padSize[i];
		obj.numPixels *= obj.dims[i];
	}
	obj.updateCuDims();
	cudaFree(obj.buffer);
	cudaCheck("cudaMalloc of padded buffer in cudata::contiguousPad()",cudaMalloc((void**)&(obj.buffer),obj.numPixels*sizeof(cuDoubleComplex)));
	unsigned long *cuPadIndxStart, *cuPadSize;
	cudaCheck("cudaMalloc of cuStartPadIndx in cudata::contiguousPad()",cudaMalloc((void**)&cuPadIndxStart,obj.numDims*sizeof(unsigned long)));
	cudaCheck("cudaMalloc of cuPadSize in cudata::contiguousPad()",cudaMalloc((void**)&cuPadSize,obj.numDims*sizeof(unsigned long)));
	cudaCheck("cudaMemcpy cupadIndxStart in cudata::contiguousPad()",cudaMemcpy(cuPadIndxStart,padIndxStart,obj.numDims*sizeof(unsigned long),cudaMemcpyHostToDevice));
	cudaCheck("cudaMemcpy of cuPadSize in cudata::contiguousPad()",cudaMemcpy(cuPadSize,padSize,obj.numDims*sizeof(unsigned long),cudaMemcpyHostToDevice));
	int gridSize, blockSize, minGridSize;
	cudaCheck("Block size calculator in cudata::contiguousPad()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuContiguousPad,0,0));
	gridSize = (obj.numPixels+blockSize-1)/blockSize;
	cudaDeviceSynchronize();
	cuContiguousPad<<<gridSize,blockSize>>>(obj.numPixels,obj.numDims,buffer,obj.buffer,padVal,cuPadIndxStart,cuDims,obj.cuDims,cuPadSize);
	cudaDeviceSynchronize();
	cudaFree(cuPadIndxStart);
	cudaFree(cuPadSize);
}
/*canvas-pad to buffer*/
template <>
void cudata<cuDoubleComplex>::canvasPad(cudata &obj, unsigned long *startPasteIndx, unsigned long *canvasDims, cuDoubleComplex canvasPadVal)
{
	this->boolCopy(obj);
	this->rowMajor();
	unsigned long *padSize = new unsigned long[numDims];
	for(int i=0; i<numDims; i++)
		padSize[i] = obj.dims[i] - dims[i];
	obj.updateCuDims();
	cudaFree(obj.buffer);
	cudaCheck("cudaMalloc of padded buffer in cudata::canvasPad()",cudaMalloc((void**)&(obj.buffer),obj.numPixels*sizeof(cuDoubleComplex)));
	unsigned long *cuStartPasteIndx, *cuPadSize;
	cudaCheck("cudaMalloc of cuStartPadIndx in cudata::canvasPad()",cudaMalloc((void**)&cuStartPasteIndx,obj.numDims*sizeof(unsigned long)));
	cudaCheck("cudaMalloc of cuPadSize in cudata::canvasPad()",cudaMalloc((void**)&cuPadSize,obj.numDims*sizeof(unsigned long)));
	cudaCheck("cudaMemcpy cupadIndxStart in cudata::canvasPad()",cudaMemcpy(cuStartPasteIndx,startPasteIndx,obj.numDims*sizeof(unsigned long),cudaMemcpyHostToDevice));
	cudaCheck("cudaMemcpy of cuPadSize in cudata::canvasPad()",cudaMemcpy(cuPadSize,padSize,obj.numDims*sizeof(unsigned long),cudaMemcpyHostToDevice));
	int gridSize, blockSize, minGridSize;
	cudaCheck("Block size calculator in cudata::canvasPad()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuContiguousPad,0,0));
	gridSize = (obj.numPixels+blockSize-1)/blockSize;
	cudaDeviceSynchronize();
	cuCanvasPad<<<gridSize,blockSize>>>(obj.numPixels,obj.numDims,buffer,obj.buffer,canvasPadVal,cuStartPasteIndx,cuDims,obj.cuDims,cuPadSize);
	cudaDeviceSynchronize();
	cudaFree(cuStartPasteIndx);
	cudaFree(cuPadSize);
	delete[] padSize;
}
/*computes the eigenvectors and eigenvalues of gram matrix*/
template <>
void cudata<cuDoubleComplex>::eigen(cusolver_struct *cusolver)
{
	/*make sure buffer is row major*/
		this->rowMajor();
	/*make sure cusolver handle, workspace and variables are initialized*/
		if(cusolver->cusolver_status != CUSOLVER_STATUS_SUCCESS)
		{
			/**/
				cudaCheck("create cusolver handle in eigen()",(cusolver->cusolver_status = cusolverDnCreate(&(cusolver->cusolver_DnHandle))));
				cusolver->eigen_mode = CUSOLVER_EIG_MODE_VECTOR;
				cusolver->fill_mode = CUBLAS_FILL_MODE_UPPER;
			/**/
				cudaCheck("cudaMalloc cuW in eigen()", cudaMalloc((void**)&(cusolver->cuW),(this->dims[0])*sizeof(double)));
				cudaCheck("cudaMalloc cuInfo in eigen()", cudaMalloc((void**)&(cusolver->cuInfo),sizeof(int)));
				cudaCheck("create cusolver workspace in eigen()",cusolverDnZheevd_bufferSize(cusolver->cusolver_DnHandle,cusolver->eigen_mode,cusolver->fill_mode,this->dims[0],this->buffer,this->dims[0],cusolver->cuW,&(cusolver->Lwork)));
			/**/
				cudaCheck("cudaMalloc cuWork in eigen()", cudaMalloc((void**)&(cusolver->cuWork),(cusolver->Lwork)*sizeof(cuDoubleComplex)));
		}
	/*execute eigen decomposition*/
		cudaCheck("Gram matrix eigen decomposition",cusolverDnZheevd(cusolver->cusolver_DnHandle, cusolver->eigen_mode, cusolver->fill_mode, this->dims[0] , this->buffer, this->dims[0], cusolver->cuW, cusolver->cuWork, cusolver->Lwork, cusolver->cuInfo));
	/*check to see if any errors occurred*/
		cudaCheck("cudaMemcpy cuInfo to info in eigen()", cudaMemcpy(&(cusolver->hostInfo),cusolver->cuInfo,sizeof(int),cudaMemcpyDeviceToHost));
		if(cusolver->hostInfo != 0)
		{
			if(cusolver->hostInfo < 0)
				cout << "eigen decomposition failed: the " << cusolver->hostInfo << "'th input parameter is invalid\n";
			else
				cout << "eigen decomposition failed: the " << cusolver->hostInfo << "'th off-diagonal element failed to converge\n";
		}
}
/*prints out eigenvalues (eVals), eigenvectors (E), the two eigen-identity(M*E = E*diag(eVals)) matrices and their difference (ME-EeVals, a.k.a the secular which should be ~0)*/
template <>
void cudata<cuDoubleComplex>::eigenCheck(cusolver_struct *cusolver, cudata &M)
{
	/*allocate cuDoubleComplex device and host memory for eigen values (for dgmm function and printing to file)*/
		cuDoubleComplex *cueVals = nullptr;
		cudaCheck("cuMalloc cueVals", cudaMalloc((void**)&(cueVals),this->dims[0]*sizeof(cuDoubleComplex)));
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in doubleTocuComplex()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuDoubleTocuComplex,0,0));
		gridSize = (this->dims[0]+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cuDoubleTocuComplex<<<gridSize,blockSize>>>(this->dims[0],cusolver->cuW,cueVals);
		cudaDeviceSynchronize();
		complex<double> *eVals = new complex<double>[this->dims[0]];
		cudaCheck("cudaMemcpy eVals", cudaMemcpy(eVals,cueVals,this->dims[0]*sizeof(complex<double>),cudaMemcpyDeviceToHost));
	/*open file to print eVals*/
		ofstream ofs("eVals.txt");
		ofs.precision(5);
		for(int k=0; k<(this->dims[0]); k++)
			ofs << eVals[this->dims[0]-1-k].real() << "\n";
		ofs.close();
	/*allocate necessary cudata objects*/
		cudata hemmOut(M);
		cudata dgmmOut(M);
		cudata subtract(M);
	/*print E to file*/
		this->printb2f("E.txt",'c');
	/*compute M*E and print to file*/
		M.hemm(*cublas_handle,*this,hemmOut);
		hemmOut.printb2f("ME.txt",'c');
	/*compute E*diag(eVals) and print to file*/
		this->dgmm(*cublas_handle,cueVals,dgmmOut);
		dgmmOut.printb2f("EeVal.txt",'c');
	/*compute ME-EeVal (the secular matrix) and print to file*/
		geam(*cublas_handle,dgmmOut,hemmOut,subtract);
		subtract.printb2f("ME-EeVal.txt",'c');
	/*free dynamic memory*/
		delete[] eVals;
		cudaFree(cueVals);
}
/*reads in mask*/
int readMask(string filename, int **mask)
{
	ifstream ifs(filename);
	string str, tempStr;
	stringstream ss;
	getline(ifs,str);
	ss.str(str);
	int i=0;
	while(ss.good()==true)
	{
		*mask = (int*)realloc(*mask,(i+1)*sizeof(int));
		ss >> tempStr;
		(*mask)[i] = stoi(tempStr);
		i++;
		ss >> ws;
	}
	ifs.close();
	ss.clear();
	return i;
}
/*hemm inplace*/
template <class T>
void cudata<T>::hemm(cublasHandle_t cublas_handle, cudata &in, cudata &out)
{
	cuDoubleComplex alpha;
	alpha.x = 1;
	alpha.y = 0;
	cuDoubleComplex beta;
	beta.x = 0;
	beta.y = 0;
	cudaCheck("cudata::hemm",cublasZhemm(cublas_handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,dims[0],dims[0],&alpha,this->buffer,dims[0],in.buffer,in.dims[0],&beta,out.buffer,dims[0]));
}

/*updates the object's dimensions that are stored on the device with the dimensions of the host*/



template <>
void cudata<cuDoubleComplex>::setConst(double real, double imag)
{
	int gridSize, blockSize, minGridSize;
	cudaCheck("Block size calculator in cudata:cuSetConst()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuSetConst,0,0));
	gridSize = (numPixels+blockSize-1)/blockSize;
	cudaDeviceSynchronize();
	cuSetConst<<<gridSize,blockSize>>>(numPixels,buffer,real,imag);
	cudaDeviceSynchronize();
}
/***Non-member functions***/

/*element-wise multiplication*/
void ewMultiply(cudata<cuDoubleComplex> &out, cuDoubleComplex a, cudata<cuDoubleComplex> &in1, cudata<cuDoubleComplex> &in2)
{
	int gridSize, blockSize, minGridSize;
	cudaCheck("Block size calculator in ewMultiply()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuewMultiply,0,0));
	gridSize = (out.numPixels+blockSize-1)/blockSize;
	cudaDeviceSynchronize();
	cuewMultiply<<<gridSize,blockSize>>>(out.numPixels,out.buffer,a,in1.buffer,in2.buffer);
	cudaDeviceSynchronize();
}
/*geam*/
void geam(cublasHandle_t cublas_handle, cudata<cuDoubleComplex>&A,cudata<cuDoubleComplex>&B,cudata<cuDoubleComplex>&out)
{
	cuDoubleComplex alpha;
	alpha.x = 1;
	alpha.y = 0;
	cuDoubleComplex beta;
	beta.x = -1;
	beta.y = 0;
	cublasZgeam(cublas_handle,CUBLAS_OP_T,CUBLAS_OP_T,A.dims[0],A.dims[0],&alpha,A.buffer,A.dims[0],&beta,B.buffer,B.dims[0],out.buffer,out.dims[0]);
}

/*cuda error handling function*/
//cudaCheck(__FILE__,__LINE__);//paste this in code when debugging
void cudaCheck(string description, cudaError_t error)
{
	if(error != cudaSuccess)
	{
		cout << cudaGetErrorName(error) << " at " << description << ": " << cudaGetErrorString(error) << endl;
		throw runtime_error("");
	}
}
void cudaCheck(string description, cublasStatus_t error)
{
	if(error == CUBLAS_STATUS_SUCCESS)
	{
		return;
	}
	else if(error == CUBLAS_STATUS_NOT_INITIALIZED)
	{
		cout << "cublas error" << " at " << description << ": " << "the library was not initialized" << endl;
	}
	else if(error == CUBLAS_STATUS_ALLOC_FAILED)
	{
		cout << "cublas error" << " at " << description << ": " << "the resource allocation failed" << endl;
	}
	else if(error == CUBLAS_STATUS_INVALID_VALUE)
	{
		cout << "cublas error" << " at " << description << ": " << "an invalid numerical value was used as an argument" << endl;
	}
	else if(error == CUBLAS_STATUS_ARCH_MISMATCH)
	{
		cout << "cublas error" << " at " << description << ": " << "an absent device architectural feature is required" << endl;
	}
	else if(error == CUBLAS_STATUS_MAPPING_ERROR)
	{
		cout << "cublas error" << " at " << description << ": " << "an access to GPU memory space failed" << endl;
	}
	else if(error == CUBLAS_STATUS_EXECUTION_FAILED)
	{
		cout << "cublas error" << " at " << description << ": " << "the GPU program failed to execute" << endl;
	}
	else if(error == CUBLAS_STATUS_INTERNAL_ERROR)
	{
		cout << "cublas error" << " at " << description << ": " << "an internal operation failed" << endl;
	}
	else if(error == CUBLAS_STATUS_NOT_SUPPORTED)
	{
		cout << "cublas error" << " at " << description << ": " << "the feature required is not supported" << endl;
	}
	else
	{
		cout << "cublas error" << " at " << description << ": " << "error not logged" << endl;
	}
	throw runtime_error("");
}
void cudaCheck(string description, cufftResult error)
{
	if(error == CUFFT_SUCCESS)
		return;
	else if(error == CUFFT_INVALID_PLAN)
	{
		cout << "cuFFT error" << " at " << description << ": " << "cuFFT was passed an invalid plan handle" << endl;
	}
	else if(error == CUFFT_ALLOC_FAILED)
	{
		cout << "cuFFT error" << " at " << description << ": " << "cuFFT failed to allocate GPU or CPU memory" << endl;
	}
	else if(error == CUFFT_INVALID_TYPE)
	{
		cout << "cuFFT error" << " at " << description << ": " << "No longer used" << endl;
	}
	else if(error == CUFFT_INVALID_VALUE)
	{
		cout << "cuFFT error" << " at " << description << ": " << "User specified an invalid pointer or parameter" << endl;
	}
	else if(error == CUFFT_INTERNAL_ERROR)
	{
		cout << "cuFFT error" << " at " << description << ": " << "Driver or internal cuFFT library error" << endl;
	}
	else if(error == CUFFT_EXEC_FAILED)
	{
		cout << "cuFFT error" << " at " << description << ": " << "Failed to execute an FFT on the GPU" << endl;
	}
	else if(error == CUFFT_SETUP_FAILED)
	{
		cout << "cuFFT error" << " at " << description << ": " << "The cuFFT library failed to initialize" << endl;
	}
	else if(error == CUFFT_INVALID_SIZE)
	{
		cout << "cuFFT error" << " at " << description << ": " << "User specified an invalid transform size" << endl;
	}
	else if(error == CUFFT_UNALIGNED_DATA)
	{
		cout << "cuFFT error" << " at " << description << ": " << "No longer used" << endl;
	}
	else if(error == CUFFT_INCOMPLETE_PARAMETER_LIST)
	{
		cout << "cuFFT error" << " at " << description << ": " << " Missing parameters in call" << endl;
	}
	else if(error == CUFFT_INVALID_DEVICE)
	{
		cout << "cuFFT error" << " at " << description << ": " << "Execution of a plan was on different GPU than plan creation" << endl;
	}
	else if(error == CUFFT_PARSE_ERROR)
	{
		cout << "cuFFT error" << " at " << description << ": " << "Internal plan database error" << endl;
	}
	else if(error == CUFFT_NO_WORKSPACE)
	{
		cout << "cuFFT error" << " at " << description << ": " << "No workspace has been provided prior to plan execution" << endl;
	}
	else if(error == CUFFT_NOT_IMPLEMENTED)
	{
		cout << "cuFFT error" << " at " << description << ": " << "Function does not implemente functionality for parameters given" << endl;
	}
	else if(error == CUFFT_LICENSE_ERROR)
	{
		cout << "cuFFT error" << " at " << description << ": " << "Used in previous versions" << endl;
	}
	else if(error == CUFFT_NOT_SUPPORTED)
	{
		cout << "cuFFT error" << " at " << description << ": " << "Operation is not supported for parameters given" << endl;
	}
	else
	{
		cout << "cuFFT error" << " at " << description << ": " << "error not logged" << endl;
	}
	throw runtime_error("");
}
void cudaCheck(string description,cusolverStatus_t error)
{
	if(error == CUSOLVER_STATUS_SUCCESS)
		return;
	if(error == CUSOLVER_STATUS_NOT_INITIALIZED)
		throw runtime_error("The cuSolver library was not initialized. This is usually caused by the lack of a prior call, an error in the CUDA Runtime API called by the cuSolver routine, or an error in the hardware setup");
	if(error == CUSOLVER_STATUS_ALLOC_FAILED)
		throw runtime_error("Resource allocation failed inside the cuSolver library. This is usually caused by a cudaMalloc() failure");
	if(error == CUSOLVER_STATUS_INVALID_VALUE)
		throw runtime_error("An unsupported value or parameter was passed to the function (a negative vector size, for example)");
	if(error == CUSOLVER_STATUS_ARCH_MISMATCH)
		throw runtime_error("The function requires a feature absent from the device architecture; usually caused by the lack of support for atomic operations or double precision");
	if(error == CUSOLVER_STATUS_EXECUTION_FAILED)
		throw runtime_error("The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons");
	if(error == CUSOLVER_STATUS_INTERNAL_ERROR)
		throw runtime_error("An internal CuSolver operation failed. This error is usually caused by a cudaMemcpyAsync() failure");
	if(error == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
		throw runtime_error("The matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function");
}
void cudaCheck(string file, unsigned long line)
{
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		char str[200];
		sprintf(str,"cuda error %s: line %d	%s\n",file.c_str(),line,cudaGetErrorString(error));
		throw runtime_error(str);
	}
}

#endif /* __CUDATA_CU__ */
