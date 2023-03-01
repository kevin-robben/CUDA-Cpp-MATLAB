#ifndef __CUDATA_CUH__
#define __CUDATA_CUH__
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <complex.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <exception>
#include <stdexcept>
#include <ctime>
#include <cstdlib>
#include "mex.h"
#include "gpu/mxGPUArray.h"

using namespace std;
#define MAX_DIM 5

#define IDX2R5(indx, dims) (indx[4]+dims[4]*indx[3]+dims[4]*dims[3]*indx[2]+dims[4]*dims[3]*dims[2]*indx[1]+dims[4]*dims[3]*dims[2]*dims[1]*indx[0])
#define IDX2C5(indx, dims) (indx[0]+indx[1]*dims[0]+indx[2]*dims[0]*dims[1]+indx[3]*dims[0]*dims[1]*dims[2]+indx[4]*dims[0]*dims[1]*dims[2]*dims[3])
#define MOD(a,b)	 ( ((a%b)<0) ? ((a%b)+b) : (a%b) )

/*cuda error handling function*/
void cudaCheck(string description, cudaError_t error);
void cudaCheck(string description, cublasStatus_t error);
void cudaCheck(string description, cufftResult error);
void cudaCheck(string description, cusolverStatus_t error);
void cudaCheck(string file, unsigned long line);

typedef cufftDoubleComplex ComplexPixelDev;
/**/
struct cusolver_struct
{
	cusolverDnHandle_t cusolver_DnHandle;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_NOT_INITIALIZED;
	cusolverEigMode_t eigen_mode;
	cublasFillMode_t fill_mode;
	double *cuW = nullptr;
	cuDoubleComplex *cuWork = nullptr;
	int Lwork;
	int *cuInfo = nullptr;
	int hostInfo;
	~cusolver_struct();
};
/**/
struct cufft_struct
{
	cufftHandle cufft_handle;
	cufftResult cufft_result = CUFFT_INVALID_PLAN;
	bool batchDim0 = false;
	int rank;
	int *n = nullptr;
	int *inembed = nullptr;
	int istride = 1;
	int idist = 1;
	int *onembed = nullptr;
	int ostride = 1;
	int odist = 1;
	int batch = 1;
	~cufft_struct();
	cufft_struct(unsigned long *dims, int numDims);
	cufft_struct(unsigned long *dims, int numDims, int batchDim);
	void plan();
};
/**/
template <class T>
class cudata
{
public:
	unsigned long *dims = nullptr;
	unsigned long numDims=0;
	unsigned long numPixels=1;
	unsigned long *cuDims = nullptr;
	T *buffer = nullptr;
	bool isMexCuda = false;
	bool isIdentity = false;
	bool isDiagonal = false;
	bool isNonDiagonal = false;
	bool isSymmetric = false;
	bool isHermitian = false;
	bool isRowMajor = true;//if true then buffer is stored in row major format, else buffer is stored in column major format
	cublasHandle_t *cublas_handle = nullptr;//this will not be destroyed in destructor!

	//destructor
	~cudata()
	{
		if(isMexCuda == false)
			cudaFree(buffer);
		buffer = nullptr;
		cudaFree(cuDims);
		cuDims = nullptr;
		delete[] dims;
	}
	//default constructor
	cudata()
	{
		return;
	}
	//MATLAB mexcuda constructor
	cudata(mxArray const * const mp);
	//dims only constructor
	template <class longT1, class longT2>
	cudata(longT1 *dimsIn, longT2 numDimsIn)
	{
		//copy in dims array and calculate numDims and numPixels
			numPixels = 1;
			numDims = numDimsIn;
		/*allocate memory for host dims*/
			dims = new unsigned long [numDims];
			for(int i=0; i<numDims; i++)
			{
				dims[i] = dimsIn[i];
				numPixels *= dims[i];
			}
		/*allocate memory for device dims and copy host dims into it*/
			cudaFree(cuDims);
			cudaCheck("cudaMalloc for cuDims in dims only constructor",cudaMalloc((void**)&cuDims,numDims*sizeof(unsigned long)));
			updateCuDims();
		/*allocate memory for device buffer*/
			cudaCheck("cudaMalloc for buffer in dims only constructor",cudaMalloc((void**)&buffer,numPixels*sizeof(T)));
	}
	//dims only reconstructor
	template <class longT1, class longT2>
	void reconstruct(longT1 *dimsIn, longT2 numDimsIn)
	{
		//copy in dims array and calculate numDims and numPixels
			numPixels = 1;
			numDims = numDimsIn;
		/*allocate memory for host dims*/
			delete dims;
			dims = new unsigned long [numDims];
			for(int i=0; i<numDims; i++)
			{
				dims[i] = dimsIn[i];
				numPixels *= dims[i];
			}
		/*allocate memory for device dims and copy host dims into it*/
			cudaFree(cuDims);
			cudaCheck("cudaMalloc for cuDims in dims only constructor",cudaMalloc((void**)&cuDims,numDims*sizeof(unsigned long)));
			updateCuDims();
		/*allocate memory for device buffer*/
			cudaFree(buffer);
			cudaCheck("cudaMalloc for buffer in dims only constructor",cudaMalloc((void**)&buffer,numPixels*sizeof(T)));
	}
	/*read from file constructor*/
	cudata(string);
	/*deep copy constructor*/
	cudata(const cudata&);
	//write linear array to buffer
	void linear(T x0, T dx);
	//import real or imaginary part of image
	void import_real(string filename);
	void import_imag(string filename);
	/*import matlab complex data*/
	void import_matlab_complex(string filename);
	/*deep copy constructor*/
	cudata& operator=(const cudata&);
	void axpy(T, cudata &);
	void axmy(T, cudata &);
	/*element-wise addition operator*/
	void add(cudata &, cudata &);
	void add(cudata &, double, cudata &);
	/*element-wise subtraction operator*/
	void subtract(cudata &, cudata &);
	void subtract(cudata &, T, cudata &);
	/*element-wise division operator*/
	cudata& operator*=(double);
	cudata& operator/=(double);
	/*multiplication operator*/
	cudata& operator*(cudata&);
	void multiply(cudata&, cudata&);
	void multiply_(cudata&);
	/*returns euclidean norm of buffer (the sqrt of the absolute magnitude)*/
	double nrm2();
	void cuSqrt(cudata &);
	T dot(cudata &);
	/*return maximum magnitude of cudata array*/
	double max_magnitude();
	/*print buffer to file*/
	void printb2f(string, char);
	void printb2f(string, int, char);
	void printmap2f(string, unsigned long *);
	/*make sure buffer is in column major format*/
	void colMajor();
	/*make sure buffer is in row major format*/
	void rowMajor()
	{
		if(!isRowMajor)
			throw runtime_error("isRowMajor bool is false for an instance of cudata. Program cannot continue until bool is flagged true.");
	}
	/*descIndx*/
	void calcAscendingMap(unsigned long*);
	/*remap buffer indexing*/
	void mapIndx(unsigned long*);
	/*computes the element-wise absolute square of cudata object and returns the answer*/
	void ewAbsSq(cudata &);
	/*computes the element-wise absolute square of cudata object and returns the answer in-pace*/
	void ewAbsSq_();
	/*computes the multidimensional fourier transform of cudata buffer in-place*/
	void fft_(cufft_struct*, int);
	/*computes the multidimensional fourier transform of cudata buffer out-of-place*/
	void fft(cufft_struct*, int, cudata &);
	/*canvas-cut buffer*/
	void canvasCut(cudata &, unsigned long *, unsigned long *);
	/*contiguous-cut to buffer*/
	void contiguousCut(cudata &, unsigned long*, unsigned long*);
	/*contiguous-pad to buffer*/
	void contiguousPad(cudata &, unsigned long*, unsigned long*, T);
	/*canvas-pad to buffer*/
	void canvasPad(cudata &, unsigned long*, unsigned long*, T);
	/*computes the eigenvectors and eigenvalues of gram matrix*/
	void eigen(cusolver_struct *);
	/*prints out eigenvalues (eVals), eigenvectors (E), the two eigen-identity(ME = EeVals) matrices and their difference (ME-EeVals, a.k.a the secular which should be ~0)*/
	void eigenCheck(cusolver_struct*, cudata&);
	/*hemm inplace*/
	void hemm(cublasHandle_t, cudata &, cudata &);
	/*dgmm inplace*/
	void dgmm(cublasHandle_t, T*, cudata &);
	/*set buffer equal to constant value*/
	void setConst(double, double);
	/**/
	void updateCuDims()
	{
		if(cuDims == nullptr)
			cudaCheck("cudaMalloc of cuDims in updateCuDims",cudaMalloc((void**)&cuDims,numDims*sizeof(unsigned long)));
		cudaCheck("updating cuDims on device from dims on host",cudaMemcpy(cuDims,dims,numDims*sizeof(unsigned long),cudaMemcpyHostToDevice));
	}
	void boolCopy(const cudata &obj)
	{
		isDiagonal = obj.isDiagonal;
		isNonDiagonal = obj.isNonDiagonal;
		isHermitian = obj.isHermitian;
		isIdentity = obj.isIdentity;
		isRowMajor = obj.isRowMajor;
		isSymmetric = obj.isSymmetric;
		cublas_handle = obj.cublas_handle;
	}





	//need to define if row-major or col-major
	//need a function or string that gives cuData a label.
	//need to work on adding more error handling
};

/***Non-Member functions***/

/*computes the element-wise multiplication of two cudata buffers, and a constant, and returns product in out*/
void ewMultiply(cudata<cuDoubleComplex> &out, cuDoubleComplex a, cudata<cuDoubleComplex> &in1, cudata<cuDoubleComplex> &in2);
/*matrix vector multiplication*/
void matrixVectorMultiply(cublasHandle_t cublas_handle, cudata<cuDoubleComplex> &out, cuDoubleComplex a, cudata<cuDoubleComplex> &A, cudata<cuDoubleComplex> &x);
/*matrix matrix multiplication*/
void matrixMatrixMultiply(cublasHandle_t cublas_handle, cudata<cuDoubleComplex> &out, cuDoubleComplex a, cudata<cuDoubleComplex> &A, cuDoubleComplex b, cudata<cuDoubleComplex> &B);
/*geam out of place*/
void geam(cublasHandle_t,cudata<cuDoubleComplex>&,cudata<cuDoubleComplex>&,cudata<cuDoubleComplex>&);


/*index vector to linear column-major index*/
inline __host__ __device__ unsigned long IV2CI(unsigned long* IV, unsigned long *dims, unsigned long numDims)
{
	unsigned long CI = 0, dimProduct;
	for(int i=0; i<numDims; i++)
	{
		dimProduct = 1;
		for(int k=0; k<i; k++)
			dimProduct *= dims[k];
		CI += dimProduct*IV[i];
	}
	return CI;
}
/*linear column-major index to index vector*/
inline __host__ __device__ unsigned long* CI2IV(unsigned long CI, unsigned long* IV, unsigned long *dims, unsigned long numDims)
{
	unsigned long dimProduct;
	for(int i=0; i<numDims; i++)
	{
		dimProduct = 1;
		for(int k=0; k<i; k++)
			dimProduct *= dims[k];
		IV[i] = (CI/dimProduct) % dims[i];
	}
	return IV;
}
/*index vector to linear row-major index*/
inline __host__ __device__ unsigned long IV2RI(unsigned long* IV, unsigned long *dims, unsigned long numDims)
{
	unsigned long RI = 0, dimProduct;
	for(int i=0; i<numDims; i++)
	{
		dimProduct = 1;
		for(int k=(i+1); k<numDims; k++)
			dimProduct *= dims[k];
		RI += dimProduct*IV[i];
	}
	return RI;
}
/*linear row-major index to index vector*/
inline __host__ __device__ unsigned long* RI2IV(unsigned long RI, unsigned long* IV, unsigned long *dims, unsigned long numDims)
{
	unsigned long dimProduct;
	for(int i=0; i<numDims; i++)
	{
		dimProduct = 1;
		for(int k=(i+1); k<numDims; k++)
			dimProduct *= dims[k];
		IV[i] = (RI/dimProduct) % dims[i];
	}
	return IV;
}
/*linear column-major index to linear row-major index*/
inline __host__ __device__ unsigned long CI2RI(unsigned long CI, unsigned long* dims, unsigned long numDims)
{
	unsigned long RI;
	unsigned long* IV = new unsigned long[numDims];
	IV = CI2IV(CI,IV,dims,numDims);
	RI = IV2RI(IV,dims,numDims);
	delete[] IV;
	return RI;
}
/*linear row-major index to linear column-major index*/
inline __host__ __device__ unsigned long RI2CI(unsigned long RI, unsigned long* dims, unsigned long numDims)
{
	unsigned long CI;
	unsigned long* IV = new unsigned long[numDims];
	IV = RI2IV(RI,IV,dims,numDims);
	CI = IV2CI(IV,dims,numDims);
	delete[] IV;
	return CI;
}


#endif /* __CUDATA_CUH__ */
