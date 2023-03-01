#ifndef __GIRAF__
#define __GIRAF__
#include "cudata.cuh"
#define SOLVE_LS_TYPE "CG" //solve type for Least Squares subproblem
//cuda kernels
static __global__ void cuxToGMap(unsigned long numGpixels, unsigned long *cuGmap, unsigned long *Gdims, unsigned long numGdims, unsigned long *xDims, unsigned long numxDims, unsigned long *fDims, unsigned long numfDims)
{
	/*This kernel assumes row major indexing*/
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numGpixels)
	{
			long long dimsProduct, row, col;
			long long *longDims = new long long [numfDims];
			long long sum;
		/*column calculation*/
			unsigned long *col_vector = new unsigned long [numfDims];
			col = i % Gdims[0];//Gdims[0] = number of filter pixels
			col_vector = RI2IV(col,col_vector,fDims,numfDims);
			col = IV2RI(col_vector,xDims,numxDims);
		/*row calculation*/
			unsigned long *row_vector = new unsigned long[numfDims];
			row	= i / Gdims[0];
			row_vector = RI2IV(row,row_vector,fDims,numfDims);
			row = IV2RI(row_vector,xDims,numxDims);
		/*copy unsigned long dims into long long dims array*/
			for(int j=0; j<numxDims; j++)
				longDims[j] = xDims[j];
		/**/
			sum = 0;
			for(int j=0; j<numxDims; j++)
			{
				dimsProduct = 1;
				for(int k=(j+1); k<numxDims; k++)
					dimsProduct *= longDims[k];
				sum += dimsProduct * MOD( ((col/dimsProduct)-(row/dimsProduct)) , longDims[j] );
			}
			cuGmap[i] = sum;
			delete[] longDims;
			delete[] col_vector;
			delete[] row_vector;
	}
}
static __global__ void cuMakeG(unsigned long numPixels, unsigned long *cuGmap, cuDoubleComplex *buffer, cuDoubleComplex *cuG, double epsilon, unsigned long diag_indx_spacing)
{
	/*This kernel assumes row major indexing*/
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPixels)
	{
		cuG[i].x = buffer[cuGmap[i]].x;
		cuG[i].y = buffer[cuGmap[i]].y;
		if(i%diag_indx_spacing == 0)
			cuG[i].x += epsilon;
	}
}
static __global__ void cuMakeG2GpaddedMap(unsigned long numGpaddedPixels, unsigned long numGdims, unsigned long *Gdims, unsigned long *GpaddedDims, unsigned long *G2GpaddedMap, bool isRCC)
{
	/*This kernel assumes row major indexing*/
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numGpaddedPixels)
	{
		unsigned long *GpaddedIV = new unsigned long[numGdims];
		GpaddedIV = RI2IV(i,GpaddedIV,GpaddedDims,numGdims);
		for(int k=1; k<numGdims; k++)
		{
			if(GpaddedIV[k] >= Gdims[k])
			{
				G2GpaddedMap[i] = numGpaddedPixels;
				delete[] GpaddedIV;
				return;
			}
		}
		if(isRCC == true)
		{
			for(int k=1; k<numGdims; k++)
				GpaddedIV[k] = (Gdims[k]-1) - GpaddedIV[k];
		}
		G2GpaddedMap[i] = IV2RI(GpaddedIV,Gdims,numGdims);
		delete[] GpaddedIV;
	}
}
static __global__ void cuMakeh2ifftDmap(unsigned long numDpixels, unsigned long numDims, unsigned long *hDims, unsigned long *Ddims, unsigned long *h2ifftDmap)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numDpixels)
	{
		unsigned long *hIV = new unsigned long[numDims];//gram matrix index vector
		unsigned long *DIV = new unsigned long[numDims];//D index vector
		unsigned long centerIndex;
		DIV = RI2IV(i,DIV,Ddims,numDims);//calculate the index vector for the i'th filter element
		for(int k=0; k<numDims; k++)//run through all index vector dimensions and map the filter index to D index
		{
			centerIndex = hDims[k]/2;//
			if(DIV[k] <= centerIndex)//if before or on center index of D
				hIV[k] = (hDims[k]-1) - centerIndex + DIV[k] ;//wrap around
			else if(DIV[k] < (Ddims[k] - centerIndex))
			{
				h2ifftDmap[i] = numDpixels;
				delete[] hIV;
				delete[] DIV;
				return;
			}
			else//these will be pushed closer to DC (fIV[div] ends up as DC)
				hIV[k] = DIV[k] - (Ddims[k] - centerIndex);
		}
		h2ifftDmap[i] = IV2RI(hIV,hDims,numDims);
		delete[] hIV;
		delete[] DIV;
	}
}
static __global__ void cuMapG2paddedBuffer(unsigned long numPixels, cuDoubleComplex *unpaddedBuffer, cuDoubleComplex *paddedBuffer, unsigned long *map, bool complexConj)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPixels)
	{
		if(map[i] == numPixels)
		{
			paddedBuffer[i].x = 0;
			paddedBuffer[i].y = 0;
		}
		else
		{
			paddedBuffer[i].x = unpaddedBuffer[map[i]].x;
			if(complexConj == false)
				paddedBuffer[i].y = unpaddedBuffer[map[i]].y;
			else
				paddedBuffer[i].y = -unpaddedBuffer[map[i]].y;

		}
	}
}
static __global__ void cuMaph2ifftD(unsigned long numPixels, cuDoubleComplex *hBuffer, cuDoubleComplex *Dbuffer, unsigned long *map)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPixels)
	{
		if(map[i] == numPixels)
		{
			Dbuffer[i].x = 0;
			Dbuffer[i].y = 0;
		}
		else
		{
			Dbuffer[i].x = hBuffer[map[i]].x;
			Dbuffer[i].y = hBuffer[map[i]].y;
		}
	}
}
static __global__ void cuMakeffth(unsigned long numhPixels, unsigned long *hDims, unsigned long *Gdims, unsigned long numGdims, cuDoubleComplex *h, cuDoubleComplex *G_padded, cuDoubleComplex *G_rcc_padded, double *eVals, double q, double e)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numhPixels)
	{
		unsigned long *hIV = new unsigned long[numGdims-1];
		hIV = RI2IV(i,hIV,hDims,numGdims-1);
		unsigned long *GIV = new unsigned long[numGdims];
		for(int k=1; k<numGdims; k++)
			GIV[k] = hIV[k-1];
		unsigned long indx;
		h[i].x = 0;
		h[i].y = 0;
		//double penalty;
		for(unsigned long k=0; k<Gdims[0]; k++)
		{
			GIV[0] = k;
			indx = IV2RI(GIV,Gdims,numGdims);
			h[i].x += (G_padded[indx].x*G_rcc_padded[indx].x - G_padded[indx].y*G_rcc_padded[indx].y)*pow(sqrt(eVals[k])+e,-2.0*q);
			h[i].y += (G_padded[indx].x*G_rcc_padded[indx].y + G_rcc_padded[indx].x*G_padded[indx].y)*pow(sqrt(eVals[k])+e,-2.0*q);
		}
		delete[] hIV;
		delete[] GIV;
	}
}
static __global__ void cuCalc_filter(unsigned long numfPixels, unsigned long *fDims, unsigned long *Gdims, unsigned long numGdims, cuDoubleComplex *f, cuDoubleComplex *G, double *eVals, double q, double e)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numfPixels)
	{
		unsigned long *fIV = new unsigned long[numGdims-1];
		fIV = RI2IV(i,fIV,fDims,numGdims-1);
		unsigned long *GIV = new unsigned long[numGdims];
		for(int k=1; k<numGdims; k++)
			GIV[k] = fIV[k-1];
		unsigned long indx;
		f[i].x = 0;
		f[i].y = 0;
		for(unsigned long k=0; k<Gdims[0]; k++)
		{
			GIV[0] = k;
			indx = IV2RI(GIV,Gdims,numGdims);
			f[i].x += G[indx].x * pow(sqrt(eVals[k])+e,-q);
			f[i].y += G[indx].y * pow(sqrt(eVals[k])+e,-q);
		}
		delete[] fIV;
		delete[] GIV;
	}
}
static __global__ void cuCalcADMMy(unsigned long numPixels, cuDoubleComplex *y, cuDoubleComplex *D, double gamma)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPixels)
	{
		y[i].x *= gamma / (D[i].x + gamma);
		y[i].y *= gamma / (D[i].y + gamma);
	}
}
static __global__ void cuCalcADMMx(unsigned long numPixels, cuDoubleComplex *x, cuDoubleComplex *A, cuDoubleComplex *b, double lambda, double gamma)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPixels)
	{
		//x = (x*gamma*lambda + Ab)/(A + lambda*gamma)
		x[i].x = (x[i].x*gamma*lambda+A[i].x*b[i].x) / (A[i].x + lambda*gamma);
		x[i].y = (x[i].y*gamma*lambda+A[i].x*b[i].y) / (A[i].x + lambda*gamma);
	}
}
static __global__ void cuCalcADMMq_nm1(unsigned long numPixels, cuDoubleComplex *q_nm1, cuDoubleComplex *x, cuDoubleComplex *y)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPixels)
	{
		//q_nm1 = q_nm1 + y - x
		q_nm1[i].x += y[i].x - x[i].x;
		q_nm1[i].y += y[i].y - x[i].y;
	}
}
static __global__ void cuCalcRank(unsigned long numPixels, double *eVals, double* rankArr, double e, double p)
{
	unsigned long i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<numPixels)
	{
		rankArr[i] = eVals[i] / pow( (sqrt(eVals[i]) + e) , 2.0 - p );
	}
}

//function headers
/*computes the map of x to G*/
unsigned long *xToGmap(cudata<cuDoubleComplex>&,cudata<cuDoubleComplex>&,const cudata<cuDoubleComplex>&,unsigned long*);
/*calculate G to G_padded map*/
unsigned long* makeG2GpaddedMap(cudata<cuDoubleComplex> &, cudata<cuDoubleComplex> &, unsigned long *, bool);
/*calculate h to ifftD map*/
unsigned long* makeh2ifftDmap(cudata<cuDoubleComplex> &, cudata<cuDoubleComplex> &, unsigned long *);
/*computes the Gram matrix (G) for GIRAF algorithm via G index map*/
void makeG(cudata<cuDoubleComplex>&,cudata<cuDoubleComplex>&,unsigned long*, double epsilon);
/*prints G index map to file*/
void printxToGmap(cudata<cuDoubleComplex>&,unsigned long*);
void map_G2paddedBuffer(cudata<cuDoubleComplex>&, cudata<cuDoubleComplex>&, unsigned long*, bool);
void map_h2ifftD(cudata<cuDoubleComplex>&, cudata<cuDoubleComplex>&, unsigned long*);
/*multiply tempD1 by complex conjugate and weight by eigenvalue*/
double cgCalcBeta(cudata<cuDoubleComplex> &, cudata<cuDoubleComplex> &);
void make_ffth(cudata<cuDoubleComplex>&, cudata<cuDoubleComplex>&, cudata<cuDoubleComplex>&, double*, double, double);
void calc_filter(cudata<cuDoubleComplex> &, cudata<cuDoubleComplex> &, double *, double, double);
inline void calcADMMy(cudata<cuDoubleComplex> &, cudata<cuDoubleComplex> &, double);
inline void calcADMMx(cudata<cuDoubleComplex> &, cudata<cuDoubleComplex> &, cudata<cuDoubleComplex> &, double, double);
inline void calcADMMq_nm1(cudata<cuDoubleComplex> &, cudata<cuDoubleComplex> &, cudata<cuDoubleComplex> &);
inline double calcRank(cudata<cuDoubleComplex> &, double*, double, double);

class feed_GIRAF
{
private:
	cudata<cuDoubleComplex> x;
	cudata<cuDoubleComplex> f;
public:
	cudata<cuDoubleComplex> b, Mx, fftMx, g, G, G_padded, G_rcc_padded, h, D, rhs, cg1, cg2, r_cg, p_cg;
	cufft_struct *cufft1, *cufft2, *cufft3;
	cusolver_struct cusolver1;
	unsigned long *Gdims, *Gmap, *G2GpaddedMap, *G2GrccPaddedMap, *h2ifftDmap;
	feed_GIRAF(cudata<cuDoubleComplex> x, cudata<cuDoubleComplex> f): x(x), f(f), b(x), Mx(x), fftMx(x), g(x), D(x), rhs(x), cg1(x), cg2(x), r_cg(x), p_cg(x)
	{
		cufft1 = new cufft_struct(Mx.dims,Mx.numDims);
		cufft1->plan();
	/*construct filtered gram Matrix (G) cudata object*/
		Gdims = new unsigned long[f.numDims+1];
		Gdims[0] = f.numPixels;
		for(int i=1; i<(f.numDims+1); i++)
			Gdims[i] = f.dims[i-1];
		G.reconstruct(Gdims,f.numDims+1);//filtered gram matrix of size (f.numPixels)^2, extracted from the larger unfiltered gram matrix of size (x.numPixels)^2 (except for the first row, the larger gram matrix is never entirely computed in this p
		G.cublas_handle = x.cublas_handle;
	/*calculate x to G matrix map*/
		Gmap = xToGmap(G,x,f,Gmap);
	/*construct G_padded, G_rcc_padded and h objects based off of G, and then initialize their own cufft plan*/
		unsigned long *GpaddedDims = new unsigned long [G.numDims];
		unsigned long *hDims = new unsigned long [f.numDims];
		GpaddedDims[0] = G.dims[0];//this is just the number of total pixels in filter (i.e. number of eigenvectors)
		for(int i=1; i<G.numDims; i++)
		{
			GpaddedDims[i] = 2*G.dims[i] - 1;
			hDims[i-1] = 2*f.dims[i-1] - 1;
		}
		G_padded.reconstruct(GpaddedDims,G.numDims);
		G_rcc_padded.reconstruct(GpaddedDims,G.numDims);
		h.reconstruct(hDims,f.numDims);
		delete[] GpaddedDims;
		delete[] hDims;
		G_padded.cublas_handle = x.cublas_handle;
		G_rcc_padded.cublas_handle = x.cublas_handle;
		h.cublas_handle = x.cublas_handle;
		cufft2 = new cufft_struct(G_padded.dims,G_padded.numDims,0);//batchDim = 0 ensures that the fft is not actually computed over G_padded.dims[0], but rather batched across G_padded.dims[0]
		cufft2->plan();
		cufft3 = new cufft_struct(h.dims,h.numDims);
		cufft3->plan();
	/*calculate G to G_padded map*/
		G2GpaddedMap = makeG2GpaddedMap(G,G_padded,G2GpaddedMap,false);
	/*calculate G to G_rcc_padded map*/
		G2GrccPaddedMap = makeG2GpaddedMap(G,G_rcc_padded,G2GrccPaddedMap,true);
	/*construct D*/
		D.setConst(0,0);
	/*calculate h to fftD map*/
		h2ifftDmap = makeh2ifftDmap(h,D,h2ifftDmap);

	}
	~feed_GIRAF()
	{
		delete cufft1;
		delete cufft2;
		delete cufft3;
		delete[] Gdims;
		cudaFree(G2GpaddedMap);
		cudaFree(G2GrccPaddedMap);
		cudaFree(h2ifftDmap);
		cudaFree(Gmap);
	}
};
/*GIRAF function*/
inline int GIRAF(cudata<cuDoubleComplex>&,cudata<cuDoubleComplex>&,cudata<cuDoubleComplex>&,cudata<cuDoubleComplex>&,double,double,int,feed_GIRAF&,double*, string LSsolver, double del);
inline int GIRAF(cudata<cuDoubleComplex> &x, cudata<cuDoubleComplex> &M, cudata<cuDoubleComplex> &A, cudata<cuDoubleComplex> &f, double p, double lambda, int numCycles, feed_GIRAF &mem, double *cost, string LSsolver, double del)
{
	try
	{
		//initialize necessary variables
			int max_CG_cycles = 1000;
			double q = 1.0 - p/2.0;
			double e, e_min;
			double gamma;
			mem.b = x;
			mem.Mx = x;
			mem.Mx.isNonDiagonal = true;
			mem.rhs = mem.b;
			cuDoubleComplex alpha, cgTemp;
			double beta, r_new_norm, r_old_norm;
			cudata<cuDoubleComplex> y(x), q_nm1(x);
			q_nm1.setConst(0,0);
		//for loop over GIRAF cycles
			for(int n=0; n<numCycles; n++)
			{
				cout << "GIRAF Cycle:" << n+1 << "/" << numCycles << endl;
				//Build G and calculate eigen-decomposition
					//compute g:
						//compute Mx
							cudaDeviceSynchronize();
							mem.Mx = M*x;
						//compute fft(Mx)
							cudaDeviceSynchronize();
							mem.Mx.fft(mem.cufft1, CUFFT_FORWARD, mem.fftMx);
							//cudaDeviceSynchronize();
							//mem.fftMx /= sqrt((double)mem.fftMx.numPixels);
						//compute element-wise absolute square of fftMx and store in g
							cudaDeviceSynchronize();
							mem.fftMx.ewAbsSq(mem.g);
						//compute g = fft(|fft(Mx)|^2)
							cudaDeviceSynchronize();
							mem.g.fft_(mem.cufft1,CUFFT_INVERSE);
							//cudaDeviceSynchronize();
							//mem.g /= (double)(mem.g.numPixels);//unitary accounting
					//calculate G from g via filter support map
						cudaDeviceSynchronize();
						makeG(mem.g,mem.G,mem.Gmap,0.0);
					//Compute eigen-decomposition of G
						cudaDeviceSynchronize();
						mem.G.eigen(&mem.cusolver1);
						cudaMemcpy(&e, &mem.cusolver1.cuW[f.numPixels-1], sizeof(double),cudaMemcpyDeviceToHost);//note that cuSolver returns largest eigenvalue last (i.e. ascending order), which is why I use index: f.numPixels - 1
                        cudaMemcpy(&e_min, &mem.cusolver1.cuW[0], sizeof(double),cudaMemcpyDeviceToHost);//note that cuSolver returns smallest eigenvalue first (i.e. ascending order), which is why I use index: 0	
                        cout << "	min S.V. = " << sqrt(e_min) << endl;
                        cout << "	max S.V. = " << sqrt(e) << endl;
						e = sqrt(sqrt(e*e_min));
                        cout << "	epsilon = " << e << endl;
                        cout << "    rank = " << calcRank(f,mem.cusolver1.cuW,p,e) << endl;
					//map G to G_padded and G_rcc_padded (reverse complex conjugate) and insert zero padding
						map_G2paddedBuffer(mem.G,mem.G_padded,mem.G2GpaddedMap,false);
						map_G2paddedBuffer(mem.G,mem.G_rcc_padded,mem.G2GrccPaddedMap,true);
					//fft in-place both G_padded and G_rcc_padded
						mem.G_padded.fft_(mem.cufft2,CUFFT_FORWARD);
						mem.G_rcc_padded.fft_(mem.cufft2,CUFFT_FORWARD);
						cudaDeviceSynchronize();
					//multiply G_padded by G_rcc_padded, reweight by eigenvalue and sum into h
						make_ffth(mem.h,mem.G_padded,mem.G_rcc_padded,mem.cusolver1.cuW,q,e);
						//mem.h /= mem.h.nrm2();
						cout << "	||FFT(h)|| = " << mem.h.nrm2() << endl;
					//ifft in-place
						mem.h.fft_(mem.cufft3,CUFFT_INVERSE);
						cudaDeviceSynchronize();
						mem.h /= (double)(mem.h.numPixels);
					//map h to D (note ifftshift required) and fft in-place
						map_h2ifftD(mem.h,mem.D,mem.h2ifftDmap);
						mem.D.fft_(mem.cufft1,CUFFT_FORWARD);
						//mem.D /= sqrt((double)(mem.D.numPixels));
						cout << "	||D|| = " << mem.D.nrm2() << endl;
				//Solve Least Squares Problem
					if(LSsolver == "CG")//solve by Conjugate Gradient ( lhs*x = rhs )
					{
						//compute cg1 = lhs*x
							x.fft(mem.cufft1,CUFFT_FORWARD,mem.cg1);
							//missing multiplication by 'M' here
							mem.cg1.multiply(mem.D,mem.cg1);
							mem.cg1.fft_(mem.cufft1,CUFFT_INVERSE);
							mem.cg1 /= (double)(mem.cg1.numPixels);
							mem.cg2.multiply(A,x);
							mem.cg1.add(mem.cg2,lambda,mem.cg1);
							mem.r_cg.subtract(mem.rhs,mem.cg1);//r0 = rhs - lhs*x
							mem.p_cg = mem.r_cg;
						for(int k=0; k<max_CG_cycles; k++)
						{
							r_old_norm = mem.r_cg.nrm2();
							//compute alpha
								//compute cg1 = lhs*p
									mem.p_cg.fft(mem.cufft1,CUFFT_FORWARD,mem.cg1);//cg1 = Fp
									mem.cg1.multiply_(mem.D);//cg1 = DFx
									mem.cg1.fft_(mem.cufft1,CUFFT_INVERSE);//cg1 = F*DFp
									mem.cg1 /= (double)(mem.cg1.numPixels);
									mem.cg2.multiply(A,mem.p_cg);//cg2 = Ap
									mem.cg1.add(mem.cg2,lambda,mem.cg1);//cg1 = (A+lambdaF*DF)p
								//compute cgTemp = < p , (A+lambdaF*DF)p >
									cgTemp = mem.p_cg.dot(mem.cg1);
								//compute alpha =
									alpha.x = (r_old_norm*r_old_norm)*(cgTemp.x)/(cgTemp.x*cgTemp.x+cgTemp.y*cgTemp.y);
									alpha.y = -(r_old_norm*r_old_norm)*(cgTemp.y)/(cgTemp.x*cgTemp.x+cgTemp.y*cgTemp.y);
							//update x
								x.axpy(alpha,mem.p_cg);
							//compute r_new
								mem.r_cg.axmy(alpha,mem.cg1);
							//compute |r_new| and break if necessary
								r_new_norm = mem.r_cg.nrm2();
								if(r_new_norm < 1e-8)
								{
									cout << "   CG break at cycle " << k << " with |r| = " << r_new_norm << endl;
									break;
								}
								if(k == max_CG_cycles-1)
								{
									n = numCycles-1;
									cout << "Early break due to max CG cycles reached" << endl;
								}
							//compute beta
								beta = (r_new_norm*r_new_norm)/(r_old_norm*r_old_norm);
							//update p_cg
								mem.p_cg.add(mem.r_cg,beta,mem.p_cg);
						}
					}
					else if(LSsolver == "ADMM")
					{
						//set gamma = max(D) / delhttp://mathworld.wolfram.com/HyperbolicTangent.html
							gamma = mem.D.max_magnitude() / del;
							cout << "	gamma = " << gamma << endl; 
						//y = x - q_nm1
							y.subtract(x,q_nm1);
						//y = FFT(y) (in place)
							y.fft_(mem.cufft1,CUFFT_FORWARD);
							y /= sqrt((double)(y.numPixels));
						//y = y*gamma/(D + gamma)
							calcADMMy(y,mem.D,gamma);
						//x = y + q_nm1
							x.add(y,q_nm1);
						//x = iFFT(x) (in place)
							x.fft_(mem.cufft1,CUFFT_INVERSE);
							x /= sqrt((double)(x.numPixels));
						//x = (x*gamma*lambda + Ab)/(A + lambda*gamma)
							calcADMMx(x,A,mem.b,lambda,gamma);
						//y = iFFT(y) (jsut doing this for q_nm1 calculation below)
							y.fft_(mem.cufft1,CUFFT_INVERSE);
							y /= sqrt((double)(y.numPixels));
						//q_nm1 = q_nm1 + y - x
							calcADMMq_nm1(q_nm1,x,y);
					}
					else
					{
						throw runtime_error("Invalid Least Squares Method");
					}
				//Decide if early exit
					cout << "	||x|| = " << x.nrm2() << endl;
			}
		//compute cost squares: C(x) = ||Ax-b||^2 + (||T(x)||_p)^p = ||Ax-b||^2 + ||sqrt(D)*F*Mx||^2 where sigma_i^p is the i'th eigenvalue of the Gram matrix
			cudata<cuDoubleComplex> cost_temp1(mem.Mx);
			cudata<cuDoubleComplex> cost_temp2(mem.Mx);
			cost_temp1.multiply(A,x);
			cost_temp1.subtract(cost_temp1,mem.b);
			double cost_df = cost_temp1.nrm2();
			cost_df *= cost_df;
			cout << "||Ax-b||^2 = " << cost_df << endl;
			double eVals[f.numPixels];
			cudaCheck("cudaMemcpy eValline(1:N) = s2; %Create full mask for zeropadded data", cudaMemcpy(eVals,mem.cusolver1.cuW,f.numPixels*sizeof(unsigned long),cudaMemcpyDeviceToHost));
			cost_temp2.cuSqrt(mem.D);
			x.fft(mem.cufft1,CUFFT_FORWARD,cost_temp1);
			cost_temp1 /= sqrt(cost_temp1.numPixels);
			cudaDeviceSynchronize();
			cost_temp2.multiply_(cost_temp1);
			cudaDeviceSynchronize();
			double cost_rt = cost_temp2.nrm2();
			cost_rt *= cost_rt;
			cout << "\n||sqrt(D)*F*x||^2 = " << cost_rt << endl;
			*cost = sqrt(cost_rt + cost_df);
			cout << "Cost = " << *cost << endl;
			/*stringstream ss;
			ofstream ofs;
			ss << "outputs//final_evals_lambda_10e" << log10(lambda) << ".txt" << endl;
			ofs.open(ss.str());
			for(int i=0;i<f.numPixels;i++)
				ofs << eVals[i] << endl;
			ofs.close();
			ss.clear();*/
		//calculate effective filter - this is purely for supplementary analysis and not required by GIRAF algorithm
			calc_filter(f,mem.G,mem.cusolver1.cuW,q,e);
		//GIRAF completion message
			cout << "\nGIRAF Complete\n";
			return 0;
	}
	catch(const exception& e)
	{
		cout << "\nGIRAF prematurely terminated by catching exception: " << e.what() << endl;
		return 1;
	}
}
/***functions***/
/*computes the map of x to G*/
inline unsigned long *xToGmap(cudata<cuDoubleComplex> &G, cudata<cuDoubleComplex> &x, const cudata<cuDoubleComplex> &f, unsigned long *cuGmap)
{
	//allocate Gmap to same size as G
		cudaCheck("cudaMalloc Gmap",cudaMalloc((void**)&cuGmap,G.numPixels*sizeof(unsigned long)));
	//execute the kernel that calculates the x To G map
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in calcGramSlidingWindowIndices()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuxToGMap,0,0));
		gridSize = (G.numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cuxToGMap<<<gridSize,blockSize>>>(G.numPixels,cuGmap,G.cuDims,G.numDims,x.cuDims,x.numDims,f.cuDims,f.numDims);
		cudaDeviceSynchronize();
		//printxToGmap(G,cuGmap);
		return cuGmap;
}
/*computes the Gram matrix (G) for GIRAF algorithm via G index map*/
inline void makeG(cudata<cuDoubleComplex> &g, cudata<cuDoubleComplex> &G, unsigned long *cuGmap, double epsilon)
{
	/*make sure G is indexed row major*/
		g.rowMajor();
	/*execute build-gram-matrix-from-map kernel*/
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in makeG()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuMakeG,0,0));
		gridSize = (G.numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cuMakeG<<<gridSize,blockSize>>>(G.numPixels,cuGmap,g.buffer,G.buffer,epsilon,sqrt(G.numPixels)+1);
		cudaDeviceSynchronize();
}
/*prints G index map to file*/
inline void printxToGmap(cudata<cuDoubleComplex> &G, unsigned long *cuGmap)
{
	unsigned long *Gmap = new unsigned long [G.numPixels];
	cudaCheck("cudaMemcpy Gmap",cudaMemcpy(Gmap,cuGmap,G.numPixels*sizeof(unsigned long),cudaMemcpyDeviceToHost));
	ofstream ofs("Gmap.txt");
	//print dimensions on first line
		for(int i=0; i<G.numDims; i++)
			ofs << G.dims[i] << "\t";
		ofs << endl;
	unsigned long CI, RI;
	unsigned long *IV = new unsigned long[G.numDims];
	for(CI=0; CI<G.numPixels; CI++)
	{
			IV = CI2IV(CI,IV,G.dims,G.numDims);
			RI = IV2RI(IV,G.dims,G.numDims);
			ofs << Gmap[RI] << '\t';
		/*print delimiting characters*/
			bool endlFlag=true;
			for(int i=0; i<G.numDims; i++)
			{
				for(int k=0; k<=i; k++)
				{
					if(IV[k] != (G.dims[k]-1))
						endlFlag = false;
				}
				if(endlFlag)
					ofs << endl;
			}
	}
	ofs.close();
	ofs.open("GmapRowMajor.txt");
	//print dimensions on first line
		for(int i=0; i<G.numDims; i++)
			ofs << G.dims[i] << "\t";
		ofs << endl;
		for(int i=0; i<144; i++)
		{
			ofs << Gmap[i] << "\t";
			if((i+1)%12 == 0)
				ofs<< endl;
		}
	ofs.close();
	delete[] Gmap;
	delete[] IV;

}
/*calculate G to G_padded map*/
inline unsigned long* makeG2GpaddedMap(cudata<cuDoubleComplex> &G, cudata<cuDoubleComplex> &G_padded, unsigned long *G2GpaddedMap, bool isRCC)
{
	//allocate device memory
		cudaCheck("cudaMalloc cuG2GpaddedMap",cudaMalloc((void**)&(G2GpaddedMap),G_padded.numPixels*sizeof(unsigned long)));
	//calculate map
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in makeG2GpaddedMap()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuMakeG2GpaddedMap,0,0));
		gridSize = (G_padded.numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cuMakeG2GpaddedMap<<<gridSize,blockSize>>>(G_padded.numPixels,G.numDims,G.cuDims,G_padded.cuDims,G2GpaddedMap,isRCC);
		cudaDeviceSynchronize();
		return G2GpaddedMap;
}
/*calculate h to ifftD map*/
inline unsigned long* makeh2ifftDmap(cudata<cuDoubleComplex> &h, cudata<cuDoubleComplex> &D, unsigned long *h2ifftDmap)
{
	//allocate device memory
		cudaCheck("cudaMalloc h2ifftDmap",cudaMalloc((void**)&(h2ifftDmap),D.numPixels*sizeof(unsigned long)));
	//calculate map
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in cuMakeh2ifftDmap()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuMakeh2ifftDmap,0,0));
		gridSize = (D.numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cuMakeh2ifftDmap<<<gridSize,blockSize>>>(D.numPixels,D.numDims,h.cuDims,D.cuDims,h2ifftDmap);
		cudaDeviceSynchronize();
	return h2ifftDmap;
}
inline void map_G2paddedBuffer(cudata<cuDoubleComplex> &G, cudata<cuDoubleComplex> &G_padded, unsigned long *map, bool complexConj)
{
	//calculate map
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in cuMapG2paddedBuffer()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuMapG2paddedBuffer,0,0));
		gridSize = (G_padded.numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cuMapG2paddedBuffer<<<gridSize,blockSize>>>(G_padded.numPixels,G.buffer,G_padded.buffer,map,complexConj);
		cudaDeviceSynchronize();
}
inline void map_h2ifftD(cudata<cuDoubleComplex> &h, cudata<cuDoubleComplex> &ifftD, unsigned long *map)
{
	//calculate map
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in cuMaph2ifftD()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuMaph2ifftD,0,0));
		gridSize = (ifftD.numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cuMaph2ifftD<<<gridSize,blockSize>>>(ifftD.numPixels,h.buffer,ifftD.buffer,map);
		cudaDeviceSynchronize();
}
inline void make_ffth(cudata<cuDoubleComplex> &h, cudata<cuDoubleComplex> &G_padded, cudata<cuDoubleComplex> &G_rcc_padded, double *eVals, double q, double e)
{
	//calculate map
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in cuMakeiffth()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuMakeffth,0,0));
		gridSize = (h.numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cuMakeffth<<<gridSize,blockSize>>>(h.numPixels,h.cuDims,G_padded.cuDims,G_padded.numDims,h.buffer,G_padded.buffer,G_rcc_padded.buffer,eVals,q,e);
		cudaDeviceSynchronize();
}
inline double cgCalcBeta(cudata<cuDoubleComplex> &r_old, cudata<cuDoubleComplex> &r_new)
{
	double num, denom;
	cudaCheck("cublasDznrm2 for numerator in cgCalcBeta()",cublasDznrm2(*(r_new.cublas_handle),(int)(r_new.numPixels),r_new.buffer,1,&num));
	cudaCheck("cublasDznrm2 for denominator in cgCalcBeta()",cublasDznrm2(*(r_old.cublas_handle),(int)(r_old.numPixels),r_old.buffer,1,&denom));
	return (num*num)/(denom*denom);
}
/*inline void makePaddedCopies(cudata x_UP, cudata M_UP, cudata A_UP, cudata &x, cudata &M, cudata &A, cudata &f)
{
	unsigned long *padSize = new unsigned long[x_UP.numDims];
	unsigned long *padIndxStart = new unsigned long[x_UP.numDims];
	for(int i=0; i<x_UP.numDims; i++)
	{
		padSize[i] = 2*f.dims[i];
		padIndxStart[i] = (x_UP.dims[i] + 1) / 2;
	}
	cuDoubleComplex padVal;
	padVal.x = 0;
	padVal.y = 0;
	x_UP.contiguousPad(x,padIndxStart,padSize,padVal);
	A_UP.contiguousPad(A,padIndxStart,padSize,padVal);
	delete[] padSize;
	delete[] padIndxStart;
}
inline void copyBackUnpadded(cudata &x_UP, cudata &x, cudata &f)
{
	unsigned long *cutSize = new unsigned long[x_UP.numDims];
	unsigned long *cutIndxStart = new unsigned long[x_UP.numDims];
	for(int i=0; i<x_UP.numDims; i++)
	{
		cutSize[i] = 2*f.dims[i];
		cutIndxStart[i] = (x_UP.dims[i] + 1) / 2;
	}
	x.contiguousCut(x_UP,cutSize,cutIndxStart);
	delete[] cutSize;
	delete[] cutIndxStart;
}*/
inline void calc_filter(cudata<cuDoubleComplex> &f, cudata<cuDoubleComplex> &G, double *eVals, double q, double e)
{
	//calculate filter
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in calc_filter()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuCalc_filter,0,0));
		gridSize = (f.numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cuCalc_filter<<<gridSize,blockSize>>>(f.numPixels,f.cuDims,G.cuDims,G.numDims,f.buffer,G.buffer,eVals,q,e);
		cudaDeviceSynchronize();
}
inline void calcADMMy(cudata<cuDoubleComplex> &y, cudata<cuDoubleComplex> &D, double gamma)
{
	//calculate y for ADMM
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in calcADMMy()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuCalcADMMy,0,0));
		gridSize = (y.numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cuCalcADMMy<<<gridSize,blockSize>>>(y.numPixels,y.buffer,D.buffer,gamma);
		cudaDeviceSynchronize();
}
inline void calcADMMx(cudata<cuDoubleComplex> &x, cudata<cuDoubleComplex> &A, cudata<cuDoubleComplex> &b, double lambda, double gamma)
{
	//calculate y for ADMM
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in calcADMMx()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuCalcADMMx,0,0));
		gridSize = (x.numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cuCalcADMMx<<<gridSize,blockSize>>>(x.numPixels,x.buffer,A.buffer,b.buffer,lambda,gamma);
		cudaDeviceSynchronize();
}
inline void calcADMMq_nm1(cudata<cuDoubleComplex> &q_nm1, cudata<cuDoubleComplex> &x, cudata<cuDoubleComplex> &y)
{
	//calculate y for ADMM
		int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in calcADMMq_nm1()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuCalcADMMq_nm1,0,0));
		gridSize = (q_nm1.numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cuCalcADMMq_nm1<<<gridSize,blockSize>>>(q_nm1.numPixels,q_nm1.buffer,x.buffer,y.buffer);
		cudaDeviceSynchronize();
}
inline double calcRank(cudata<cuDoubleComplex> &f, double *eVals, double p, double e)
{
    //allocate device memory
        double *rankArr;
		cudaCheck("cudaMalloc rankArr",cudaMalloc((void**)&(rankArr),f.numPixels*sizeof(double)));
	//calculate rankArr
        int gridSize, blockSize, minGridSize;
		cudaCheck("Block size calculator in calcRank()",cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,cuCalcRank,0,0));
		gridSize = (f.numPixels+blockSize-1)/blockSize;
		cudaDeviceSynchronize();
		cuCalcRank<<<gridSize,blockSize>>>(f.numPixels, eVals, rankArr, e, p);
		cudaDeviceSynchronize();
    //calc rank
        double rank;
        cudaCheck("cublasDasum rank in calcRank()",cublasDasum(*(f.cublas_handle),(int)(f.numPixels),rankArr,1,&rank));
        cudaFree(rankArr);
        return rank;
}
#endif /* __GIRAF__ */
