#include "src/cudata.cu"
#include "src/GIRAF.cu"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
	//Initialize error log
		string str;
		str = "log.txt";
		streambuf *coutbuf = cout.rdbuf();//save old cout buffer
		ofstream out(str);//make new cout buffer
		cout.rdbuf(out.rdbuf());
    	//initialize the MathWorks GPU API
    		mxInitGPU();
	//Initialize cublas library
		cublasHandle_t cublas_handle;//Declare cublas library handle
		cublasCreate(&cublas_handle);//Initialize cublas library handle
	//Initialize cudata instance of image x
		cudata<cuDoubleComplex> x(prhs[0]);
		x.cublas_handle = &cublas_handle;
		x.isIdentity = false;
		x.isNonDiagonal = true;
	//Initialize cudata instance of filter f
		cudata<cuDoubleComplex> f(prhs[1]);
		f.cublas_handle = &cublas_handle;
	//Initialize cudata instance of mask
		cudata<cuDoubleComplex> mask(prhs[2]);
		mask.cublas_handle = &cublas_handle;
		mask.isIdentity = false;
		mask.isDiagonal = true;
		mask.cublas_handle = &cublas_handle;
	//Initialize M (this is identity for 2DIR routines)
		cudata<cuDoubleComplex> M;
		M.isIdentity = true;
		M.cublas_handle = &cublas_handle;
	//Initialize other GIRAF parameters
		int numCycles = (int)(*mxGetPr(prhs[3]));
		double lambda = (double)(*mxGetPr(prhs[4]));
		string LSsolver(mxArrayToString(prhs[5]),mxGetN(prhs[5]));
		double del = (double)(*mxGetPr(prhs[6]));
		double p = 0.0;/*This is the power "p" of the Schatten-p quasi norm*/
		cout << "LSsovler[" << mxGetN(prhs[5])+1 <<"] = " << LSsolver << endl;
		cout << "filterDims = [" << f.dims[0] << "," << f.dims[1] << "]" << endl;
		cout << "xDims = [" << x.dims[0] << "," << x.dims[1] << "]" << endl;
		cout << "lambda = " << lambda << endl;
		cout << "numCycles = " << numCycles << endl;
		double cost;
	//Run GIRAF
		feed_GIRAF snacks(x,f);//must preallocate memory before running GIRAF
        int recon_status = GIRAF(x,M,mask,f,p,lambda,numCycles,snacks,&cost,LSsolver,del);
		plhs[0] = mxCreateDoubleScalar(cost);
	//Clean up: destroy cublas library handle and reset cout buffer
		cublasDestroy(cublas_handle);//destroy cublas library handle
		cout.rdbuf(coutbuf);
    //Print out reconstruction status
        if( recon_status == 0 )//now run GIRAF routine
            cout << "GIRAF Recon Status: Success" << endl;
        else
            cout << "GIRAF Recon Status: Failed - please check log file for troubleshooting" << endl;
}
