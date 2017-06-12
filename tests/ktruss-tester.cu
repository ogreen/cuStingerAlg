#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "static_k_truss/k_truss.cuh"	
using namespace cuStingerAlgs;

#define CUDA(call, ...) do {                        \
        cudaError_t _e = (call);                    \
        if (_e == cudaSuccess) break;               \
        fprintf(stdout,                             \
                "CUDA runtime error: %s (%d)\n",    \
                cudaGetErrorString(_e), _e);        \
        return -1;                                  \
    } while (0)


#define STAND_PRINTF(sys, time, triangles) printf("%s : \t%ld \t%f\n", sys,triangles, time);


// int arrayBlocks[]={16000};
// int arrayBlockSize[]={32,64,96,128,192,256};
// int arrayThreadPerIntersection[]={1,2,4,8,16,32};
// int arrayThreadShift[]={0,1,2,3,4,5};
// int arrayBlocks[]={64000};
// int arrayBlockSize[]={256};
// int arrayThreadPerIntersection[]={32};
// int arrayThreadShift[]={5};

int arrayBlocks[]={16000};
int arrayBlockSize[]={32};
int arrayThreadPerIntersection[]={1};
int arrayThreadShift[]={0};


// int arrayBlocks[]={64000};
// int arrayBlockSize[]={64};
// int arrayThreadPerIntersection[]={8};
// int arrayThreadShift[]={3};

int runKtruss(vertexId_t nv,length_t ne, int alg, int maxk, length_t*  off,vertexId_t*  ind,string graphName)
{
	int device = 0;
	int run    = 2;
//  int scriptMode =atoi(argv[PAR_SCRIPT]);
//	int sps =atoi(argv[PAR_SP]);	
//	int tsp =atoi(argv[PAR_T_SP]);	
//	int nbl =atoi(argv[PAR_NUM_BL]);
//	int shifter =atoi(argv[PAR_SHIFT]);
		
	struct cudaDeviceProp prop;
	cudaGetDeviceProperties	(&prop,device);	
    length_t *d_off = NULL;	
    vertexId_t* d_ind = NULL;
	triangle_t *d_triangles = NULL;  

   	int * triNE = (int *) malloc ((ne ) * sizeof (int));	
	int64_t allTrianglesCPU=0;
	
		cudaSetDevice(device);
		CUDA(cudaMalloc(&d_off, sizeof(length_t)*(nv+1)));
		CUDA(cudaMalloc(&d_ind, sizeof(vertexId_t)*ne));
		CUDA(cudaMalloc(&d_triangles, sizeof(triangle_t)*(nv+1)));

		CUDA(cudaMemcpy(d_off, off, sizeof(length_t)*(nv+1), cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(d_ind, ind, sizeof(vertexId_t)*ne, cudaMemcpyHostToDevice));

		triangle_t* h_triangles = (triangle_t *) malloc ( sizeof(triangle_t)*(nv+1)  );		

		float minTime=10e9,time,minTimecuStinger=10e9;

		int64_t sumDevice=0;

		int blocksToTest=sizeof(arrayBlocks)/sizeof(int);
		int blockSizeToTest=sizeof(arrayBlockSize)/sizeof(int);
		int tSPToTest=sizeof(arrayThreadPerIntersection)/sizeof(int);
		for(int b=0;b<blocksToTest; b++){
		    int blocks=arrayBlocks[b];
			for(int bs=0; bs<blockSizeToTest; bs++){
			    int sps=arrayBlockSize[bs];
			    for(int t=0; t<tSPToTest;t++){
		            int tsp=arrayThreadPerIntersection[t];
					int shifter=arrayThreadShift[t];
					int nbl=sps/tsp;

					cuStinger custing(defaultInitAllocater,defaultUpdateAllocater);

					cuStingerInitConfig cuInit;
					cuInit.initState =eInitStateCSR;
					cuInit.maxNV = nv+1;
					cuInit.useVWeight = false;cuInit.isSemantic = false; cuInit.useEWeight = true;
					cuInit.csrNV 			= nv;	cuInit.csrNE	   		= ne;
					cuInit.csrOff 			= off;	cuInit.csrAdj 			= ind;
					cuInit.csrVW 			= NULL;	cuInit.csrEW			= NULL;

					cudaEvent_t ce_start,ce_stop;
					float totalTime;

					if(alg&1){
						custing.initializeCuStinger(cuInit);
						kTruss kt;
						kt.setInitParameters(nv,ne, tsp,nbl,shifter,blocks, sps);
						kt.Init(custing);
						kt.copyOffsetArrayDevice(d_off);
						kt.Reset();
						start_clock(ce_start, ce_stop);
						
						kt.Run(custing);
						totalTime = end_clock(ce_start, ce_stop);
						cout << "graph="<< graphName<< endl; 
						cout << "k=" << kt.getMaxK() << ":" << totalTime << endl; 
						kt.Release();
						custing.freecuStinger();

					}
					if(alg&2){
						cuStinger custing2(defaultInitAllocater,defaultUpdateAllocater);
						custing2.initializeCuStinger(cuInit);

						kTruss kt2;
						kt2.setInitParameters(nv,ne, tsp,nbl,shifter,blocks, sps);
						kt2.Init(custing2);
						kt2.copyOffsetArrayDevice(d_off);
						kt2.Reset();
						start_clock(ce_start, ce_stop);
						
						kt2.RunDynamic(custing2);

						totalTime = end_clock(ce_start, ce_stop);
						cout << "graph="<< graphName<< endl; 
						cout << "k=" << kt2.getMaxK() << ":" << totalTime << endl; 
						kt2.Release();

						if(totalTime<minTimecuStinger) minTimecuStinger=totalTime; 

						custing2.freecuStinger();
					}
					if(alg&4){
						custing.initializeCuStinger(cuInit);
						kTruss kt;
						kt.setInitParameters(nv,ne, tsp,nbl,shifter,blocks, sps);
						kt.Init(custing);
						kt.copyOffsetArrayDevice(d_off);
						kt.Reset();
						start_clock(ce_start, ce_stop);
						
						kt.RunForK(custing,maxk);
						totalTime = end_clock(ce_start, ce_stop);
						cout << "graph="<< graphName<< endl; 
						cout << "k=" << kt.getMaxK() << ":" << totalTime << endl; 
						kt.Release();
						custing.freecuStinger();

					}


			    }
			}	
		}
		free(h_triangles);

		CUDA(cudaFree(d_off));
		CUDA(cudaFree(d_ind));
		CUDA(cudaFree(d_triangles));
	// }
	free(triNE);
    return 0;
}



int main(const int argc, char *argv[]){
	int device=0;
    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
 
    length_t nv, ne,*off;
    vertexId_t *adj;

	bool isDimacs,isSNAP,isRmat=false,isMarket;
	string filename(argv[1]);
	isDimacs = filename.find(".graph")==std::string::npos?false:true;
	isSNAP   = filename.find(".txt")==std::string::npos?false:true;
	isRmat 	 = filename.find("kron")==std::string::npos?false:true;
	isMarket = filename.find(".mtx")==std::string::npos?false:true;

	string graphName=filename;
	int period = filename.find(".");

	int i, lasti=0;
	for(i = graphName.find("/", 0); i != std::string::npos; i = graphName.find("/", i)){
		i++;
		lasti=i;
	}
	graphName = graphName.substr(lasti,period-lasti);

	// cout << lasti << " " << period << endl;
	// cout << graphName << endl;

	if(isDimacs){
	    readGraphDIMACS(argv[1],&off,&adj,&nv,&ne,isRmat);
	}
	else if(isSNAP){
	    readGraphSNAP(argv[1],&off,&adj,&nv,&ne,true);
	}
	else if(isMarket){
		readGraphMatrixMarket(argv[1],&off,&adj,&nv,&ne,(isRmat)?false:true);
	}
	else{ 
		cout << "Unknown graph type" << endl;
	}
	int alg=3,maxk=3;
	if (argc>=3)
		alg = atoi(argv[2]);
	if (argc>=4)
		maxk = atoi(argv[3]);

	cout << "Vertices: " << nv << "    Edges: " << ne  << "      " << off[nv] << endl;

	cudaEvent_t ce_start,ce_stop;

	runKtruss(nv,ne,alg,maxk,off,adj,graphName);



	free(off);
	free(adj);
    return 0;	
}
