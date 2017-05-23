#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

// #include "static_triangle_counting/cct.hpp"

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
int arrayBlockSize[]={192};
int arrayThreadPerIntersection[]={8};
int arrayThreadShift[]={3};


void initHostTriangleArray(triangle_t* h_triangles, vertexId_t nv){	
	for(vertexId_t sd=0; sd<(nv);sd++){
		h_triangles[sd]=0;
	}
}

int64_t sumTriangleArray(triangle_t* h_triangles, vertexId_t nv){	
	int64_t sum=0;
	for(vertexId_t sd=0; sd<(nv);sd++){
	  sum+=h_triangles[sd];
	}
	return sum;
}

int comparecuStingerAndCSR(cuStinger& custing, vertexId_t nv,length_t ne, int maxK, length_t*  off,vertexId_t*  ind)
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
	
	// if(run&1){
	// 	cudaEvent_t startCPU, stopCPU;
	// 	float timeCPU;
	// 	cudaEventCreate(&startCPU); cudaEventCreate(&stopCPU);
	// 	cudaEventRecord(startCPU, 0);
	// 	hostCountTriangles (nv, off,ind, triNE, &allTrianglesCPU);
	// 	cudaEventRecord(stopCPU, 0);cudaEventSynchronize(stopCPU);
		
	// 	cudaThreadSynchronize();cudaEventElapsedTime(&timeCPU, startCPU, stopCPU);
	// 	// STAND_PRINTF("CPU", timeCPU,allTrianglesCPU)
	// }	

	// if(run&2){
		cudaSetDevice(device);
		CUDA(cudaMalloc(&d_off, sizeof(length_t)*(nv+1)));
		CUDA(cudaMalloc(&d_ind, sizeof(vertexId_t)*ne));
		CUDA(cudaMalloc(&d_triangles, sizeof(triangle_t)*(nv+1)));

		CUDA(cudaMemcpy(d_off, off, sizeof(length_t)*(nv+1), cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(d_ind, ind, sizeof(vertexId_t)*ne, cudaMemcpyHostToDevice));

		triangle_t* h_triangles = (triangle_t *) malloc ( sizeof(triangle_t)*(nv+1)  );		

		float minTime=10e9,time,minTimecuStinger=10e9;

		int64_t sumDevice=0;
		initHostTriangleArray(h_triangles,nv);

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
				cuInit.useVWeight = false;cuInit.isSemantic = false; cuInit.useEWeight = false;
				cuInit.csrNV 			= nv;	cuInit.csrNE	   		= ne;
				cuInit.csrOff 			= off;	cuInit.csrAdj 			= ind;
				cuInit.csrVW 			= NULL;	cuInit.csrEW			= NULL;

				custing.initializeCuStinger(cuInit);

					cudaEvent_t ce_start,ce_stop;
					float totalTime;

					kTruss kt;
					kt.setInitParameters(nv,ne, tsp,nbl,shifter,blocks, sps);
					kt.Init(custing);
					kt.copyOffsetArrayDevice(d_off);
					kt.Reset();
					start_clock(ce_start, ce_stop);
					
					if(maxK==-1)
						kt.Run(custing);
					else
						kt.RunForK(custing,maxK);


						// KTrussOneIteration(custing, d_triangles, tsp,nbl,shifter,blocks, sps);

					totalTime = end_clock(ce_start, ce_stop);
					cout << "Total time for k-Truss = " << kt.getMaxK() << " : " << totalTime << endl; 
					kt.Release();

					if(totalTime<minTimecuStinger) minTimecuStinger=totalTime; 

					custing.freecuStinger();


					// CUDA(cudaMemcpy(d_triangles, h_triangles, sizeof(triangle_t)*(nv+1), cudaMemcpyHostToDevice));
					// start_clock(ce_start, ce_stop);
					// 	KTrussOneIteration(custing, d_triangles, tsp,nbl,shifter,blocks, sps);
					// time = end_clock(ce_start, ce_stop);
					// CUDA(cudaMemcpy(h_triangles, d_triangles, sizeof(triangle_t)*(nv+1), cudaMemcpyDeviceToHost));

					// if(time<minTimecuStinger) minTimecuStinger=time; 
					// sumDevice=sumTriangleArray(h_triangles,nv);initHostTriangleArray(h_triangles,nv);

					// printf("### %d %d %d %d %d \t\t %ld \t %f\n", blocks,sps, tsp, nbl, shifter,sumDevice, time);
			    }
			}	
		}
		// STAND_PRINTF("GPU - csr     ", minTime,sumDevice)
		// STAND_PRINTF("GPU - custing ", minTimecuStinger,sumDevice)
		// cout << "Vertices " << nv << endl;
		// cout << "Edges " << ne << endl;

		cout << nv << ", " << ne << ", "<< minTime << ", " << minTimecuStinger<< endl;

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
	int maxK=-1;
	if (argc==3)
		maxK = atoi(argv[2]);

	cout << "Vertices: " << nv << "    Edges: " << ne  << "      " << off[nv] << endl;

	cudaEvent_t ce_start,ce_stop;
	cuStinger custing(defaultInitAllocater,defaultUpdateAllocater);

	cuStingerInitConfig cuInit;
	cuInit.initState =eInitStateCSR;
	cuInit.maxNV = nv+1;
	cuInit.useVWeight = false;
	cuInit.isSemantic = false;  // Use edge types and vertex types
	cuInit.useEWeight = false;
	// CSR data
	cuInit.csrNV 			= nv;
	cuInit.csrNE	   		= ne;
	cuInit.csrOff 			= off;
	cuInit.csrAdj 			= adj;
	cuInit.csrVW 			= NULL;
	cuInit.csrEW			= NULL;

	custing.initializeCuStinger(cuInit);

	comparecuStingerAndCSR(custing,nv,ne,maxK,off,adj);

	custing.freecuStinger();
	cout << "Vertices: " << nv << "    Edges: " << ne  << endl;


	free(off);
	free(adj);
    return 0;	
}
