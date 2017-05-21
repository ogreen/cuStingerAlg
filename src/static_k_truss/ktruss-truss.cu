

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>

#include "update.hpp"
#include "cuStinger.hpp"

#include "operators.cuh"

#include "static_k_truss/k_truss.cuh"	

using namespace cuStingerAlgs;

void KTrussOneIteration(cuStinger& custing,
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter, const int thread_blocks, const int blockdim, kTrussData* devData);


namespace cuStingerAlgs {


void kTruss::setInitParameters(length_t nv, length_t ne,length_t maxK, int tsp, int nbl, int shifter,int blocks, int  sps){
	hostKTrussData.nv 	= nv;
	hostKTrussData.ne 	= ne;
	hostKTrussData.maxK = maxK;
	hostKTrussData.currK = 3;

	hostKTrussData.tsp 		= tsp;
	hostKTrussData.nbl 		= nbl;
	hostKTrussData.shifter 	= shifter;
	hostKTrussData.blocks 	= blocks;
	hostKTrussData.sps 		=sps;


	if (hostKTrussData.currK>hostKTrussData.maxK){
		cout << "**** The smallest supported TRUSS is k=3 ****" << endl;
	}
}


void kTruss::Init(cuStinger& custing){

	hostKTrussData.isActive 		 =  (int32_t*) allocDeviceArray(hostKTrussData.nv, sizeof(int32_t));
	hostKTrussData.offsetArray	     =  (int32_t*) allocDeviceArray(hostKTrussData.nv+1, sizeof(int32_t));
	hostKTrussData.trianglePerVertex =  (triangle_t*) allocDeviceArray(hostKTrussData.nv, sizeof(triangle_t));
	hostKTrussData.trianglePerEdge	 =  (triangle_t*) allocDeviceArray(hostKTrussData.ne, sizeof(triangle_t));
	hostKTrussData.src				 =  (vertexId_t*) allocDeviceArray(hostKTrussData.ne, sizeof(vertexId_t));
	hostKTrussData.dst	 			 =  (vertexId_t*) allocDeviceArray(hostKTrussData.ne, sizeof(vertexId_t));

	deviceKTrussData = (kTrussData*)allocDeviceArray(1, sizeof(kTrussData));

	cusLB = new cusLoadBalance(custing);

	SyncDeviceWithHost();
	Reset();
}

void kTruss::copyOffsetArrayHost(length_t* hostOffsetArray){
	copyArrayHostToDevice(hostOffsetArray, hostKTrussData.offsetArray, hostKTrussData.nv+1, sizeof(length_t));
}

void kTruss::copyOffsetArrayDevice(length_t* deviceOffsetArray){
	copyArrayDeviceToDevice(deviceOffsetArray, hostKTrussData.offsetArray, hostKTrussData.nv+1, sizeof(length_t));
}

void kTruss::Reset(){
	hostKTrussData.currK 	= 3;
	hostKTrussData.counter 	= 0;
	resetEdgeArray();
	resetVertexArray();

	SyncDeviceWithHost();
}

void kTruss::resetVertexArray(){
	cudaMemset((void*)hostKTrussData.trianglePerVertex,0,hostKTrussData.nv*sizeof(int));
}


void kTruss::resetEdgeArray(){
	cudaMemset((void*)hostKTrussData.trianglePerEdge,0,hostKTrussData.ne*sizeof(int));
}

void kTruss::Release(){
	delete cusLB;

	freeDeviceArray(hostKTrussData.isActive);
	freeDeviceArray(hostKTrussData.offsetArray);
	freeDeviceArray(hostKTrussData.trianglePerEdge);
	freeDeviceArray(hostKTrussData.trianglePerVertex);

	freeDeviceArray(deviceKTrussData);

}

void kTruss::Run(cuStinger& custing){

	allVinG_TraverseVertices<kTrussOperators::init>(custing,deviceKTrussData);

	Reset();
	hostKTrussData.activeVertices=custing.nv;
	SyncDeviceWithHost();
	int sumDeletedEdges=0;

	//while(hostKTrussData.currK  < hostKTrussData.maxK && hostKTrussData.activeVertices>0){
	while(hostKTrussData.activeVertices>0){

		KTrussOneIteration(custing, hostKTrussData.trianglePerVertex, hostKTrussData.tsp,
				hostKTrussData.nbl,hostKTrussData.shifter,hostKTrussData.blocks, hostKTrussData.sps,
				deviceKTrussData);

		// cout << "Current number of deleted edges is " << hostKTrussData.counter << endl;

		allVinG_TraverseVertices<kTrussOperators::findUnderK>(custing,deviceKTrussData);
		SyncHostWithDevice();
		cout << "Current number of deleted edges is " << hostKTrussData.counter << endl;
		sumDeletedEdges+=hostKTrussData.counter;
		BatchUpdateData *bud;
		BatchUpdate* bu;
		if(hostKTrussData.counter!=0){
			bud = new BatchUpdateData(hostKTrussData.counter,true,hostKTrussData.nv);
			copyArrayDeviceToHost(hostKTrussData.src,bud->getSrc(),hostKTrussData.counter,sizeof(int));
			copyArrayDeviceToHost(hostKTrussData.dst,bud->getDst(),hostKTrussData.counter,sizeof(int));

			bu = new BatchUpdate(*bud);

			bu->sortDeviceBUD(hostKTrussData.sps);
			// cout << "Hello" << endl;
			// for(int32_t e=0; e<hostKTrussData.counter; e++){
			// 	if(bud->getSrc()[e]> 18772|| bud->getDst()[e] > 18772 )
			// 	printf("Batch update: (#%d) (%d %d)\n", e,bud->getSrc()[e],bud->getDst()[e]);
			// }
			// length_t allocs;
			// custing.edgeInsertions(*bu,allocs);
			custing.edgeDeletionsSorted(*bu);
			delete bu;
			delete bud;

		}
		else
			break;

		hostKTrussData.activeVertices=0;
	
		SyncDeviceWithHost();

		allVinG_TraverseVertices<kTrussOperators::countActive>(custing,deviceKTrussData);
		SyncHostWithDevice();

		resetEdgeArray();
		resetVertexArray();
		
        cout << "Number of active vertices is : " << hostKTrussData.activeVertices   << endl;
		hostKTrussData.currK++;
		hostKTrussData.counter=0;

		SyncDeviceWithHost();

		if(hostKTrussData.currK >= (hostKTrussData.maxK-2))
		  break;
	}
	cout << "The number of initial edges  : " << hostKTrussData.ne << endl;
	cout << "The number of delete edges   : " << sumDeletedEdges <<  endl;
	cout << "The number of leftover edges : " << hostKTrussData.ne - sumDeletedEdges << endl;
}

// length_t kTruss::getIterationCount(){
// 	SyncHostWithDevice();
// 	return hostKTrussData.iteration;
// }


}// cuStingerAlgs namespace
