

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>

#include "timer.h"

#include "update.hpp"
#include "cuStinger.hpp"

#include "operators.cuh"

#include "static_k_truss/k_truss.cuh"	

using namespace cuStingerAlgs;

void KTrussOneIteration(cuStinger& custing,
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter, const int thread_blocks, const int blockdim, kTrussData* devData);

void callDeviceDifferenceTriangles(cuStinger& custing, BatchUpdate& bu, 
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_intersection,
    const int num_intersec_perblock, const int shifter, const int thread_blocks,
    const int blockdim, bool deletion);

namespace cuStingerAlgs {


void kTruss::setInitParameters(length_t nv, length_t ne,int tsp, int nbl, int shifter,int blocks, int  sps){
	hostKTrussData.nv 	= nv;
	hostKTrussData.ne 	= ne;

	hostKTrussData.tsp 		= tsp;
	hostKTrussData.nbl 		= nbl;
	hostKTrussData.shifter 	= shifter;
	hostKTrussData.blocks 	= blocks;
	hostKTrussData.sps 		=sps;
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
	hostKTrussData.counter 	= 0;
	hostKTrussData.ne_remaining	= hostKTrussData.ne;
	hostKTrussData.fullTriangleIterations 	= 0;


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

	hostKTrussData.maxK = 3;SyncDeviceWithHost();

	while(1){

	// if(hostKTrussData.maxK >=5)
	// 	break;

		bool needStop=false;
		bool more = findTrussOfK(custing,needStop);
		if(more==false && needStop){
			hostKTrussData.maxK--; SyncDeviceWithHost();
			break;
		}
		hostKTrussData.maxK++; SyncDeviceWithHost();
	}
	// cout << "Found the maximal KTruss at : " << hostKTrussData.maxK << endl;
	cout << "The number of full triangle counting iterations is  : " << hostKTrussData.fullTriangleIterations << endl;


}

void kTruss::RunForK(cuStinger& custing,int maxK){

	hostKTrussData.maxK = maxK;SyncDeviceWithHost();

	bool exitOnFirstIteration;
	findTrussOfK(custing,exitOnFirstIteration);
}


bool kTruss::findTrussOfK(cuStinger& custing, bool& stop){

	allVinG_TraverseVertices<kTrussOperators::init>(custing,deviceKTrussData);

	// Reset();
	resetEdgeArray();
	resetVertexArray();

	hostKTrussData.counter 	= 0;
	hostKTrussData.activeVertices=custing.nv;
	SyncDeviceWithHost();
	int sumDeletedEdges=0;
	stop=true;

	while(hostKTrussData.activeVertices>0){

		hostKTrussData.fullTriangleIterations++;
		SyncDeviceWithHost();


		KTrussOneIteration(custing, hostKTrussData.trianglePerVertex, hostKTrussData.tsp,
				hostKTrussData.nbl,hostKTrussData.shifter,hostKTrussData.blocks, hostKTrussData.sps,
				deviceKTrussData);

		allVinG_TraverseVertices<kTrussOperators::findUnderK>(custing,deviceKTrussData);
		SyncHostWithDevice();
		// cout << "Current number of deleted edges is " << hostKTrussData.counter << endl;
		sumDeletedEdges+=hostKTrussData.counter;
		if(hostKTrussData.counter==hostKTrussData.ne_remaining){
			stop = true;
			return false;
		}
		if(hostKTrussData.counter!=0){
			BatchUpdateData *bud;
			BatchUpdate* bu;
			bud = new BatchUpdateData(hostKTrussData.counter,true,hostKTrussData.nv);
			copyArrayDeviceToHost(hostKTrussData.src,bud->getSrc(),hostKTrussData.counter,sizeof(int));
			copyArrayDeviceToHost(hostKTrussData.dst,bud->getDst(),hostKTrussData.counter,sizeof(int));
			bu = new BatchUpdate(*bud);

			bu->sortDeviceBUD(hostKTrussData.sps);
			custing.edgeDeletionsSorted(*bu);
			delete bu;
			delete bud;
		}
		else{
			// cout << "The maxK is                  : " << hostKTrussData.maxK << endl;
			// cout << "This is the first iteration  : " << stop << endl;
			// cout << "The number of delete edges   : " << sumDeletedEdges <<  endl;
			// cout << "The number of leftover edges : " << hostKTrussData.ne_remaining<< endl;

			return false;
		}
		hostKTrussData.ne_remaining-=hostKTrussData.counter;

		hostKTrussData.activeVertices=0;
	
		SyncDeviceWithHost();

		allVinG_TraverseVertices<kTrussOperators::countActive>(custing,deviceKTrussData);
		SyncHostWithDevice();

		resetEdgeArray();
		resetVertexArray();
		
		hostKTrussData.counter=0;

		SyncDeviceWithHost();
		stop=false;
	}

	return true;
}

void kTruss::RunDynamic(cuStinger& custing){

	hostKTrussData.maxK = 3;SyncDeviceWithHost();
	allVinG_TraverseVertices<kTrussOperators::init>(custing,deviceKTrussData);

	resetEdgeArray();
	resetVertexArray();
	SyncDeviceWithHost();

	KTrussOneIteration(custing, hostKTrussData.trianglePerVertex, hostKTrussData.tsp,
				hostKTrussData.nbl,hostKTrussData.shifter,hostKTrussData.blocks, hostKTrussData.sps,
				deviceKTrussData);
	SyncHostWithDevice();

	allVinG_TraverseVertices<kTrussOperators::resetWeights>(custing,deviceKTrussData);

	while(1){

		// if(hostKTrussData.maxK >=5)
		// break;
		// cout << "New iteration" << endl;
		bool needStop=false;
		bool more = findTrussOfKDynamic(custing,needStop);
		if(more==false && needStop){
			hostKTrussData.maxK--; SyncDeviceWithHost();
			break;
		}
		hostKTrussData.maxK++; SyncDeviceWithHost();
	}
	// cout << "Found the maximal KTruss at : " << hostKTrussData.maxK << endl;
}

bool kTruss::findTrussOfKDynamic(cuStinger& custing,bool& stop){

	hostKTrussData.counter 	= 0;
	SyncDeviceWithHost();
	allVinG_TraverseVertices<kTrussOperators::countActive>(custing,deviceKTrussData);
	SyncHostWithDevice();
	stop=true;

	while(hostKTrussData.activeVertices>0){

		allVinG_TraverseVertices<kTrussOperators::findUnderKDynamic>(custing,deviceKTrussData);
		SyncHostWithDevice();
		// cout << "Current number of deleted edges is " << hostKTrussData.counter << endl;

		if(hostKTrussData.counter==hostKTrussData.ne_remaining){
			stop = true;
			return false;
		}
		if(hostKTrussData.counter!=0){
			BatchUpdateData *bud;
			BatchUpdate* bu;
			bud = new BatchUpdateData(hostKTrussData.counter,true,hostKTrussData.nv);

			copyArrayDeviceToHost(hostKTrussData.src,bud->getSrc(),hostKTrussData.counter,sizeof(int));
			copyArrayDeviceToHost(hostKTrussData.dst,bud->getDst(),hostKTrussData.counter,sizeof(int));
			bu = new BatchUpdate(*bud);
			bu->sortDeviceBUD(hostKTrussData.sps);

			custing.edgeDeletionsSorted(*bu);

			callDeviceDifferenceTriangles(custing, *bu, hostKTrussData.trianglePerVertex, 
			hostKTrussData.tsp, hostKTrussData.nbl,hostKTrussData.shifter,
			hostKTrussData.blocks, hostKTrussData.sps,true);
		
			delete bu;
			delete bud;
		}
		else{
			return false;
		}
		hostKTrussData.ne_remaining-=hostKTrussData.counter;

		hostKTrussData.activeVertices=0;
		hostKTrussData.counter=0;
	
		SyncDeviceWithHost();

		allVinG_TraverseVertices<kTrussOperators::countActive>(custing,deviceKTrussData);
		SyncHostWithDevice();
	
		stop=false;
	}

	return true;
		
}

void kTruss::RunForKDynamic(cuStinger& custing,int maxK){
	
}



}// cuStingerAlgs namespace
