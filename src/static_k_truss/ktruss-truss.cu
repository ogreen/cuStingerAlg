


#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>

#include "update.hpp"
#include "cuStinger.hpp"

#include "operators.cuh"

#include "static_k_truss/k_truss.cuh"	

namespace cuStingerAlgs {

/// TODO - changed hostKatzdata to pointer so that I can try to inherit it in the streaming case.
	


void kTruss::setInitParameters(length_t nv, length_t ne,length_t maxK){
	hostKTrussData.nv 	= nv;
	hostKTrussData.ne 	= ne;
	hostKTrussData.maxK = maxK;

}


void kTruss::Init(cuStinger& custing){

	hostKTrussData.isActive 		 = (bool*) allocDeviceArray(hostKTrussData.nv, sizeof(bool));
	hostKTrussData.offsetArray	     =  (length_t*) allocDeviceArray(hostKTrussData.nv, sizeof(length_t));
	hostKTrussData.trianglePerEdge	 =  (triangle_t*) allocDeviceArray(hostKTrussData.ne, sizeof(triangle_t));

	deviceKTrussData = (kTrussData*)allocDeviceArray(1, sizeof(kTrussData));
	cusLB = new cusLoadBalance(custing);

	SyncDeviceWithHost();
	Reset();

	cout << "--------------------------------------" << endl << "ODED - YOU NEED to copy the CSR OFFSET ARRAY" << endl << "--------------------------------------" << endl;
}

void kTruss::Reset(){
	// hostKTrussData.iteration = 1;

	resetEdgeArray();
	// if(isStatic==true){
	// 	hostKTrussData.nPathsPrev = hostKTrussData.nPathsData;
	// 	hostKTrussData.nPathsCurr = hostKTrussData.nPathsData+(hostKTrussData.nv);
	// }
	// else{
	// 	hostKTrussData.nPathsPrev = hPathsPtr[0];
	// 	hostKTrussData.nPathsCurr = hPathsPtr[1];
	// }
	// SyncDeviceWithHost();

}

void kTruss::resetEdgeArray(){
	cudaMemset((void*)hostKTrussData.trianglePerEdge,0,hostKTrussData.ne);
}

void kTruss::Release(){
	delete cusLB;

	freeDeviceArray(hostKTrussData.isActive);
	freeDeviceArray(hostKTrussData.offsetArray);
	freeDeviceArray(hostKTrussData.trianglePerEdge);

	freeDeviceArray(deviceKTrussData);

}

void kTruss::Run(cuStinger& custing){

	// allVinG_TraverseVertices<kTrussOperator::init>(custing,deviceKTrussData);
	// standard_context_t context(false);

	// hostKTrussData.iteration = 1;
	
	// hostKTrussData.nActive = hostKTrussData.nv;
	// while(hostKTrussData.nActive  > hostKTrussData.K && hostKTrussData.iteration < hostKTrussData.maxIteration){

	// 	hostKTrussData.alphaI          = pow(hostKTrussData.alpha,hostKTrussData.iteration);
	// 	hostKTrussData.lowerBoundConst = pow(hostKTrussData.alpha,hostKTrussData.iteration+1)/((1.0-hostKTrussData.alpha));
	// 	hostKTrussData.upperBoundConst = pow(hostKTrussData.alpha,hostKTrussData.iteration+1)/((1.0-hostKTrussData.alpha*(double)hostKTrussData.maxDegree));
	// 	hostKTrussData.nActive = 0; // Each iteration the number of active vertices is set to zero.
	
	// 	SyncDeviceWithHost(); // Passing constants to the device.

	// 	allVinG_TraverseVertices<kTrussOperator::initNumPathsPerIteration>(custing,deviceKTrussData);
	// 	allVinA_TraverseEdges_LB<kTrussOperator::updatePathCount>(custing,deviceKTrussData,*cusLB);
	// 	allVinG_TraverseVertices<kTrussOperator::updateKatzAndBounds>(custing,deviceKTrussData);

	// 	SyncHostWithDevice();
	// 	hostKTrussData.iteration++;

	// 	if(isStatic){
	// 		// Swapping pointers.
	// 		ulong_t* temp = hostKTrussData.nPathsCurr; hostKTrussData.nPathsCurr=hostKTrussData.nPathsPrev; hostKTrussData.nPathsPrev=temp;	
	// 	}else{
	// 		hostKTrussData.nPathsPrev = hPathsPtr[hostKTrussData.iteration - 1];
	// 		hostKTrussData.nPathsCurr = hPathsPtr[hostKTrussData.iteration - 0];
	// 	}

	// 	length_t oldActiveCount = hostKTrussData.nActive;
	// 	hostKTrussData.nActive = 0; // Resetting active vertices for sorting operations.

	// 	SyncDeviceWithHost();

	// 	mergesort(hostKTrussData.lowerBoundSort,hostKTrussData.vertexArray,oldActiveCount, greater_t<double>(),context);

	// 	// allVinG_TraverseVertices<kTrussOperator::countActive>(custing,deviceKTrussData);
	// 	allVinA_TraverseVertices<kTrussOperator::countActive>(custing,deviceKTrussData,hostKTrussData.vertexArray,oldActiveCount);

	// 	//allVinA_TraverseVertices<kTrussOperator::printKID>(custing,deviceKTrussData,hostKTrussData.vertexArray, custing.nv);
	// 	SyncHostWithDevice();
	// 	cout << hostKTrussData.nActive << endl;
	// }
	// // cout << "@@ " << hostKTrussData.iteration << " @@" << endl;
	// SyncHostWithDevice();
}

// length_t kTruss::getIterationCount(){
// 	SyncHostWithDevice();
// 	return hostKTrussData.iteration;
// }


}// cuStingerAlgs namespace
