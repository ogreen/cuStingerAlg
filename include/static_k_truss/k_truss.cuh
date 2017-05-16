#pragma once

#include "algs.cuh"
#include "operators.cuh"

typedef int32_t triangle_t;

namespace cuStingerAlgs {

class kTrussData{
public:
	int currK;
	int maxK;

	bool* isActive;
	length_t* offsetArray;
	length_t* trianglePerEdge;

	length_t numDeletedEdges;
	length_t nv;
	length_t ne; // undirected-edges

};


// Label propogation is based on the values from the previous iteration.
class kTruss:public StaticAlgorithm{
public:
	void setInitParameters(length_t nv, length_t ne,length_t maxK);


	virtual void Init(cuStinger& custing);
	virtual void Reset();
	virtual void Run(cuStinger& custing);
	virtual void Release();

	void resetEdgeArray();


	virtual void SyncHostWithDevice(){
		copyArrayDeviceToHost(deviceKTrussData,&hostKTrussData,1, sizeof(kTrussData));
	}
	virtual void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostKTrussData,deviceKTrussData,1, sizeof(kTrussData));
	}

	length_t getIterationCount();

	// const kTrussData* getHostKatzData(){return hostKTrussData;}
	// const kTrussData* getDeviceKatzData(){return deviceKTrussData;}

	// virtual void copyKCToHost(double* hostArray){
	// 	copyArrayDeviceToHost(hostKTrussData->KC,hostArray, hostKTrussData->nv, sizeof(double));
	// }

	// virtual void copynPathsToHost(ulong_t* hostArray){
	// 	copyArrayDeviceToHost(hostKTrussData->nPathsData,hostArray, (hostKTrussData->nv)*hostKTrussData->maxIteration, sizeof(ulong_t));
	// }


protected:
	kTrussData hostKTrussData, *deviceKTrussData;

private:
	cusLoadBalance* cusLB;
};


class kTrussOperators{
public:

// // Used at the very beginning
// static __device__ void init(cuStinger* custing,vertexId_t src, void* metadata){
// 	katzData* kd = (katzData*)metadata;
// 	kd->nPathsPrev[src]=1;
// 	kd->nPathsCurr[src]=0;
// 	kd->KC[src]=0.0;
// 	kd->isActive[src]=true;
// 	kd->indexArray[src]=src;
// }

// // Used every iteration
// static __device__ void initNumPathsPerIteration(cuStinger* custing,vertexId_t src, void* metadata){
// 	katzData* kd = (katzData*)metadata;
// 	kd->nPathsCurr[src]=0;
// }

// static __device__ void updatePathCount(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
// 	katzData* kd = (katzData*)metadata;
// 	atomicAdd(kd->nPathsCurr+src, kd->nPathsPrev[dst]);
// }

// static __device__ void updateKatzAndBounds(cuStinger* custing,vertexId_t src, void* metadata){
// 	katzData* kd = (katzData*)metadata;
// 	kd->KC[src]=kd->KC[src] + kd->alphaI * (double)kd->nPathsCurr[src];
// 	kd->lowerBound[src]=kd->KC[src] + kd->lowerBoundConst * (double)kd->nPathsCurr[src];
// 	kd->upperBound[src]=kd->KC[src] + kd->upperBoundConst * (double)kd->nPathsCurr[src];   

// 	if(kd->isActive[src]){
// 		length_t pos = atomicAdd(&(kd -> nActive),1);
// 		kd->vertexArray[pos] = src;
// 		kd->lowerBoundSort[pos]=kd->lowerBound[src];
// 	}
// }


// static __device__ void countActive(cuStinger* custing,vertexId_t src, void* metadata){
// 	katzData* kd = (katzData*)metadata;
// 	if (kd->upperBound[src] > kd->lowerBound[kd->vertexArray[kd->K-1]]) {
// 		atomicAdd(&(kd -> nActive),1);
// 	}
// 	else{
// 		kd->isActive[src] = false;
// 	}
// }

};



} // cuStingerAlgs namespace
