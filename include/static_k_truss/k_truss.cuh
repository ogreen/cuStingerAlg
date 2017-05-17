#pragma once

#include "algs.cuh"
#include "operators.cuh"

typedef int32_t triangle_t;

namespace cuStingerAlgs {

class kTrussData{
public:
	int currK;
	int maxK;

	int tsp;
	int nbl;
	int shifter;
	int blocks;
	int sps;

	int32_t* isActive;
	int32_t* offsetArray;
	int32_t* trianglePerEdge;
	int32_t* trianglePerVertex;

	vertexId_t* src;
	vertexId_t* dst;
	int  counter;
	int  activeVertices;

	// int numDeletedEdges;
	length_t nv;
	length_t ne; // undirected-edges

};


// Label propogation is based on the values from the previous iteration.
class kTruss:public StaticAlgorithm{
public:
	void setInitParameters(length_t nv, length_t ne,length_t maxK, int tsp, int nbl, int shifter,int blocks, int  sps);
	virtual void Init(cuStinger& custing);

	virtual void Reset();
	virtual void Run(cuStinger& custing);
	virtual void Release();

	void copyOffsetArrayHost(length_t* hostOffsetArray);
	void copyOffsetArrayDevice(length_t* deviceOffsetArray);
	void resetEdgeArray();
	void resetVertexArray();


	virtual void SyncHostWithDevice(){
		copyArrayDeviceToHost(deviceKTrussData,&hostKTrussData,1, sizeof(kTrussData));
	}
	virtual void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostKTrussData,deviceKTrussData,1, sizeof(kTrussData));
	}

	length_t getIterationCount();

	length_t getCurrK(){return hostKTrussData.currK;}

protected:
	kTrussData hostKTrussData, *deviceKTrussData;

private:
	cusLoadBalance* cusLB;
};


class kTrussOperators{
public:

static __device__ void init(cuStinger* custing,vertexId_t src, void* metadata){
	kTrussData* kt = (kTrussData*)metadata;
	kt->isActive[src]=1;
}

// Used at the very beginning
static __device__ void findUnderK(cuStinger* custing,vertexId_t src, void* metadata){
	kTrussData* kt = (kTrussData*)metadata;

	length_t srcLen=custing->dVD->used[src];
	if(kt->isActive[src]==0)
		return;
	if(srcLen==0){
		kt->isActive[src]=0;
		return;
	}
	vertexId_t* adj_src=custing->dVD->adj[src]->dst;
	for(vertexId_t adj=0; adj<srcLen; adj+=1){
		vertexId_t dst = adj_src[adj];
		
		// int* ptr = &kt->counter;
		int pos = kt->offsetArray[src]+adj;

		if (kt->trianglePerEdge[pos] < kt->currK){
			int spot = atomicAdd(&(kt->counter), 1);
			kt->src[spot]=src;
			kt->dst[spot]=dst;
		}
		// else{
		// 	printf("#### %d %d %d\n",src,dst,kt->trianglePerEdge[pos]);			
		// }
	}
}

static __device__ void countActive(cuStinger* custing,vertexId_t src, void* metadata){
	kTrussData* kt = (kTrussData*)metadata;

	length_t srcLen=custing->dVD->used[src];
	if(srcLen==0){
		kt->isActive[src]=0;
	}
	else{
		int* ptr = &kt->activeVertices;
		atomicAdd(&(kt->activeVertices), 1);
	}
}


// // Used every iteration
// static __device__ void initNumPathsPerIteration(cuStinger* custing,vertexId_t src, void* metadata){
// 	katzData* kd = (katzData*)metadata;
// 	kd->nPathsCurr[src]=0;
// }


};



} // cuStingerAlgs namespace
	