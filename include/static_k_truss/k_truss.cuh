#pragma once

#include "algs.cuh"
#include "operators.cuh"

typedef int32_t triangle_t;

namespace cuStingerAlgs {

class kTrussData{
public:
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

	int fullTriangleIterations;

	// int numDeletedEdges;
	length_t nv;
	length_t ne; // undirected-edges
	length_t ne_remaining; // undirected-edges

};


// Label propogation is based on the values from the previous iteration.
class kTruss:public StaticAlgorithm{
public:
	void setInitParameters(length_t nv, length_t ne, int tsp, int nbl, int shifter,int blocks, int  sps);
	virtual void Init(cuStinger& custing);

	virtual void Reset();
	virtual void Run(cuStinger& custing);
	bool findTrussOfK(cuStinger& custing,bool& stop);
	void RunForK(cuStinger& custing,int maxK);

	void RunDynamic(cuStinger& custing);
	bool findTrussOfKDynamic(cuStinger& custing,bool& stop);
	void RunForKDynamic(cuStinger& custing,int maxK);



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

	length_t getMaxK(){return hostKTrussData.maxK;}

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
		int pos = kt->offsetArray[src]+adj;

		if (kt->trianglePerEdge[pos] < (kt->maxK-2)){
			int spot = atomicAdd(&(kt->counter), 1);
			kt->src[spot]=src;
			kt->dst[spot]=dst;
		}
	}
}

static __device__ void findUnderKDynamic(cuStinger* custing,vertexId_t src, void* metadata){
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
		if (custing->dVD->adj[src]->ew[adj] < (kt->maxK-2)){
			int spot = atomicAdd(&(kt->counter), 1);
			kt->src[spot]=src;
			kt->dst[spot]=dst;
		}
	}
}

static __device__ void countActive(cuStinger* custing,vertexId_t src, void* metadata){
	kTrussData* kt = (kTrussData*)metadata;
	length_t srcLen=custing->dVD->used[src];
	if(srcLen==0 && !kt->isActive[src]){
		kt->isActive[src]=0;
	}
	else{
		atomicAdd(&(kt->activeVertices), 1);
	}
}

static __device__ void resetWeights(cuStinger* custing,vertexId_t src, void* metadata){
	kTrussData* kt = (kTrussData*)metadata;

	length_t srcLen=custing->dVD->used[src];
	int pos=kt->offsetArray[src];

	for(vertexId_t adj=0; adj<srcLen; adj+=1){
		custing->dVD->adj[src]->ew[adj]=kt->trianglePerEdge[pos+adj];
	}

}

};


} // cuStingerAlgs namespace
	
