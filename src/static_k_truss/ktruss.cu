
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


__device__ void initialize(const vertexId_t diag_id, const length_t u_len, length_t v_len,
    length_t* const __restrict__ u_min, length_t* const __restrict__ u_max,
    length_t* const __restrict__ v_min, length_t* const __restrict__ v_max,
    int* const __restrict__ found)
{
	if (diag_id == 0){
		*u_min=*u_max=*v_min=*v_max=0;
		*found=1;
	}
	else if (diag_id < u_len){
		*u_min=0; *u_max=diag_id;
		*v_max=diag_id;*v_min=0;
	}
	else if (diag_id < v_len){
		*u_min=0; *u_max=u_len;
		*v_max=diag_id;*v_min=diag_id-u_len;
	}
	else{
		*u_min=diag_id-v_len; *u_max=u_len;
		*v_min=diag_id-u_len; *v_max=v_len;
	}
}

__device__ void workPerThread(const length_t uLength, const length_t vLength, 
	const int threadsPerIntersection, const int threadId,
    int * const __restrict__ outWorkPerThread, int * const __restrict__ outDiagonalId){
  int totalWork = uLength + vLength;
  int remainderWork = totalWork%threadsPerIntersection;
  int workPerThread = totalWork/threadsPerIntersection;

  int longDiagonals  = (threadId > remainderWork) ? remainderWork:threadId;
  int shortDiagonals = (threadId > remainderWork) ? (threadId - remainderWork):0;

  *outDiagonalId = ((workPerThread+1)*longDiagonals) + (workPerThread*shortDiagonals);
  *outWorkPerThread = workPerThread + (threadId < remainderWork);
}

__device__ void bSearch(unsigned int found, const vertexId_t diagonalId,
    vertexId_t const * const __restrict__ uNodes, vertexId_t const * const __restrict__ vNodes,
    length_t const * const __restrict__ uLength, 
    length_t * const __restrict__ outUMin, length_t * const __restrict__ outUMax,
    length_t * const __restrict__ outVMin, length_t * const __restrict__ outVMax,    
    length_t * const __restrict__ outUCurr,
    length_t * const __restrict__ outVCurr){
  	length_t length;
	
	while(!found) {
	    *outUCurr = (*outUMin + *outUMax)>>1;
	    *outVCurr = diagonalId - *outUCurr;
	    if(*outVCurr >= *outVMax){
			length = *outUMax - *outUMin;
			if(length == 1){
				found = 1;
				continue;
			}
	    }

	    unsigned int comp1 = uNodes[*outUCurr] > vNodes[*outVCurr-1];
	    unsigned int comp2 = uNodes[*outUCurr-1] > vNodes[*outVCurr];
	    if(comp1 && !comp2){
			found = 1;
	    }
	    else if(comp1){
	      *outVMin = *outVCurr;
	      *outUMax = *outUCurr;
	    }
	    else{
	      *outVMax = *outVCurr;
	      *outUMin = *outUCurr;
	    }
  	}

	if((*outVCurr >= *outVMax) && (length == 1) && (*outVCurr > 0) &&
	(*outUCurr > 0) && (*outUCurr < (*uLength - 1))){
		unsigned int comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
		unsigned int comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
		if(!comp1 && !comp2){(*outUCurr)++; (*outVCurr)--;}
	}
}


__device__ int fixStartPoint(const length_t uLength, const length_t vLength,
    length_t * const __restrict__ uCurr, length_t * const __restrict__ vCurr,
    vertexId_t const * const __restrict__ uNodes, vertexId_t const * const __restrict__ vNodes){
	
	unsigned int uBigger = (*uCurr > 0) && (*vCurr < vLength) && (uNodes[*uCurr-1] == vNodes[*vCurr]);
	unsigned int vBigger = (*vCurr > 0) && (*uCurr < uLength) && (vNodes[*vCurr-1] == uNodes[*uCurr]);
	*uCurr += vBigger;
	*vCurr += uBigger;
	return (uBigger + vBigger);
}

__device__ vertexId_t* binSearch(vertexId_t *a, vertexId_t x, length_t n)
{
	length_t min = 0, max = n, acurr, curr;// = (min+max)/2
	do {
		curr = (min+max)/2;
		acurr = a[curr];
		min = (x > acurr) ? curr : min;
		max = (x < acurr) ? curr : max;
	} while (x != acurr || min != max);
	return a + curr;
}

__device__ int32_t findIndexOfVertex(cuStinger* custing,vertexId_t src,vertexId_t dst__){
	length_t srcLen=custing->dVD->used[src];
	vertexId_t* adj_src=custing->dVD->adj[src]->dst;
	for(vertexId_t adj=0; adj<srcLen; adj+=1){
		vertexId_t dst = adj_src[adj];
		if(dst==dst__)
			return adj;
	}
	// printf("This should never happpen\n");
	return -1;
}

__device__ void indexBinarySearch(vertexId_t* data, length_t arrLen, vertexId_t key, int& pos){
	int low = 0;
	int high = arrLen-1;
	while(high >= low) {
	     int middle = (low + high) / 2;
	     if(data[middle] == key) {
			 pos=middle;
	         return;
	     }
	     if(data[middle] < key) {
	         low = middle + 1;
	     }
	     if(data[middle] > key) {
	         high = middle - 1;
	     }
	}
	return;		
}

__device__ void findIndexOfTwoVerticesBinary(cuStinger* custing,vertexId_t src,vertexId_t v1, vertexId_t v2,
			int &pos_v1, int &pos_v2){
	vertexId_t* adj_src=custing->dVD->adj[src]->dst;
	length_t srcLen=custing->dVD->used[src];
	pos_v1=-1;
	pos_v2=-1;

	indexBinarySearch(adj_src, srcLen, v1,pos_v1);
	indexBinarySearch(adj_src, srcLen, v2,pos_v2);
}

__device__ void findIndexOfTwoVertices(cuStinger* custing,vertexId_t src,vertexId_t v1, vertexId_t v2,
			int &pos_v1, int &pos_v2){
	length_t srcLen=custing->dVD->used[src];
	vertexId_t* adj_src=custing->dVD->adj[src]->dst;

	pos_v1=-1;
	pos_v2=-1;
	for(vertexId_t adj=0; adj<srcLen; adj+=1){
		vertexId_t dst = adj_src[adj];
		if(dst==v1)
			pos_v1=adj;
		if(dst==v2)
			pos_v2=adj;
		if(pos_v1!=-1 && pos_v2!=-1)
			return;
	}
	printf("This should never happpen\n");
	return;
}


template <bool uMasked, bool vMasked, bool subtract, bool upd3rdV>
__device__ void intersectCount(cuStinger* custing,const length_t uLength, const length_t vLength,
    vertexId_t const * const __restrict__ uNodes, vertexId_t const * const __restrict__ vNodes,
    length_t * const __restrict__ uCurr, length_t * const __restrict__ vCurr,
    int * const __restrict__ workIndex, int * const __restrict__ workPerThread,
    int * const __restrict__ triangles, int found, triangle_t * const __restrict__ outPutTriangles, 
    vertexId_t const * const __restrict__ uMask, vertexId_t const * const __restrict__ vMask, triangle_t multiplier, 
	vertexId_t src, vertexId_t dest,vertexId_t u, vertexId_t v)
{
  if((*uCurr < uLength) && (*vCurr < vLength)){
    int comp;
    int vmask;
    int umask;
    while(*workIndex < *workPerThread){
    	vmask = (vMasked) ? vMask[*vCurr] : 0;
        umask = (uMasked) ? uMask[*uCurr] : 0;
		comp = uNodes[*uCurr] - vNodes[*vCurr];

		*triangles += (comp == 0 && !umask && !vmask);

		if (upd3rdV && comp == 0 && !umask && !vmask)
			if (subtract) {
				// atomicSub(outPutTriangles + uNodes[*uCurr], multiplier);

				// Ktruss
				vertexId_t common = uNodes[*uCurr];
				length_t pos_id;

					length_t posu,posv;
					findIndexOfTwoVerticesBinary(custing,common,u,v,posu,posv);
					
					if(posu!=-1)
						atomicSub(custing->dVD->adj[common]->ew+posu,1);
					else
						printf("1");

					if(posv!=-1)
						atomicSub(custing->dVD->adj[common]->ew+posv,1);
					else
						printf("2");

						atomicSub(custing->dVD->adj[u]->ew+*uCurr,1);
						atomicSub(custing->dVD->adj[v]->ew+*vCurr,1);


			}
			// else {
				// 	atomicAdd(outPutTriangles + uNodes[*uCurr], multiplier);
			// }
		*uCurr += (comp <= 0 && !vmask) || umask;
		*vCurr += (comp >= 0 && !umask) || vmask;
		*workIndex += (comp == 0&& !umask && !vmask) + 1;

		if((*vCurr == vLength) || (*uCurr == uLength)){
			break;
		}
    }
    *triangles -= ((comp == 0) && (*workIndex > *workPerThread) && (found));
  }
}



// u_len < v_len
template <bool uMasked, bool vMasked, bool subtract, bool upd3rdV>
__device__ triangle_t count_triangles(cuStinger* custing,vertexId_t u, vertexId_t const * const __restrict__ u_nodes, length_t u_len,
    vertexId_t v, vertexId_t const * const __restrict__ v_nodes, length_t v_len, int threads_per_block,
    volatile vertexId_t* __restrict__ firstFound, int tId, triangle_t * const __restrict__ outPutTriangles,
    vertexId_t const * const __restrict__ uMask, vertexId_t const * const __restrict__ vMask, triangle_t multiplier, 
	vertexId_t src, vertexId_t dest)
{

	// Partitioning the work to the multiple thread of a single GPU processor. The threads should get a near equal number of the elements to Tersect - this number will be off by 1.
	int work_per_thread, diag_id;
	workPerThread(u_len, v_len, threads_per_block, tId, &work_per_thread, &diag_id);
	triangle_t triangles = 0;
	int work_index = 0,found=0;
	length_t u_min,u_max,v_min,v_max,u_curr,v_curr;

	firstFound[tId]=0;

	if(work_per_thread>0){
		// For the binary search, we are figuring out the initial poT of search.
		initialize(diag_id, u_len, v_len,&u_min, &u_max,&v_min, &v_max,&found);
    	u_curr = 0; v_curr = 0;

	    bSearch(found, diag_id, u_nodes, v_nodes, &u_len, &u_min, &u_max, &v_min,
        &v_max, &u_curr, &v_curr);

    	int sum = fixStartPoint(u_len, v_len, &u_curr, &v_curr, u_nodes, v_nodes);
    	work_index += sum;
	    if(tId > 0)
	      firstFound[tId-1] = sum;
	    triangles += sum;
	    intersectCount<uMasked, vMasked, subtract, upd3rdV>(custing,
	    	u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
	        &work_index, &work_per_thread, &triangles, firstFound[tId], outPutTriangles, 
	        uMask, vMask, multiplier,src,dest,u,v);
	}
	return triangles;
}


__device__ void workPerBlock(const length_t numVertices,
    length_t * const __restrict__ outMpStart,
    length_t * const __restrict__ outMpEnd, int blockSize)
{
	length_t verticesPerMp = numVertices/gridDim.x;
	length_t remainderBlocks = numVertices % gridDim.x;
	length_t extraVertexBlocks = (blockIdx.x > remainderBlocks)? remainderBlocks:blockIdx.x;
	length_t regularVertexBlocks = (blockIdx.x > remainderBlocks)? blockIdx.x - remainderBlocks:0;

	length_t mpStart = ((verticesPerMp+1)*extraVertexBlocks) + (verticesPerMp*regularVertexBlocks);
	*outMpStart = mpStart;
	*outMpEnd = mpStart + verticesPerMp + (blockIdx.x < remainderBlocks);
}


__global__ void devicecuStingerKTruss(cuStinger* custing,
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter,kTrussData* devData)
{
	vertexId_t nv = custing->nv;
	// Partitioning the work to the multiple thread of a single GPU processor. The threads should get a near equal number of the elements to intersect - this number will be off by no more than one.
	int tx = threadIdx.x;
 	vertexId_t this_mp_start, this_mp_stop;

	const int blockSize = blockDim.x;
	workPerBlock(nv, &this_mp_start, &this_mp_stop, blockSize);

	__shared__ triangle_t  s_triangles[1024];
	__shared__ vertexId_t firstFound[1024];

	length_t adj_offset=tx>>shifter;
	length_t* firstFoundPos=firstFound + (adj_offset<<shifter);
	for (vertexId_t src = this_mp_start; src < this_mp_stop; src++){
		length_t srcLen=custing->dVD->getUsed()[src];
	    triangle_t tCount = 0;	    
		for(int k=adj_offset; k<srcLen; k+=number_blocks){
			vertexId_t dest = custing->dVD->getAdj()[src]->dst[k];
			int destLen=custing->dVD->getUsed()[dest];

			 if (dest<src) 
			 	continue;

			bool avoidCalc = (src == dest) || (destLen < 2) || (srcLen < 2);
			if(avoidCalc)
				continue;

	        bool sourceSmaller = (srcLen<destLen);
	        vertexId_t small = sourceSmaller? src : dest;
	        vertexId_t large = sourceSmaller? dest : src;
	        length_t small_len = sourceSmaller? srcLen : destLen;
	        length_t large_len = sourceSmaller? destLen : srcLen;

	        const vertexId_t* small_ptr = custing->dVD->getAdj()[small]->dst;
	        const vertexId_t* large_ptr = custing->dVD->getAdj()[large]->dst;
	        // triangle_t triFound = count_triangles<false,false,false,true>
	        triangle_t triFound = count_triangles<false,false,false,false>
						(custing,small, small_ptr, small_len,
						 large,large_ptr, large_len,
						 threads_per_block,firstFoundPos,
						 tx%threads_per_block, outPutTriangles, NULL, NULL,1,src,dest);
	        tCount +=triFound; 
	        int pos=devData->offsetArray[src]+k;
	        atomicAdd(devData->trianglePerEdge+pos,triFound);
            pos=-1;
			indexBinarySearch(custing->dVD->getAdj()[dest]->dst, destLen, src,pos);

	        pos=devData->offsetArray[dest]+pos;
	        atomicAdd(devData->trianglePerEdge+pos,triFound);
		}
	//	s_triangles[tx] = tCount;
	//	blockReduce(&outPutTriangles[src],s_triangles,blockSize);
	}
}

void KTrussOneIteration(cuStinger& custing,
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter, const int thread_blocks, const int blockdim, kTrussData* devData){

	devicecuStingerKTruss<<<thread_blocks, blockdim>>>(custing.devicePtr(), outPutTriangles, threads_per_block,number_blocks,shifter,devData);
}


__global__ void devicecuStingerNewTriangles(cuStinger* custing, BatchUpdateData *bud,
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter, bool deletion,
    length_t const * const __restrict__ redCU)
{
	length_t batchSize = *(bud->getBatchSize());
	// Partitioning the work to the multiple thread of a single GPU processor. The threads should get a near equal number of the elements to intersect - this number will be off by no more than one.
	int tx = threadIdx.x;
 	length_t this_mp_start, this_mp_stop;

	// length_t *d_off = bud->getOffsets();
	vertexId_t * d_ind = bud->getDst();
	vertexId_t * d_seg = bud->getSrc();

	const int blockSize = blockDim.x;
	workPerBlock(batchSize, &this_mp_start, &this_mp_stop, blockSize);

	__shared__ vertexId_t firstFound[1024];

	length_t adj_offset=tx>>shifter;
	length_t* firstFoundPos=firstFound + (adj_offset<<shifter);
	for (length_t edge = this_mp_start+adj_offset; edge < this_mp_stop; edge+=number_blocks){
		if (bud->getIndDuplicate()[edge]==1) {// this means it's a duplicate edge
			continue;
		}
		vertexId_t src = d_seg[edge];
		vertexId_t dest= d_ind[edge];

		if (src < dest) continue;

		length_t srcLen=custing->dVD->getUsed()[src];
		length_t destLen=custing->dVD->getUsed()[dest];		

		bool avoidCalc = ((src == dest) || (destLen == 0) || (srcLen == 0));
		if(avoidCalc)
			continue;

		bool sourceSmaller = (srcLen<destLen);
        vertexId_t small = sourceSmaller? src : dest;
        vertexId_t large = sourceSmaller? dest : src;
        length_t small_len = sourceSmaller? srcLen : destLen;
        length_t large_len = sourceSmaller? destLen : srcLen;
		if(small_len==0)
			printf("hello oded\n");

        const vertexId_t* small_ptr = custing->dVD->getAdj()[small]->dst;
	    const vertexId_t* large_ptr = custing->dVD->getAdj()[large]->dst;

		triangle_t tCount = count_triangles<false,false,true,true>(
								custing,small, small_ptr, small_len,
								large,large_ptr, large_len,
								threads_per_block,firstFoundPos,
								tx%threads_per_block, outPutTriangles,
								NULL, NULL, 2,src,dest);
		__syncthreads();
	}
}


template <bool uMasked, bool vMasked, bool subtract, bool upd3rdV>
__device__ void intersectCountAsymmetric(cuStinger* custing,const length_t uLength, const length_t vLength,
    vertexId_t const * const __restrict__ uNodes, vertexId_t const * const __restrict__ vNodes,
    length_t * const __restrict__ uCurr, length_t * const __restrict__ vCurr,
    int * const __restrict__ workIndex, int * const __restrict__ workPerThread,
    int * const __restrict__ triangles, int found, triangle_t * const __restrict__ outPutTriangles, 
    vertexId_t const * const __restrict__ uMask, vertexId_t const * const __restrict__ vMask, triangle_t multiplier, 
	vertexId_t src, vertexId_t dest,vertexId_t u, vertexId_t v)
{


  if((*uCurr < uLength) && (*vCurr < vLength)){
    int comp;
    int vmask;
    int umask;
    while(*workIndex < *workPerThread){
    	vmask = (vMasked) ? vMask[*vCurr] : 0;
        umask = (uMasked) ? uMask[*uCurr] : 0;
		comp = uNodes[*uCurr] - vNodes[*vCurr];

		*triangles += (comp == 0 && !umask && !vmask);

		if (upd3rdV && comp == 0 && !umask && !vmask)
			if (subtract) {
				atomicSub(outPutTriangles + uNodes[*uCurr], multiplier);

				// Ktruss
				length_t pos_id;
				vertexId_t common = uNodes[*uCurr];

				if(dest==u)
					atomicSub(custing->dVD->adj[dest]->ew+*uCurr,1);
				else
					atomicSub(custing->dVD->adj[dest]->ew+*vCurr,1);
			}
			// else {
			// 	atomicAdd(outPutTriangles + uNodes[*uCurr], multiplier);
			// }
		*uCurr += (comp <= 0 && !vmask) || umask;
		*vCurr += (comp >= 0 && !umask) || vmask;
		*workIndex += (comp == 0&& !umask && !vmask) + 1;

		if((*vCurr == vLength) || (*uCurr == uLength)){
			break;
		}
    }
    *triangles -= ((comp == 0) && (*workIndex > *workPerThread) && (found));
  }
}



// u_len < v_len
template <bool uMasked, bool vMasked, bool subtract, bool upd3rdV>
__device__ triangle_t count_trianglesAsymmetric(cuStinger* custing,vertexId_t u, vertexId_t const * const __restrict__ u_nodes, length_t u_len,
    vertexId_t v, vertexId_t const * const __restrict__ v_nodes, length_t v_len, int threads_per_block,
    volatile vertexId_t* __restrict__ firstFound, int tId, triangle_t * const __restrict__ outPutTriangles,
    vertexId_t const * const __restrict__ uMask, vertexId_t const * const __restrict__ vMask, triangle_t multiplier, 
	vertexId_t src, vertexId_t dest)
{
	// Partitioning the work to the multiple thread of a single GPU processor. The threads should get a near equal number of the elements to Tersect - this number will be off by 1.
	int work_per_thread, diag_id;
	workPerThread(u_len, v_len, threads_per_block, tId, &work_per_thread, &diag_id);
	triangle_t triangles = 0;
	int work_index = 0,found=0;
	length_t u_min,u_max,v_min,v_max,u_curr,v_curr;

	firstFound[tId]=0;

	if(work_per_thread>0){
		// For the binary search, we are figuring out the initial poT of search.
		initialize(diag_id, u_len, v_len,&u_min, &u_max,&v_min, &v_max,&found);
    	u_curr = 0; v_curr = 0;

	    bSearch(found, diag_id, u_nodes, v_nodes, &u_len, &u_min, &u_max, &v_min,
        &v_max, &u_curr, &v_curr);

    	int sum = fixStartPoint(u_len, v_len, &u_curr, &v_curr, u_nodes, v_nodes);
    	work_index += sum;
	    if(tId > 0)
	      firstFound[tId-1] = sum;
	    triangles += sum;
	    intersectCountAsymmetric<uMasked, vMasked, subtract, upd3rdV>(custing,
	    	u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
	        &work_index, &work_per_thread, &triangles, firstFound[tId], outPutTriangles, 
	        uMask, vMask, multiplier,src,dest,u,v);
	}
	return triangles;
}





__global__ void deviceBUTwoCUOneTriangles (cuStinger* custing, BatchUpdateData *bud,
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter, bool deletion,
    length_t const * const __restrict__ redCU, length_t const * const __restrict__ redBU)
{
	length_t batchsize = *(bud->getBatchSize());
	// Partitioning the work to the multiple thread of a single GPU processor. The threads should get a near equal number of the elements to intersect - this number will be off by no more than one.
	int tx = threadIdx.x;
 	vertexId_t this_mp_start, this_mp_stop;

	length_t *d_off = bud->getOffsets();
	vertexId_t * d_ind = bud->getDst();
	vertexId_t * d_seg = bud->getSrc();

	const int blockSize = blockDim.x;
	workPerBlock(batchsize, &this_mp_start, &this_mp_stop, blockSize);

	__shared__ vertexId_t firstFound[1024];

	length_t adj_offset=tx>>shifter;
	length_t* firstFoundPos=firstFound + (adj_offset<<shifter);
	for (length_t edge = this_mp_start+adj_offset; edge < this_mp_stop; edge+=number_blocks){
		if (bud->getIndDuplicate()[edge]) // this means it's a duplicate edge
			continue;
			
		vertexId_t src = bud->getSrc()[edge];
		vertexId_t dest= bud->getDst()[edge];

		// length_t srcLen= redBU[src];
		// length_t destLen=redCU[dest];
		length_t srcLen= d_off[src+1] - d_off[src];
		length_t destLen=custing->dVD->getUsed()[dest];


		bool avoidCalc = (src == dest) || (srcLen == 0);
		if(avoidCalc)
			continue;

        vertexId_t const * const src_ptr = d_ind + d_off[src];
        vertexId_t const * const src_mask_ptr = bud->getIndDuplicate() + d_off[src];
        vertexId_t const * const dst_ptr = custing->dVD->getAdj()[dest]->dst;

		bool sourceSmaller = (srcLen<destLen);
        vertexId_t small = sourceSmaller? src : dest;
        vertexId_t large = sourceSmaller? dest : src;
        length_t small_len = sourceSmaller? srcLen : destLen;
        length_t large_len = sourceSmaller? destLen : srcLen;

        vertexId_t const * const small_ptr = sourceSmaller? src_ptr : dst_ptr;
        vertexId_t const * const small_mask_ptr = sourceSmaller? src_mask_ptr : NULL;
        vertexId_t const * const large_ptr = sourceSmaller? dst_ptr : src_ptr;
        vertexId_t const * const large_mask_ptr = sourceSmaller? NULL : src_mask_ptr;



		// triangle_t tCount=0;
		triangle_t tCount = (sourceSmaller)?
								count_trianglesAsymmetric<true,false,true,true>(
								custing,small, small_ptr, small_len,
								large,large_ptr, large_len,
								threads_per_block,firstFoundPos,
								tx%threads_per_block, outPutTriangles,
								small_mask_ptr, large_mask_ptr, 1,src,dest):
								count_trianglesAsymmetric<false,true,true,true>(
								custing,small, small_ptr, small_len,
								large,large_ptr, large_len,
								threads_per_block,firstFoundPos,
								tx%threads_per_block, outPutTriangles,
								small_mask_ptr, large_mask_ptr, 1,src,dest)
							;

		atomicSub(outPutTriangles + src, tCount*1);
		atomicSub(outPutTriangles + dest, tCount*1);
		__syncthreads();
	}
}


void callDeviceDifferenceTriangles(cuStinger& custing, BatchUpdate& bu,
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_intersection,
    const int num_intersec_perblock, const int shifter, const int thread_blocks,
    const int blockdim, bool deletion)
{
	dim3 numBlocks(1, 1);
	length_t batchsize = *(bu.getHostBUD()->getBatchSize());
	length_t nv = *(bu.getHostBUD()->getNumVertices());

	numBlocks.x = ceil((float)nv/(float)blockdim);
	length_t* redCU;length_t* redBU;

	numBlocks.x = ceil((float)(batchsize*threads_per_intersection)/(float)blockdim);

	// cout << "The block dim is " << blockdim << " and the number of blocks is  " << numBlocks.x << endl;

	// Calculate all new traingles regardless of repetition
		devicecuStingerNewTriangles<<<numBlocks, blockdim>>>(custing.devicePtr(), bu.getDeviceBUD()->devicePtr(), outPutTriangles, threads_per_intersection,num_intersec_perblock,shifter,deletion, redCU);

	// Calculate triangles formed by ALL new edges
		// deviceBUThreeTriangles<<<numBlocks,blockdim>>>(custing.devicePtr(), bu.getDeviceBUD()->devicePtr(), outPutTriangles, threads_per_intersection,num_intersec_perblock,shifter,deletion,redBU);
	
	// Calculate triangles formed by two new edges
		deviceBUTwoCUOneTriangles<<<numBlocks,blockdim>>>(custing.devicePtr(), bu.getDeviceBUD()->devicePtr(), outPutTriangles, threads_per_intersection,num_intersec_perblock,shifter,deletion,redCU,redBU);
}



