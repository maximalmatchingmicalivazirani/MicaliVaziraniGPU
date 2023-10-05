#ifndef CONNECTEDCOMPONENTS
#define CONNECTEDCOMPONENTS
const int INF2 = 1e9;

__global__ void linkAllEdges(
                                          DSU_CU budCC,
                                          unsigned int * VertexClaimedByBridge,
                                          unsigned int * edges_x,
                                          unsigned int * edges_y,
                                          unsigned long long int * numEdges){
    for (int edgeIndex = 0; edgeIndex < numEdges[0]; edgeIndex++){
        auto x = edges_x[edgeIndex];
        auto y = edges_y[edgeIndex];
        if (VertexClaimedByBridge[y]==INF2){
            VertexClaimedByBridge[y]=x;
            continue;
        }
        if (budCC[VertexClaimedByBridge[y]]!=budCC[x]){
            budCC.linkTo(budCC[x],budCC[VertexClaimedByBridge[y]]);
        }
    }
}


__global__ void setGroupRoot(
                                        DSU_CU budCC,
                                        unsigned int * bridgeIndices,
                                        unsigned int * bridgeGroupRoot,
                                        unsigned int * numBridges){
    unsigned int edgeIndex = threadIdx.x + blockIdx.x*(blockDim.x);
    if(edgeIndex >= numBridges[0]) return;
    bridgeGroupRoot[edgeIndex]=budCC(bridgeIndices[edgeIndex]);
    //assert(budCC.directParent[bridgeGroupRoot[edgeIndex]]==-1);
}

__global__ void setGroupRoot(
                                        unsigned int * claimed_A,
                                        unsigned int * bridgeIndices,
                                        unsigned int * bridgeGroupRoot,
                                        unsigned int * numBridges){
    unsigned int edgeIndex = threadIdx.x + blockIdx.x*(blockDim.x);
    if(edgeIndex >= numBridges[0]) return;
    bridgeGroupRoot[edgeIndex]=claimed_A[bridgeIndices[edgeIndex]];
    //assert(budCC.directParent[bridgeGroupRoot[edgeIndex]]==-1);
}

__global__ void linkAllEdges(
                                          DSU_CU budCC,
                                          unsigned int * VertexClaimedByBridge,
                                          unsigned int * edgeLL_next,
                                          unsigned int * edgeLL_prev,
                                          unsigned int * edgeLL_rank,
                                          unsigned int * edges_x,
                                          unsigned int * edges_y,
                                          unsigned long long int * numEdges){
    for (int edgeIndex = 0; edgeIndex < numEdges[0]; edgeIndex++){
        auto x = edges_x[edgeIndex];
        auto y = edges_y[edgeIndex];
        if (VertexClaimedByBridge[y]==INF2){
            VertexClaimedByBridge[y]=x;
            continue;
        }
        auto headX = edgeLL_next[x];
        auto currX = edgeLL_next[x];

        while(headX != edgeLL_prev[x]){
        auto headY = edgeLL_next[VertexClaimedByBridge[y]];
        }
    }
}


#endif