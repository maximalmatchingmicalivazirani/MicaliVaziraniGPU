#ifndef UPDATELVL
#define UPDATELVL
#include "argStructs.cuh"
#include "constants.cuh"

__device__ void updateLvlAndTenacityPassStruct_parallelDev(
                                          DSU_CU bud,
                                          updateLvlStruct updateLvl,
                                          int i){
  int n = updateLvl.supportTop[0];
  auto minlvl = [&](int u)
  { return min(updateLvl.evenlvl[u], updateLvl.oddlvl[u]); };

  auto setLvl = [&](int u, int lev)
  {
    if (lev & 1)
      updateLvl.oddlvl[u] = lev;
    else
      updateLvl.evenlvl[u] = lev;
    //verticesAtLevel[lev].push_back(u);
  };

  auto tenacity = [&](int u, int v)
  {
  if (updateLvl.mate[u] == v)
      return updateLvl.oddlvl[u] + updateLvl.oddlvl[v] + 1;
    return updateLvl.evenlvl[u] + updateLvl.evenlvl[v] + 1;
  };

  int warp_id = threadIdx.x >> 5; // local warp number
  int num_warps = blockDim.x / WARP_SZ;
  for (int vertexIndex = warp_id; vertexIndex < n; vertexIndex+=num_warps){
    int vertex = updateLvl.support[vertexIndex];
    if (vertex == updateLvl.ddfsResult[1])
      continue; // skip bud
    //myBridge[vertex] = curBridge;
    if (!(threadIdx.x % WARP_SZ)){
      updateLvl.myBridge_a[vertex]=updateLvl.curBridge[0];
      updateLvl.myBridge_b[vertex]=updateLvl.curBridge[1];
      updateLvl.myBridge_c[vertex]=updateLvl.curBridge[2];
      updateLvl.myBridge_d[vertex]=updateLvl.curBridge[3];
    }

    if (!(threadIdx.x % WARP_SZ)){
      // this part of code is only needed when bottleneck found, but it doesn't mess up anything when called on two paths
      setLvl(vertex, 2 * i + 1 - minlvl(vertex));
      if (2 * i + 1 - minlvl(vertex) < updateLvl.n[0])
        updateLvl.nonEmpty[2 * i + 1 - minlvl(vertex)]=true;
    }
    unsigned int start = updateLvl.offsets[vertex];
    unsigned int end = updateLvl.offsets[vertex + 1];
    unsigned int edgeIndex=start+lane_id();
    for(; edgeIndex < end; edgeIndex+=WARP_SZ) {
    if(updateLvl.evenlvl[vertex] > updateLvl.oddlvl[vertex] && updateLvl.edgeStatus[edgeIndex] == Bridge && 
        tenacity(vertex,updateLvl.cols[edgeIndex]) < INF && updateLvl.mate[vertex] != updateLvl.cols[edgeIndex]) {
        
        updateLvl.bridgeTenacity[edgeIndex] = tenacity(vertex,updateLvl.cols[edgeIndex]);
      }
    }
  }

  if (threadIdx.x==0){
    for (int vertexIndex = 0; vertexIndex < n; vertexIndex++){
      int vertex = updateLvl.support[vertexIndex];
      if (vertex == updateLvl.ddfsResult[1])
        continue; // skip bud
      bud.linkTo(vertex, updateLvl.ddfsResult[1]);
    }
  }
}


__device__ void updateLvlAndTenacityPassStruct_parallelDev(
                                          DSU_CU bud,
                                          updateLvlStruct updateLvl,
                                          int i,
                                          int stackIndex){
  int ListOffset = stackIndex*updateLvl.maxStackDepth[0];
  int DDFSResultOffset = stackIndex*2;
  int CurrBridgeOffset = stackIndex*4;
  int n = updateLvl.supportTop[stackIndex];
  auto minlvl = [&](int u)
  { return min(updateLvl.evenlvl[u], updateLvl.oddlvl[u]); };

  auto setLvl = [&](int u, int lev)
  {
    if (lev & 1)
      updateLvl.oddlvl[u] = lev;
    else
      updateLvl.evenlvl[u] = lev;
    //verticesAtLevel[lev].push_back(u);
  };

  auto tenacity = [&](int u, int v)
  {
  if (updateLvl.mate[u] == v)
      return updateLvl.oddlvl[u] + updateLvl.oddlvl[v] + 1;
    return updateLvl.evenlvl[u] + updateLvl.evenlvl[v] + 1;
  };

  int warp_id = threadIdx.x >> 5; // local warp number
  int num_warps = blockDim.x / WARP_SZ;
  for (int vertexIndex = warp_id; vertexIndex < n; vertexIndex+=num_warps){
    int vertex = updateLvl.support[ListOffset+vertexIndex];
    if (vertex == updateLvl.ddfsResult[DDFSResultOffset+1])
      continue; // skip bud
    //myBridge[vertex] = curBridge;
    if (!(threadIdx.x % WARP_SZ)){
      updateLvl.myBridge_a[vertex]=updateLvl.curBridge[CurrBridgeOffset+0];
      updateLvl.myBridge_b[vertex]=updateLvl.curBridge[CurrBridgeOffset+1];
      updateLvl.myBridge_c[vertex]=updateLvl.curBridge[CurrBridgeOffset+2];
      updateLvl.myBridge_d[vertex]=updateLvl.curBridge[CurrBridgeOffset+3];
    }

    if (!(threadIdx.x % WARP_SZ)){
      // this part of code is only needed when bottleneck found, but it doesn't mess up anything when called on two paths
      setLvl(vertex, 2 * i + 1 - minlvl(vertex));
      if (2 * i + 1 - minlvl(vertex) < updateLvl.n[0])
        updateLvl.nonEmpty[2 * i + 1 - minlvl(vertex)]=true;
    }
    unsigned int start = updateLvl.offsets[vertex];
    unsigned int end = updateLvl.offsets[vertex + 1];
    unsigned int edgeIndex=start+lane_id();
    for(; edgeIndex < end; edgeIndex+=WARP_SZ) {
    if(updateLvl.evenlvl[vertex] > updateLvl.oddlvl[vertex] && updateLvl.edgeStatus[edgeIndex] == Bridge && 
        tenacity(vertex,updateLvl.cols[edgeIndex]) < INF && updateLvl.mate[vertex] != updateLvl.cols[edgeIndex]) {
        
        updateLvl.bridgeTenacity[edgeIndex] = tenacity(vertex,updateLvl.cols[edgeIndex]);
      }
    }
  }

  if (threadIdx.x==0){
    for (int vertexIndex = 0; vertexIndex < n; vertexIndex++){
      int vertex = updateLvl.support[ListOffset+vertexIndex];
      if (vertex == updateLvl.ddfsResult[DDFSResultOffset+1])
        continue; // skip bud
      bud.linkTo(vertex, updateLvl.ddfsResult[DDFSResultOffset+1]);
    }
  }
}

#endif