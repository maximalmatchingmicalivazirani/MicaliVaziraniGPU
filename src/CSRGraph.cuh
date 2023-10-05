#ifndef CSRGRAPH
#define CSRGRAPH
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include "DSU.cuh"
#include <thrust/count.h>
#include <thrust/sort.h>


// kernel function
template <typename T>
__global__ void setNumInArray(T *arrays, T *index, T *value, int num_index)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= num_index || index[tid] < tid)
    return;
  arrays[index[tid]] = value[tid];
}

__global__ void resetN(
    unsigned int *childsInDDFSTreePtr,
    bool *removed,
    unsigned int *color,
    unsigned int *ddfsPredecessorsPtr_d,
    unsigned int *removedPredecessorsSize_d,
    unsigned int *predecessors_count_d,
    int *evenlvl_d,
    int *oddlvl_d,
    int *size,
    int *directParent,
    int *link,
    int *groupRoot,
    unsigned int *globalColorCounter,
    int n){
  int threadID = threadIdx.x + blockIdx.x*blockDim.x;
  if (threadID==0)
  globalColorCounter[0]=1;
  if (threadID >=n) return;
  childsInDDFSTreePtr[threadID]=0;
  removed[threadID]=false;
  color[threadID]=0;
  ddfsPredecessorsPtr_d[threadID]=0;
  removedPredecessorsSize_d[threadID]=0;
  predecessors_count_d[threadID]=0;
  evenlvl_d[threadID]=1e9;
  oddlvl_d[threadID]=1e9;
  size[threadID]=1;
  directParent[threadID]=-1;
  link[threadID]=threadID;
  groupRoot[threadID]=threadID;
}

__global__ void resetM(
    char *edgeStatus_d,
    bool *predecessors_d,
    int *budAtDDFSEncounter_d,
    int *bridgeTenacity_d,
    int *size,
    int *directParent,
    int *link,
    int *groupRoot,
    int m){
  int threadID = threadIdx.x + blockIdx.x*blockDim.x;
  if (threadID >=m) return;
  edgeStatus_d[threadID]=0;
  predecessors_d[threadID]=false;
  budAtDDFSEncounter_d[threadID]=-1;
  bridgeTenacity_d[threadID]=0;
  #if DEBUGDSU
  size[threadID]=1;
  directParent[threadID]=-1;
  link[threadID]=threadID;
  groupRoot[threadID]=threadID;
  #endif
}

__global__ void resetM(
    char *edgeStatus_d,
    bool *predecessors_d,
    int *budAtDDFSEncounter_d,
    int *bridgeTenacity_d,
    int m){
  int threadID = threadIdx.x + blockIdx.x*blockDim.x;
  if (threadID >=m) return;
  edgeStatus_d[threadID]=0;
  predecessors_d[threadID]=false;
  budAtDDFSEncounter_d[threadID]=-1;
  bridgeTenacity_d[threadID]=0;
}

struct CSRGraph
{
  const int INF = 1e9;

  CSRGraph(int _n, int _m)
  {
    m = _m;
    n = _n;
    verticesPlusEdges=n+2*m; // bridge id
    
    //VertexClaimedByBridge_d.resize(n,INF);

    claimed_A_d.resize(2*m,INF);
    //claimed_B_d.resize(2*m,INF);
    smallest_Ever_Seen_d.resize(2*m,INF);
    //claimed_Track_d.resize(2*m,INF);

    //claimed_A_h.resize(2*m,INF);
    //claimed_B_h.resize(2*m,INF);
    
    claimedNew_d.resize(1,false);
    bridgeList_groupRoot_d.resize(2*m);
    //bridgeList_groupRoot2_d.resize(2*m);

    keylabel_bridges_d.resize(2*m);
    nonzerodegrees_bridges_d.resize(2*m);
    degrees_bridges_d.resize(2*m);
    offsets_bridges_d.resize((2*m)+1);

    offsets_h.resize(n + 1);
    degrees_h.resize(n);

    rows_h.resize(2 * m);
    cols_h.resize(2 * m);
    vals_h.resize(2 * m, 1);

    rows_d.resize(2 * m);
    // This will be the dst ptr array.
    cols_d.resize(2 * m);
    vals_d.resize(2 * m, 1);

    offsets_d.resize(n + 1);

    keylabel_d.resize(n);
    nonzerodegrees_d.resize(n);
    // This will be the degrees array.
    degrees_d.resize(n);


  }

  void allocateMatchingDataStructures(int _maxStackDepth, int _numStacks=1){
    maxStackDepth=_maxStackDepth;
    numStacks=_numStacks;

    // Use for dynamically allocated SM
    stackDepth_h.resize(1);
    stackDepth_d.resize(1);

    // Augment Path data structures
    // Required to augment paths on GPU
    evenlvl_h.resize(n,INF); // n
    oddlvl_h.resize(n,INF); // n
    predecessors_h.resize(2*m); // m
    predecessors_count_h.resize(n);
    // BUD
    // DSU structure
    bud = allocate_DSU(n,maxStackDepth,numStacks);
    bud.n=n;
    bud.m=m;
    //bud.resize(n, m);

    #if DEBUGDSU
    // For CC separation of bridges
    budCC = allocate_DSU(2*m,2*m,1);
    // THIS WAS KILLING ME!!!
    //budCC.n=n;
    budCC.m=m;
    #endif

    removed_h.resize(n); // n
    // myBridge
    myBridge_a_h.resize(n); // n
    myBridge_b_h.resize(n); // n
    myBridge_c_h.resize(n); // n
    myBridge_d_h.resize(n); // n


    color_h.resize(n); // n
    mate_h.resize(n,-1); // n
    removedVerticesQueue_h.resize(n); // n
    childsInDDFSTreePtr_h.resize(n); // n
    // childsInDDFSTree -> baseAtTimeOfEncounter
    baseAtTimeOfEncounter_h.resize(2*m); // m
    edgeStatus_h.resize(2*m); // m
    bridgeTenacity_h.resize(2*m);
    bridgeList_h.resize(2*m);

    verticesInLevel_h.resize(n); // 1
    verticesInLevel_counter_h.resize(1);
    support_h.resize(n); // m
    foundPath_h.resize(1);
    n_h.resize(1);
    n_h[0]=n;


    // Booleans which can be converted to bits.
    nonempty_d.resize(n);
    predecessors_d.resize(2*m); // m
    //removed_d.resize(n); // n


    // Augment Path data structures
    // Required to augment paths on GPU
    evenlvl_d.resize(n,INF); // n
    oddlvl_d.resize(n,INF); // n
    predecessors_count_d.resize(n);
    // BUD
    // DSU structure
    //bud.resize(n, m);

    // myBridge
    myBridge_a_d.resize(n); // n
    myBridge_b_d.resize(n); // n
    myBridge_c_d.resize(n); // n
    myBridge_d_d.resize(n); // n
    mate_d.resize(n,-1); // n
    childsInDDFSTreePtr_d.resize(n); // n
    // childsInDDFSTree -> baseAtTimeOfEncounter

    edgeStatus_d.resize(2*m); // m
    bridgeTenacity_d.resize(2*m);
    baseAtTimeOfEncounter_d.resize(2*m); // m
    bridgeList_d.resize(2*m); // 1
    verticesInLevel_d.resize(n); // 1
    verticesInLevel_counter_d.resize(1);
    bridgeList_counter_d.resize(1); // 1
    bridgeList_counter_h.resize(1); // 1

    maxPairs=4*m;

    bu_bfs_key_d.resize(maxPairs); // bridge id
    bu_bfs_val_d.resize(maxPairs); // vertex id
    bu_bfs_Top_d.resize(1,0); // bridge id

    /*
    bu_bfs_key_h.resize(maxPairs); // bridge id
    bu_bfs_val_h.resize(maxPairs); // vertex id
    bu_bfs_Top_h.resize(1,0); // bridge id
    bu_bfs_key_buffer_h.resize(maxPairs); // bridge id 
    bu_bfs_val_buffer_h.resize(maxPairs); // vertex id
    bu_bfs_buffer_Top_h.resize(1,0); // bridge id
    bu_bfs_key_root_h.resize(maxPairs); // bridge id 
    bu_bfs_Top_root_h.resize(1,0); // bridge id 
    */

    bu_bfs_key_buffer_d.resize(maxPairs); // bridge id 
    bu_bfs_val_buffer_d.resize(maxPairs); // vertex id
    bu_bfs_buffer_Top_d.resize(1,0); // bridge id

    bu_bfs_key_root_d.resize(maxPairs); // bridge id 
    bu_bfs_val_root_d.resize(maxPairs); // bridge id 
    bu_bfs_Top_root_d.resize(1,0); // bridge id 


    // Timing arrays
    verticesTraversed_d.resize(n,0); // 1
    K1Time_d.resize(n,0); // 1
    K2Time_d.resize(n,0); // 1
    K3Time_d.resize(n,0); // 1
    count_d.resize(n,0); // 1

    verticesTraversed_h.resize(n,0); // 1
    K1Time_h.resize(n,0); // 1
    K2Time_h.resize(n,0); // 1
    K3Time_h.resize(n,0); // 1
    count_h.resize(n,0); // 1

    ddfsPredecessorsPtr_d.resize(n); // m
    color_d.resize(n); // n
    budAtDDFSEncounter_d.resize((2*m)); // m
    removed_d.resize(n); // n
    globalColorCounter_d.resize(1); // m

    // Stack array size
    maxStackDepth_d.resize(1,maxStackDepth); // m
    // Stack array size
    // Stack arrays
    stack1_d.resize(maxStackDepth*numStacks); // m
    stack2_d.resize(maxStackDepth*numStacks); // m
    support_d.resize(maxStackDepth*numStacks); // m
    removedVerticesQueue_d.resize(maxStackDepth*numStacks); // n
    // Stack arrays


    // Stack scalars
    stack1_Top_d.resize(1*numStacks); // 1
    stack2_Top_d.resize(1*numStacks); // 1
    support_Top_d.resize(1*numStacks); // 1
    ddfsResult_d.resize(2*numStacks);
    curr_bridge_d.resize(4*numStacks);
    removedVerticesFront_d.resize(1*numStacks,0);
    removedVerticesBack_d.resize(1*numStacks,0);
    // Stack scalars

    removedPredecessorsSize_d.resize(n);



    // 
    connectedComponentTop_d.resize(1,0);

    foundPath_d.resize(1);
    n_d.resize(1);
    n_d=n_h;
    cudaMallocHost((void**)&h_foundPathPinned, sizeof(bool)); // host pinned
    cudaMallocHost((void**)&h_nonemptyPinned, sizeof(bool)); // host pinned
    cudaMallocHost((void**)&h_numBridges, sizeof(unsigned int)); // host pinned
    cudaMallocHost((void**)&bu_bfs_Top_Pinned, sizeof(unsigned long long int)); // host pinned
    cudaMallocHost((void**)&bu_bfs_buffer_Top_Pinned, sizeof(unsigned long long int)); // host pinned
    cudaMallocHost((void**)&bu_bfs_Top_Root_Pinned, sizeof(unsigned long long int)); // host pinned
    cudaMallocHost((void**)&h_claimedNewPinned, sizeof(bool)); // host pinned

  }

  void createOffsets()
  {
    rows_d = rows_h;
    cols_d = cols_h;
    thrust::sort_by_key(thrust::device, rows_d.begin(), rows_d.end(), cols_d.begin());
    thrust::pair<thrust::device_vector<unsigned int>::iterator, thrust::device_vector<unsigned int>::iterator> new_end;
    new_end = thrust::reduce_by_key(thrust::device, rows_d.begin(), rows_d.end(), vals_d.begin(), keylabel_d.begin(), nonzerodegrees_d.begin());
    int block_size = 64;
    int num_blocks = (n + block_size - 1) / block_size;
    unsigned int *degrees_ptr_d = thrust::raw_pointer_cast(degrees_d.data());
    unsigned int *keylabel_ptr_d = thrust::raw_pointer_cast(keylabel_d.data());
    unsigned int *nonzerodegrees_ptr_d = thrust::raw_pointer_cast(nonzerodegrees_d.data());
    setNumInArray<unsigned int><<<num_blocks, block_size>>>(degrees_ptr_d, keylabel_ptr_d, nonzerodegrees_ptr_d, n);
    thrust::inclusive_scan(thrust::device, degrees_d.begin(), degrees_d.end(), offsets_d.begin() + 1); // in-place scan
    offsets_h = offsets_d;
    degrees_h = degrees_d;
    rows_h = rows_d;
    cols_h = cols_d;

    keylabel_d.clear();
    //vals_d.clear();
    nonzerodegrees_d.clear();
  }


  void createBridgeOffsets()
  {
    thrust::sort_by_key(thrust::device, bridgeList_groupRoot_d.begin(), bridgeList_groupRoot_d.begin()+h_numBridges[0], bridgeList_d.begin());
    thrust::pair<thrust::device_vector<unsigned int>::iterator, thrust::device_vector<unsigned int>::iterator> new_end;
    new_end = thrust::reduce_by_key(thrust::device, 
                                    bridgeList_groupRoot_d.begin(), 
                                    bridgeList_groupRoot_d.begin()+h_numBridges[0], 
                                    vals_d.begin(), 
                                    keylabel_bridges_d.begin(), 
                                    nonzerodegrees_bridges_d.begin());
    // Calculate the number of unique keys
    numConnectedComponents = thrust::distance(keylabel_bridges_d.begin(), new_end.first);    
    thrust::inclusive_scan(thrust::device, nonzerodegrees_bridges_d.begin(), nonzerodegrees_bridges_d.begin()+numConnectedComponents, offsets_bridges_d.begin() + 1); // in-place scan
    /*
     OKAY NOW I CAN HAVE ASSIGNED BRIDGES TO CC's
     NUM_CC = csr.numConnectedComponents
      for (int cc = 0; cc<NUM_CC;++cc){
        int start=offsets_bridges_d[cc];
        int end=offsets_bridges_d[cc+1];
        for (; start < end; start++)
          DDFS(bridgeList_d[start])
      }
    */
  }


  void reset(){
    /*
    thrust::fill(edgeStatus_h.begin(), edgeStatus_h.end(), 0); // or 999999.f if you prefer
    thrust::fill(predecessors_h.begin(), predecessors_h.end(), false); // or 999999.f if you prefer
    thrust::fill(predecessors_count_h.begin(), predecessors_count_h.end(), 0); // or 999999.f if you prefer
    thrust::fill(evenlvl_h.begin(), evenlvl_h.end(), INF); // or 999999.f if you prefer
    thrust::fill(oddlvl_h.begin(), oddlvl_h.end(), INF); // or 999999.f if you prefer
    thrust::fill(bridgeTenacity_h.begin(), bridgeTenacity_h.end(), 0); // or 999999.f if you prefer
    */

    // 1
    thrust::fill(globalColorCounter_d.begin(), globalColorCounter_d.end(), 1); // or 999999.f if you prefer

    // n
    thrust::fill(childsInDDFSTreePtr_d.begin(), childsInDDFSTreePtr_d.end(), 0); // or 999999.f if you prefer
    thrust::fill(removed_d.begin(), removed_d.end(), false); // or 999999.f if you prefer
    thrust::fill(color_d.begin(), color_d.end(), 0); // or 999999.f if you prefer
    thrust::fill(ddfsPredecessorsPtr_d.begin(), ddfsPredecessorsPtr_d.end(), 0); // or 999999.f if you prefer
    thrust::fill(removedPredecessorsSize_d.begin(), removedPredecessorsSize_d.end(), 0); // or 999999.f if you prefer
    thrust::fill(predecessors_count_d.begin(), predecessors_count_d.end(), 0); // or 999999.f if you prefer
    thrust::fill(evenlvl_d.begin(), evenlvl_d.end(), INF); // or 999999.f if you prefer
    thrust::fill(oddlvl_d.begin(), oddlvl_d.end(), INF); // or 999999.f if you prefer
    bud.reset();

    // 2*m
    thrust::fill(edgeStatus_d.begin(), edgeStatus_d.end(), 0); // or 999999.f if you prefer
    thrust::fill(predecessors_d.begin(), predecessors_d.end(), false); // or 999999.f if you prefer
    thrust::fill(budAtDDFSEncounter_d.begin(), budAtDDFSEncounter_d.end(), -1); // or 999999.f if you prefer
    thrust::fill(bridgeTenacity_d.begin(), bridgeTenacity_d.end(), 0); // or 999999.f if you prefer

  }

  unsigned long long int m; // Number of Edges
  unsigned long long int maxPairs; // Number of Edges
  unsigned long long int n; // Number of Vertices
  unsigned long long int gm_cnt;
  unsigned long long int na_cnt;
  unsigned long long int total_cnt;
  double gm_sec;
  double na_sec;
  double total_sec;
  unsigned long long int verticesPlusEdges;
  unsigned int numConnectedComponents;
  int maxSM;
  int maxStackDepth;
  int numStacks;
  int depthCutoff;

  thrust::device_vector<int> stackDepth_d;
  thrust::host_vector<int> stackDepth_h;

  thrust::host_vector<unsigned int> rows_h;
  thrust::host_vector<unsigned int> cols_h;
  thrust::host_vector<char> vals_h;

  thrust::host_vector<unsigned int> offsets_h;
  thrust::host_vector<unsigned int> keylabel_h;
  thrust::host_vector<unsigned int> nonzerodegrees_h;
  thrust::host_vector<unsigned int> degrees_h;

  thrust::device_vector<unsigned int> rows_d;
  thrust::device_vector<unsigned int> cols_d;
  thrust::device_vector<unsigned int> vals_d;

  thrust::device_vector<unsigned int> offsets_d;
  thrust::device_vector<unsigned int> offsets_bridges_d;

  thrust::device_vector<unsigned int> keylabel_d;
  thrust::device_vector<unsigned int> nonzerodegrees_d;
  thrust::device_vector<unsigned int> keylabel_bridges_d;
  thrust::device_vector<unsigned int> nonzerodegrees_bridges_d;
  thrust::device_vector<unsigned int> degrees_d;
  thrust::device_vector<unsigned int> degrees_bridges_d;

  thrust::device_vector<unsigned int> connectedComponentTop_d;



  // Required to augment paths on GPU
  thrust::host_vector<int> evenlvl_h; // n
  thrust::host_vector<int> oddlvl_h; // n
  thrust::host_vector<bool> predecessors_h; // m
  thrust::host_vector<unsigned int> predecessors_count_h; // m

  thrust::host_vector<int> verticesInLevel_h; // m
  thrust::device_vector<int> verticesInLevel_d; // m
  thrust::device_vector<unsigned int> verticesInLevel_counter_d;
  thrust::host_vector<unsigned int> verticesInLevel_counter_h;
  thrust::host_vector<unsigned int> bridgeList_h;

  //thrust::host_vector<int> bridgesInLevel_h;
  //thrust::device_vector<int> bridgesInLevel_d;
  //thrust::device_vector<unsigned int> bridgeList_u_d;
  //thrust::device_vector<unsigned int> bridgeList_v_d;
  thrust::device_vector<unsigned int> bridgeList_d;
  thrust::device_vector<unsigned int> bridgeList_counter_d;
  thrust::host_vector<unsigned int> bridgeList_counter_h;

  thrust::host_vector<unsigned int> bu_bfs_key_h;
  thrust::host_vector<unsigned int> bu_bfs_val_h;
  thrust::device_vector<unsigned int> bu_bfs_key_d;
  thrust::device_vector<unsigned int> bu_bfs_val_d;
  thrust::device_vector<unsigned long long int> bu_bfs_Top_d;
  thrust::host_vector<unsigned long long int> bu_bfs_Top_h;

  thrust::host_vector<unsigned int> bu_bfs_key_buffer_h;
  thrust::host_vector<unsigned int> bu_bfs_val_buffer_h;
  thrust::device_vector<unsigned int> bu_bfs_key_buffer_d;
  thrust::device_vector<unsigned int> bu_bfs_val_buffer_d;
  thrust::device_vector<unsigned long long int> bu_bfs_buffer_Top_d;
  thrust::host_vector<unsigned long long int> bu_bfs_buffer_Top_h;

  thrust::device_vector<unsigned int> bu_bfs_key_root_d;
  thrust::host_vector<unsigned int> bu_bfs_key_root_h;

  ;
  thrust::device_vector<unsigned int> bu_bfs_val_root_d;
  thrust::device_vector<unsigned long long int> bu_bfs_Top_root_d;
  thrust::host_vector<unsigned long long int> bu_bfs_Top_root_h;

  thrust::device_vector<unsigned int> VertexClaimedByBridge_d;
  thrust::device_vector<unsigned int> claimed_A_d;
  thrust::device_vector<unsigned int> claimed_B_d;
  thrust::device_vector<unsigned int> smallest_Ever_Seen_d;
  thrust::device_vector<unsigned int> claimed_Track_d;

  thrust::host_vector<unsigned int> claimed_A_h;
  thrust::host_vector<unsigned int> claimed_B_h;

  thrust::device_vector<unsigned int> bridgeList_groupRoot_d;
  thrust::device_vector<unsigned int> bridgeList_groupRoot2_d;

  // Timing vectors
  thrust::device_vector<unsigned int> count_d;
  thrust::device_vector<unsigned int> verticesTraversed_d;
  thrust::device_vector<long long int> K1Time_d;
  thrust::device_vector<long long int> K2Time_d;
  thrust::device_vector<long long int> K3Time_d;

  thrust::host_vector<unsigned int> count_h;
  thrust::host_vector<unsigned int> verticesTraversed_h;
  thrust::host_vector<long long int> K1Time_h;
  thrust::host_vector<long long int> K2Time_h;
  thrust::host_vector<long long int> K3Time_h;
  
  // BUD
  DSU_CU bud;
  
  // BUD for edge separation
  DSU_CU budCC;

  thrust::host_vector<bool> removed_h; // n
  // myBridge
  thrust::host_vector<int> myBridge_a_h; // n
  thrust::host_vector<int> myBridge_b_h; // n
  thrust::host_vector<int> myBridge_c_h; // n
  thrust::host_vector<int> myBridge_d_h; // n


  thrust::host_vector<int> color_h; // n
  thrust::host_vector<int> mate_h; // n
  thrust::host_vector<int> removedVerticesQueue_h; // n
  thrust::host_vector<int> childsInDDFSTreePtr_h; // n
  // childsInDDFSTree -> baseAtTimeOfEncounter
  thrust::host_vector<int> baseAtTimeOfEncounter_h; // m
  thrust::host_vector<char> edgeStatus_h; // m
  thrust::host_vector<int> bridgeTenacity_h; // m

  // Required to augment paths on GPU

  thrust::device_vector<bool> predecessors_d; // m

  // myBridge
  thrust::device_vector<int> myBridge_a_d; // n
  thrust::device_vector<int> myBridge_b_d; // n
  thrust::device_vector<int> myBridge_c_d; // n
  thrust::device_vector<int> myBridge_d_d; // n


  thrust::device_vector<int> mate_d; // n
  thrust::device_vector<int> removedVerticesQueue_d; // n
  // childsInDDFSTree -> baseAtTimeOfEncounter
  thrust::device_vector<int> baseAtTimeOfEncounter_d; // m


  // Reset every call to bfs()
  // n
  thrust::device_vector<unsigned int> childsInDDFSTreePtr_d; // n
  thrust::device_vector<unsigned int> color_d; // n
  thrust::device_vector<bool> removed_d; // n
  thrust::device_vector<unsigned int> ddfsPredecessorsPtr_d; // n
  thrust::device_vector<unsigned int> removedPredecessorsSize_d;
  thrust::device_vector<unsigned int> predecessors_count_d; // n
  thrust::device_vector<int> evenlvl_d; // n
  thrust::device_vector<int> oddlvl_d; // n

  // 2*m
  thrust::device_vector<char> edgeStatus_d; // m
  thrust::device_vector<int> bridgeTenacity_d; // m
  thrust::device_vector<int> budAtDDFSEncounter_d; // m

  thrust::device_vector<unsigned int> maxStackDepth_d;


  // Reset every bridge iteration.
  thrust::device_vector<unsigned int> stack1_d; // m
  thrust::device_vector<unsigned int> stack2_d; // m

  thrust::host_vector<unsigned int> support_h; // m

  thrust::device_vector<unsigned int> support_d; // m
  thrust::device_vector<unsigned int> globalColorCounter_d; // m


  thrust::device_vector<unsigned int> removedVerticesFront_d; // 1
  thrust::device_vector<unsigned int> removedVerticesBack_d; // 1

  thrust::device_vector<unsigned int> stack1_Top_d; // 1
  thrust::device_vector<unsigned int> stack2_Top_d; // 1
  thrust::device_vector<unsigned int> support_Top_d; // 1

  thrust::device_vector<unsigned int> ddfsResult_d; // 2
  thrust::device_vector<unsigned int> curr_bridge_d; // 4

  thrust::host_vector<bool> foundPath_h; // 1

  thrust::device_vector<bool> foundPath_d; // 1

  thrust::device_vector<bool> nonempty_d; // m
  thrust::device_vector<int> n_d; // m
  thrust::host_vector<int> n_h; // m
  thrust::device_vector<bool> claimedNew_d; // m

  // pinned memory
  bool *h_foundPathPinned, *h_nonemptyPinned, *h_claimedNewPinned;
  int * h_verticesInLevel_counterPinned;
  unsigned int*h_numBridges;
  unsigned long long int* bu_bfs_Top_Pinned, *bu_bfs_buffer_Top_Pinned, *bu_bfs_Top_Root_Pinned;

};

#endif