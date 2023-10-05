#ifndef ARGSTRUCTS
#define ARGSTRUCTS
#include "CSRGraph.cuh"
struct ddfsStruct {

  ddfsStruct(CSRGraph & csr){
    offsets=thrust::raw_pointer_cast(csr.offsets_d.data());
    cols=thrust::raw_pointer_cast(csr.cols_d.data());
    ddfsPredecessorsPtr=thrust::raw_pointer_cast(csr.ddfsPredecessorsPtr_d.data());
    removed=thrust::raw_pointer_cast(csr.removed_d.data());
    predecessors=thrust::raw_pointer_cast(csr.predecessors_d.data());
    oddlvl=thrust::raw_pointer_cast(csr.oddlvl_d.data());
    evenlvl=thrust::raw_pointer_cast(csr.evenlvl_d.data());
    stack1=thrust::raw_pointer_cast(csr.stack1_d.data());
    stack2=thrust::raw_pointer_cast(csr.stack2_d.data());
    stack1Top=thrust::raw_pointer_cast(csr.stack1_Top_d.data());
    stack2Top=thrust::raw_pointer_cast(csr.stack2_Top_d.data());
    support=thrust::raw_pointer_cast(csr.support_d.data());
    supportTop=thrust::raw_pointer_cast(csr.support_Top_d.data());
    color=thrust::raw_pointer_cast(csr.color_d.data());
    globalColorCounter=thrust::raw_pointer_cast(csr.globalColorCounter_d.data());
    budAtDDFSEncounter=thrust::raw_pointer_cast(csr.budAtDDFSEncounter_d.data());
    ddfsResult=thrust::raw_pointer_cast(csr.ddfsResult_d.data());
    curBridge=thrust::raw_pointer_cast(csr.curr_bridge_d.data());

    maxStackDepth=thrust::raw_pointer_cast(csr.maxStackDepth_d.data());

  };

  unsigned int *offsets;
  unsigned int *cols;
  unsigned int *ddfsPredecessorsPtr;
  bool *removed;
  bool *predecessors; 
  int *oddlvl;
  int *evenlvl;
  unsigned int * stack1; 
  unsigned int * stack2;
  unsigned int * stack1Top; 
  unsigned int * stack2Top; 
  unsigned int * support;
  unsigned int * supportTop; 
  unsigned int * color;
  unsigned int *globalColorCounter; 
  int*budAtDDFSEncounter;
  unsigned int *ddfsResult; 
  unsigned int * curBridge;

  unsigned int * maxStackDepth;
};


struct updateLvlStruct {

  updateLvlStruct(CSRGraph & csr){
    support=thrust::raw_pointer_cast(csr.support_d.data());
    oddlvl=thrust::raw_pointer_cast(csr.oddlvl_d.data());
    evenlvl=thrust::raw_pointer_cast(csr.evenlvl_d.data());
    mate=thrust::raw_pointer_cast(csr.mate_d.data());
    offsets=thrust::raw_pointer_cast(csr.offsets_d.data());
    cols=thrust::raw_pointer_cast(csr.cols_d.data());
    edgeStatus=thrust::raw_pointer_cast(csr.edgeStatus_d.data());
    bridgeTenacity=thrust::raw_pointer_cast(csr.bridgeTenacity_d.data());
    myBridge_a=thrust::raw_pointer_cast(csr.myBridge_a_d.data());
    myBridge_b=thrust::raw_pointer_cast(csr.myBridge_b_d.data());
    myBridge_c=thrust::raw_pointer_cast(csr.myBridge_c_d.data());
    myBridge_d=thrust::raw_pointer_cast(csr.myBridge_d_d.data());
    supportTop=thrust::raw_pointer_cast(csr.support_Top_d.data());
    ddfsResult=thrust::raw_pointer_cast(csr.ddfsResult_d.data());
    curBridge=thrust::raw_pointer_cast(csr.curr_bridge_d.data());
    nonEmpty=thrust::raw_pointer_cast(csr.nonempty_d.data());
    n=thrust::raw_pointer_cast(csr.n_d.data());

    maxStackDepth=thrust::raw_pointer_cast(csr.maxStackDepth_d.data());
  }
  unsigned int *support;
  int *oddlvl;
  int *evenlvl;
  int *mate;
  unsigned int *offsets;
  unsigned int *cols;
  char *edgeStatus;
  int *bridgeTenacity;
  int *myBridge_a;
  int *myBridge_b;
  int *myBridge_c;
  int *myBridge_d;
  unsigned int *supportTop;
  unsigned int *ddfsResult; 
  unsigned int * curBridge;
  bool * nonEmpty;
  int *n;

  unsigned int * maxStackDepth;

};

struct APStruct {
  APStruct(CSRGraph & csr){
    stackDepth=thrust::raw_pointer_cast(csr.stackDepth_d.data());
    offsets=thrust::raw_pointer_cast(csr.offsets_d.data());
    cols=thrust::raw_pointer_cast(csr.cols_d.data());
    oddlvl=thrust::raw_pointer_cast(csr.oddlvl_d.data());
    evenlvl=thrust::raw_pointer_cast(csr.evenlvl_d.data());
    edgeStatus=thrust::raw_pointer_cast(csr.edgeStatus_d.data());
    pred=thrust::raw_pointer_cast(csr.predecessors_d.data());
    childsInDDFSTreePtr=thrust::raw_pointer_cast(csr.childsInDDFSTreePtr_d.data());
    removed=thrust::raw_pointer_cast(csr.removed_d.data());
    myBridge_a=thrust::raw_pointer_cast(csr.myBridge_a_d.data());
    myBridge_b=thrust::raw_pointer_cast(csr.myBridge_b_d.data());
    myBridge_c=thrust::raw_pointer_cast(csr.myBridge_c_d.data());
    myBridge_d=thrust::raw_pointer_cast(csr.myBridge_d_d.data());
    color=thrust::raw_pointer_cast(csr.color_d.data());
    removedPredecessorsSize=thrust::raw_pointer_cast(csr.removedPredecessorsSize_d.data());
    predecessor_count=thrust::raw_pointer_cast(csr.predecessors_count_d.data());
    removedVerticesQueue=thrust::raw_pointer_cast(csr.removedVerticesQueue_d.data());
    removedVerticesQueueFront=thrust::raw_pointer_cast(csr.removedVerticesFront_d.data());
    removedVerticesQueueBack=thrust::raw_pointer_cast(csr.removedVerticesBack_d.data());
    foundPath=thrust::raw_pointer_cast(csr.foundPath_d.data());
    budAtDDFSEncounter=thrust::raw_pointer_cast(csr.budAtDDFSEncounter_d.data());
    mate=thrust::raw_pointer_cast(csr.mate_d.data());
    ddfsResult=thrust::raw_pointer_cast(csr.ddfsResult_d.data());

    maxStackDepth=thrust::raw_pointer_cast(csr.maxStackDepth_d.data());
  }
  int * stackDepth;
  unsigned int * offsets;
  unsigned int * cols;
  int * oddlvl;
  int * evenlvl;
  char * edgeStatus;
  bool * pred;
  unsigned int *childsInDDFSTreePtr;
  bool *removed;
  int * myBridge_a; 
  int * myBridge_b; 
  int * myBridge_c; 
  int * myBridge_d; 
  unsigned int * color;
  unsigned int * removedPredecessorsSize; 
  unsigned int * predecessor_count;
  int * removedVerticesQueue; 
  unsigned int * removedVerticesQueueFront; 
  unsigned int * removedVerticesQueueBack; 
  bool * foundPath;
  int * budAtDDFSEncounter; 
  int * mate;
  unsigned int *ddfsResult; 

  unsigned int * maxStackDepth;

};
#endif