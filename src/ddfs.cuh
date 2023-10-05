#ifndef DDFS
#define DDFS
#include "argStructs.cuh"
#include "ddfsMove.cuh"

__device__ void ddfsStructDev(DSU_CU bud,
                              ddfsStruct ddfs,
                              int src, int dst,
                              unsigned int * count) {
    if (ddfs.removed[bud[src]] || ddfs.removed[bud[dst]])
      return;
    ddfs.stack1Top[0]=0;
    ddfs.stack2Top[0]=0;
    ddfs.supportTop[0]=0;

    auto minlvl = [&](int u)
    { return min(ddfs.evenlvl[u], ddfs.oddlvl[u]); };
    
    ddfs.stack1[ddfs.stack1Top[0]++]=bud[src];
    ddfs.stack2[ddfs.stack2Top[0]++]=bud[dst];
    //vector<int> Sr = {bud[src]}, Sg = {bud[dst]};
    //if(Sr[0] == Sg[0])
    //    return {Sr[0],Sg[0]};
    if (ddfs.stack1[0]==ddfs.stack2[0]){
        //printf("stack1[0]=%d == stack2[0]=%d\n",stack1[0],stack2[0]);
        //return (uint64_t) stack1[0] << 32 | stack2[0];
        ddfs.ddfsResult[0]=ddfs.stack1[0];
        ddfs.ddfsResult[1]=ddfs.stack2[0];
        ddfs.curBridge[0] = src;
        ddfs.curBridge[1] = dst;
        ddfs.curBridge[2] = bud[src];
        ddfs.curBridge[3] = bud[dst];
        count[0]++;
        return;

    }
    //out_support = {Sr[0], Sg[0]};
    ddfs.support[ddfs.supportTop[0]++]=ddfs.stack1[0];
    ddfs.support[ddfs.supportTop[0]++]=ddfs.stack2[0];

    //int newRed = color[Sr[0]] = ++globalColorCounter, newGreen = color[Sg[0]] = ++globalColorCounter;
    //assert(newRed == (newGreen^1));
    int newRed = ddfs.color[ddfs.stack1[0]] = ++ddfs.globalColorCounter[0], newGreen = ddfs.color[ddfs.stack2[0]] = ++ddfs.globalColorCounter[0];
    assert(newRed == (newGreen^1));
    

    for(;;) {
        //printf("IN FOR\n");
        //if found two disjoint paths
        //if(minlvl(Sr.back()) == 0 && minlvl(Sg.back()) == 0)
        if(minlvl(ddfs.stack1[ddfs.stack1Top[0]-1]) == 0 && minlvl(ddfs.stack2[ddfs.stack2Top[0]-1]) == 0){

            //printf("stack1[%d]=%d\n",stack1Top[0]-1,stack1[stack1Top[0]-1]);
            //printf("stack2[%d]=%d\n",stack2Top[0]-1,stack2[stack2Top[0]-1]);
            //printf("minlvl(graph,stack1[stack1Top[0]]) == 0 && minlvl(graph,stack2[stack2Top[0]]) == 0\n");
            ddfs.ddfsResult[0]=ddfs.stack1[ddfs.stack1Top[0]-1];
            ddfs.ddfsResult[1]=ddfs.stack2[ddfs.stack2Top[0]-1];
            ddfs.curBridge[0] = src;
            ddfs.curBridge[1] = dst;
            ddfs.curBridge[2] = bud[src];
            ddfs.curBridge[3] = bud[dst];
            count[0]++;
            return;
        }
        int b;
        //if(minlvl(Sr.back()) >= minlvl(Sg.back()))
        //printf("stack1[%d-1]=%d ml %d stack2[%d-1]=%d ml %d \n",stack1Top[0],stack1[stack1Top[0]-1],minlvl(stack1[stack1Top[0]-1]),stack2Top[0],stack2[stack2Top[0]-1],minlvl(stack2[stack2Top[0]-1]));
        if(minlvl(ddfs.stack1[ddfs.stack1Top[0]-1]) >= minlvl(ddfs.stack2[ddfs.stack2Top[0]-1])){
            //printf("ENTERED IF\n");
            b = ddfsMove(bud,ddfs.offsets,ddfs.cols,ddfs.ddfsPredecessorsPtr,ddfs.removed,ddfs.predecessors,ddfs.stack1,
                          ddfs.stack2,ddfs.stack1Top,ddfs.stack2Top,ddfs.support,ddfs.supportTop,ddfs.color,ddfs.globalColorCounter,ddfs.budAtDDFSEncounter,newRed, newGreen);
            count[0]++;
        } else{
            //printf("ENTERED ELSE\n");
            b = ddfsMove(bud,ddfs.offsets,ddfs.cols,ddfs.ddfsPredecessorsPtr,ddfs.removed,ddfs.predecessors,ddfs.stack2,
                          ddfs.stack1,ddfs.stack2Top,ddfs.stack1Top,ddfs.support,ddfs.supportTop,ddfs.color,ddfs.globalColorCounter,ddfs.budAtDDFSEncounter,newGreen, newRed);
            count[0]++;
        }
        if(b != -1){
            //return {b,b};
            //printf("B!=-1\n");
            //return (uint64_t) b << 32 | b;
            ddfs.ddfsResult[0]=b;
            ddfs.ddfsResult[1]=b;
            ddfs.curBridge[0] = src;
            ddfs.curBridge[1] = dst;
            ddfs.curBridge[2] = bud[src];
            ddfs.curBridge[3] = bud[dst];
            count[0]++;
            return;
        }
    }
}


__device__ void ddfsStructDev(DSU_CU bud,
                              ddfsStruct ddfs,
                              int src, int dst,
                              unsigned int * count,
                              int stackIndex) {
    if (ddfs.removed[bud[src]] || ddfs.removed[bud[dst]])
      return;
    ddfs.stack1Top[stackIndex]=0;
    ddfs.stack2Top[stackIndex]=0;
    ddfs.supportTop[stackIndex]=0;
    int ListOffset = stackIndex*ddfs.maxStackDepth[0];
    int DDFSResultOffset = stackIndex*2;
    int CurrBridgeOffset = stackIndex*4;

    auto minlvl = [&](int u)
    { return min(ddfs.evenlvl[u], ddfs.oddlvl[u]); };
    
    ddfs.stack1[ListOffset+ddfs.stack1Top[stackIndex]++]=bud[src];
    ddfs.stack2[ListOffset+ddfs.stack2Top[stackIndex]++]=bud[dst];
    //vector<int> Sr = {bud[src]}, Sg = {bud[dst]};
    //if(Sr[0] == Sg[0])
    //    return {Sr[0],Sg[0]};
    if (ddfs.stack1[ListOffset]==ddfs.stack2[ListOffset]){
        //printf("stack1[0]=%d == stack2[0]=%d\n",stack1[0],stack2[0]);
        //return (uint64_t) stack1[0] << 32 | stack2[0];
        ddfs.ddfsResult[DDFSResultOffset+0]=ddfs.stack1[ListOffset];
        ddfs.ddfsResult[DDFSResultOffset+1]=ddfs.stack2[ListOffset];
        ddfs.curBridge[CurrBridgeOffset+0] = src;
        ddfs.curBridge[CurrBridgeOffset+1] = dst;
        ddfs.curBridge[CurrBridgeOffset+2] = bud[src];
        ddfs.curBridge[CurrBridgeOffset+3] = bud[dst];
        count[0]++;
        return;

    }
    //out_support = {Sr[0], Sg[0]};
    ddfs.support[ListOffset+ddfs.supportTop[stackIndex]++]=ddfs.stack1[ListOffset];
    ddfs.support[ListOffset+ddfs.supportTop[stackIndex]++]=ddfs.stack2[ListOffset];


    int localGlobalColorCounter = ddfs.globalColorCounter[0];
    ddfs.globalColorCounter[0]=localGlobalColorCounter+2;
    //int newRed = ddfs.color[ddfs.stack1[ListOffset]] = localGlobalColorCounter-1, newGreen = ddfs.color[ddfs.stack2[ListOffset]] = localGlobalColorCounter;
    int newRed = ddfs.color[ddfs.stack1[ListOffset]] = ++localGlobalColorCounter, newGreen = ddfs.color[ddfs.stack2[ListOffset]] = ++localGlobalColorCounter;

    assert(newRed == (newGreen^1));
    

    for(;;) {
        //printf("IN FOR\n");
        //if found two disjoint paths
        //if(minlvl(Sr.back()) == 0 && minlvl(Sg.back()) == 0)
        if(minlvl(ddfs.stack1[ListOffset+ddfs.stack1Top[stackIndex]-1]) == 0 && minlvl(ddfs.stack2[ListOffset+ddfs.stack2Top[stackIndex]-1]) == 0){

            //printf("stack1[%d]=%d\n",stack1Top[0]-1,stack1[stack1Top[0]-1]);
            //printf("stack2[%d]=%d\n",stack2Top[0]-1,stack2[stack2Top[0]-1]);
            //printf("minlvl(graph,stack1[stack1Top[0]]) == 0 && minlvl(graph,stack2[stack2Top[0]]) == 0\n");
            ddfs.ddfsResult[DDFSResultOffset+0]=ddfs.stack1[ListOffset+ddfs.stack1Top[stackIndex]-1];
            ddfs.ddfsResult[DDFSResultOffset+1]=ddfs.stack2[ListOffset+ddfs.stack2Top[stackIndex]-1];
            ddfs.curBridge[CurrBridgeOffset+0] = src;
            ddfs.curBridge[CurrBridgeOffset+1] = dst;
            ddfs.curBridge[CurrBridgeOffset+2] = bud[src];
            ddfs.curBridge[CurrBridgeOffset+3] = bud[dst];
            count[0]++;
            return;
        }
        int b;
        //if(minlvl(Sr.back()) >= minlvl(Sg.back()))
        //printf("stack1[%d-1]=%d ml %d stack2[%d-1]=%d ml %d \n",stack1Top[0],stack1[stack1Top[0]-1],minlvl(stack1[stack1Top[0]-1]),stack2Top[0],stack2[stack2Top[0]-1],minlvl(stack2[stack2Top[0]-1]));
        if(minlvl(ddfs.stack1[ListOffset+ddfs.stack1Top[stackIndex]-1]) >= minlvl(ddfs.stack2[ListOffset+ddfs.stack2Top[stackIndex]-1])){
            //printf("ENTERED IF\n");
            b = ddfsMove(bud,ddfs.offsets,ddfs.cols,ddfs.ddfsPredecessorsPtr,ddfs.removed,ddfs.predecessors,&ddfs.stack1[ListOffset],
                          &ddfs.stack2[ListOffset],&ddfs.stack1Top[stackIndex],&ddfs.stack2Top[stackIndex],&ddfs.support[ListOffset],&ddfs.supportTop[stackIndex],ddfs.color,ddfs.globalColorCounter,ddfs.budAtDDFSEncounter,newRed, newGreen);
            count[0]++;
        } else{
            //printf("ENTERED ELSE\n");
            b = ddfsMove(bud,ddfs.offsets,ddfs.cols,ddfs.ddfsPredecessorsPtr,ddfs.removed,ddfs.predecessors,&ddfs.stack2[ListOffset],
                          &ddfs.stack1[ListOffset],&ddfs.stack2Top[stackIndex],&ddfs.stack1Top[stackIndex],&ddfs.support[ListOffset],&ddfs.supportTop[stackIndex],ddfs.color,ddfs.globalColorCounter,ddfs.budAtDDFSEncounter,newGreen, newRed);
            count[0]++;
        }
        if(b != -1){
            //return {b,b};
            //printf("B!=-1\n");
            //return (uint64_t) b << 32 | b;
            ddfs.ddfsResult[DDFSResultOffset+0]=b;
            ddfs.ddfsResult[DDFSResultOffset+1]=b;
            ddfs.curBridge[CurrBridgeOffset+0] = src;
            ddfs.curBridge[CurrBridgeOffset+1] = dst;
            ddfs.curBridge[CurrBridgeOffset+2] = bud[src];
            ddfs.curBridge[CurrBridgeOffset+3] = bud[dst];
            count[0]++;
            return;
        }
    }
}
#endif