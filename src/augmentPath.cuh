#ifndef AUGMENTPATH
#define AUGMENTPATH
#include "CSRGraph.cuh"
#include "argStructs.cuh"
#include "augmentPathHelpers.cuh"


__device__ void augumentPathIterativeSwitchPassStructDev(DSU_CU bud,
                                                        APStruct ap){
    //printf("Entered augumentPathIterativeSwitch\n");
    /*
    __shared__ int uStack[MAXSTACKAUGPATH];
    __shared__ int vStack[MAXSTACKAUGPATH];
    __shared__ int bStack[MAXSTACKAUGPATH];
    __shared__ int origCurStack[MAXSTACKAUGPATH];
    */
    
    extern __shared__ int shared_mem[];
    int * uStack = &shared_mem[0*ap.stackDepth[0]];
    int * vStack = &shared_mem[1*ap.stackDepth[0]];
    int * bStack = &shared_mem[2*ap.stackDepth[0]];
    int * origCurStack = &shared_mem[3*ap.stackDepth[0]];
    
    //__shared__ int origBCurrStack[MAXSTACKAUGPATH];
    int * origBCurrStack;
    int *stateStack;
    //__shared__ int stateStack[MAXSTACKAUGPATH];

    int u = ap.ddfsResult[0];
    int v = ap.ddfsResult[1];
    if (u == v) return;
    int stackTop = 0;
    int thisU = u;
    int thisV = v;
    int thisB = -1;
    int thisOrigCur = -1;
    int thisOrigBcur = -1;
    int success = false;
    bool initial = true;
    //printf("Start state %d u %d v %d b %d stackTop %d\n", 
    //        0, thisU, thisV, thisB, stackTop);
    int thisState = 0;
    pushStackVars(uStack, vStack, bStack, origCurStack, origBCurrStack, stateStack, &stackTop,&thisU, &thisV, &thisB, &thisOrigCur, &thisOrigBcur, &thisState);
    while(stackTop>0){
        popStackVars(uStack, vStack, bStack, origCurStack, origBCurrStack,stateStack, &stackTop,&thisU, &thisV, &thisB, &thisOrigCur, &thisOrigBcur, &thisState);
        
        int tentativeState;
        if (thisB == -2 )
            tentativeState = 3;
        else if (thisB == -3 )
            tentativeState = 4;
        else if (thisB == -4 )
            tentativeState = 5;
        else if (thisB == -1 && thisOrigCur == -1)
            tentativeState = 0;
        else if (thisB != -1 && thisOrigCur == -1)
            tentativeState = 1;
        else if (thisB != -1 && thisOrigCur != -1)
            tentativeState = 2;
        else
            assert(false);
        //if (thisState!=tentativeState)
        //    printf("%d %d\n",thisState,tentativeState);
        //assert(thisState==tentativeState);
        thisState = tentativeState;
        switch(thisState) {
            case 0 :
                //printf("Entered case 0\n");
                augumentPathSubroutine(bud,
                                        ap.offsets,
                                        ap.cols,
                                        ap.oddlvl,
                                        ap.evenlvl,
                                        ap.pred,
                                        ap.removed,
                                        ap.mate,
                                        ap.myBridge_a,
                                        ap.myBridge_b,
                                        ap.myBridge_c,
                                        ap.myBridge_d, 
                                        ap.color, 
                                        ap.removedVerticesQueue, 
                                        ap.removedVerticesQueueBack, 
                                        ap.budAtDDFSEncounter, 
                                        thisU, 
                                        thisV, 
                                        initial,
                                        uStack,
                                        vStack,
                                        bStack,
                                        origCurStack,
                                        origBCurrStack,
                                        stateStack,
                                        &stackTop);
                initial=false;
                break; //optional
            case 1 :
                //printf("Entered case 1\n");
                openingDfsSubroutineCall(
                                    ap.offsets,
                                    ap.cols,
                                    ap.oddlvl,
                                    ap.evenlvl,
                                    ap.pred,
                                    ap.color,
                                    ap.budAtDDFSEncounter, 
                                    ap.childsInDDFSTreePtr,
                                    thisU, 
                                    thisV,
                                    thisB, 
                                    uStack,
                                    vStack,
                                    bStack,
                                    origCurStack,
                                    origBCurrStack,
                                    stateStack,
                                    &stackTop);
                break; //optional
            case 2 :
                //printf("Entered case 2\n");
                openingDfsSubroutineCheck( 
                                    ap.childsInDDFSTreePtr,
                                    thisU, 
                                    thisV,
                                    thisB, 
                                    thisOrigCur,
                                    //thisOrigBcur,
                                    success,
                                    uStack,
                                    vStack,
                                    bStack,
                                    origCurStack,
                                    origBCurrStack,
                                    stateStack,
                                    &stackTop);

                break; //optional
            case 3 :
                //printf("Entered case 3\n");
                flip(ap.removed,ap.mate,ap.removedVerticesQueue,ap.removedVerticesQueueBack,thisU,thisV);
                break; //optional
            case 4 :
                //printf("Entered case 4\n");
                success = true;
                break; //optional
            case 5 :
                //printf("Entered case 5\n");
                success = false;
                break; //optional
            // you can have any number of case statements.
            default : //Optional
                break;
        }
    }

    auto minlvl = [&](int u)
    { return min(ap.evenlvl[u], ap.oddlvl[u]); };
    char Prop = 1;
    // Remove vertices.
    ap.foundPath[0] = true;
    //printf("removedVerticesQueueBack %d removedVerticesQueueFront %d\n",removedVerticesQueueBack[0],removedVerticesQueueFront[0]);
    while (ap.removedVerticesQueueBack[0]-ap.removedVerticesQueueFront[0])
    {
        //printf("removedVerticesQueueBack %d removedVerticesQueueFront %d\n",removedVerticesQueueBack[0],removedVerticesQueueFront[0]);
        int v = ap.removedVerticesQueue[ap.removedVerticesQueueFront[0]];
        ap.removedVerticesQueueFront[0]++;
        unsigned int start = ap.offsets[v];
        unsigned int end = ap.offsets[v + 1];
        unsigned int edgeIndex = start;
        edgeIndex = start;
        for(; edgeIndex < end; edgeIndex++) {
            if (ap.edgeStatus[edgeIndex] == Prop && minlvl(ap.cols[edgeIndex]) > minlvl(v) && 
                !ap.removed[ap.cols[edgeIndex]] && ++ap.removedPredecessorsSize[ap.cols[edgeIndex]] == ap.predecessor_count[ap.cols[edgeIndex]]){
                    //printf("removing %d post augpath\n",cols[edgeIndex]);
                    removeAndPushToQueue(ap.removed,ap.removedVerticesQueue,ap.removedVerticesQueueBack,ap.cols[edgeIndex]);
                }
        }
    }
    ap.removedVerticesQueueBack[0]=0;
    ap.removedVerticesQueueFront[0]=0;
    return;
}


__device__ void augumentPathIterativeSwitchPassStructDev(DSU_CU bud,
                                                        APStruct ap,
                                                        int stackIndex){

    int ListOffset = stackIndex*ap.maxStackDepth[0];
    int DDFSResultOffset = stackIndex*2;
    int CurrBridgeOffset = stackIndex*4;

    extern __shared__ int shared_mem[];
    int * uStack = &shared_mem[0*ap.stackDepth[0]];
    int * vStack = &shared_mem[1*ap.stackDepth[0]];
    int * bStack = &shared_mem[2*ap.stackDepth[0]];
    int * origCurStack = &shared_mem[3*ap.stackDepth[0]];
    
    //__shared__ int origBCurrStack[MAXSTACKAUGPATH];
    int * origBCurrStack;
    int *stateStack;
    //__shared__ int stateStack[MAXSTACKAUGPATH];

    int u = ap.ddfsResult[DDFSResultOffset+0];
    int v = ap.ddfsResult[DDFSResultOffset+1];
    if (u == v) return;
    int stackTop = 0;
    int thisU = u;
    int thisV = v;
    int thisB = -1;
    int thisOrigCur = -1;
    int thisOrigBcur = -1;
    int success = false;
    bool initial = true;
    //printf("Start state %d u %d v %d b %d stackTop %d\n", 
    //        0, thisU, thisV, thisB, stackTop);
    int thisState = 0;
    pushStackVars(uStack, vStack, bStack, origCurStack, origBCurrStack, stateStack, &stackTop,&thisU, &thisV, &thisB, &thisOrigCur, &thisOrigBcur, &thisState);
    while(stackTop>0){
        popStackVars(uStack, vStack, bStack, origCurStack, origBCurrStack,stateStack, &stackTop,&thisU, &thisV, &thisB, &thisOrigCur, &thisOrigBcur, &thisState);
        
        int tentativeState;
        if (thisB == -2 )
            tentativeState = 3;
        else if (thisB == -3 )
            tentativeState = 4;
        else if (thisB == -4 )
            tentativeState = 5;
        else if (thisB == -1 && thisOrigCur == -1)
            tentativeState = 0;
        else if (thisB != -1 && thisOrigCur == -1)
            tentativeState = 1;
        else if (thisB != -1 && thisOrigCur != -1)
            tentativeState = 2;
        else
            assert(false);
        //if (thisState!=tentativeState)
        //    printf("%d %d\n",thisState,tentativeState);
        //assert(thisState==tentativeState);
        thisState = tentativeState;
        switch(thisState) {
            case 0 :
                //printf("Entered case 0\n");
                augumentPathSubroutine(bud,
                                        ap.offsets,
                                        ap.cols,
                                        ap.oddlvl,
                                        ap.evenlvl,
                                        ap.pred,
                                        ap.removed,
                                        ap.mate,
                                        ap.myBridge_a,
                                        ap.myBridge_b,
                                        ap.myBridge_c,
                                        ap.myBridge_d, 
                                        ap.color, 
                                        &ap.removedVerticesQueue[ListOffset], 
                                        &ap.removedVerticesQueueBack[stackIndex], 
                                        ap.budAtDDFSEncounter, 
                                        thisU, 
                                        thisV, 
                                        initial,
                                        uStack,
                                        vStack,
                                        bStack,
                                        origCurStack,
                                        origBCurrStack,
                                        stateStack,
                                        &stackTop);
                initial=false;
                break; //optional
            case 1 :
                //printf("Entered case 1\n");
                openingDfsSubroutineCall(
                                    ap.offsets,
                                    ap.cols,
                                    ap.oddlvl,
                                    ap.evenlvl,
                                    ap.pred,
                                    ap.color,
                                    ap.budAtDDFSEncounter, 
                                    ap.childsInDDFSTreePtr,
                                    thisU, 
                                    thisV,
                                    thisB, 
                                    uStack,
                                    vStack,
                                    bStack,
                                    origCurStack,
                                    origBCurrStack,
                                    stateStack,
                                    &stackTop);
                break; //optional
            case 2 :
                //printf("Entered case 2\n");
                openingDfsSubroutineCheck( 
                                    ap.childsInDDFSTreePtr,
                                    thisU, 
                                    thisV,
                                    thisB, 
                                    thisOrigCur,
                                    //thisOrigBcur,
                                    success,
                                    uStack,
                                    vStack,
                                    bStack,
                                    origCurStack,
                                    origBCurrStack,
                                    stateStack,
                                    &stackTop);

                break; //optional
            case 3 :
                //printf("Entered case 3\n");
                flip(ap.removed,ap.mate,&ap.removedVerticesQueue[ListOffset],&ap.removedVerticesQueueBack[stackIndex],thisU,thisV);
                break; //optional
            case 4 :
                //printf("Entered case 4\n");
                success = true;
                break; //optional
            case 5 :
                //printf("Entered case 5\n");
                success = false;
                break; //optional
            // you can have any number of case statements.
            default : //Optional
                break;
        }
    }

    auto minlvl = [&](int u)
    { return min(ap.evenlvl[u], ap.oddlvl[u]); };
    char Prop = 1;
    // Remove vertices.
    ap.foundPath[0] = true;
    //printf("removedVerticesQueueBack %d removedVerticesQueueFront %d\n",removedVerticesQueueBack[0],removedVerticesQueueFront[0]);
    while (ap.removedVerticesQueueBack[stackIndex]-ap.removedVerticesQueueFront[stackIndex])
    {
        //printf("removedVerticesQueueBack %d removedVerticesQueueFront %d\n",removedVerticesQueueBack[0],removedVerticesQueueFront[0]);
        int v = ap.removedVerticesQueue[ListOffset+ap.removedVerticesQueueFront[stackIndex]];
        ap.removedVerticesQueueFront[stackIndex]++;
        unsigned int start = ap.offsets[v];
        unsigned int end = ap.offsets[v + 1];
        unsigned int edgeIndex = start;
        edgeIndex = start;
        for(; edgeIndex < end; edgeIndex++) {
            if (ap.edgeStatus[edgeIndex] == Prop && minlvl(ap.cols[edgeIndex]) > minlvl(v) && 
                !ap.removed[ap.cols[edgeIndex]] && ++ap.removedPredecessorsSize[ap.cols[edgeIndex]] == ap.predecessor_count[ap.cols[edgeIndex]]){
                    //printf("removing %d post augpath\n",cols[edgeIndex]);
                    removeAndPushToQueue(ap.removed,&ap.removedVerticesQueue[ListOffset],&ap.removedVerticesQueueBack[stackIndex],ap.cols[edgeIndex]);
                }
        }
    }
    ap.removedVerticesQueueBack[stackIndex]=0;
    ap.removedVerticesQueueFront[stackIndex]=0;
    return;
}





__global__ void augumentPathIterativeSwitchPassStructGlobal(DSU_CU bud,
                                                        APStruct ap,
                                                        bool * removed,
                                                        int src,
                                                        int dst){

    if (removed[bud[src]] || removed[bud[dst]])
      return;                                                           
    //printf("Entered augumentPathIterativeSwitch\n");
    /*
    __shared__ int uStack[MAXSTACKAUGPATH];
    __shared__ int vStack[MAXSTACKAUGPATH];
    __shared__ int bStack[MAXSTACKAUGPATH];
    __shared__ int origCurStack[MAXSTACKAUGPATH];
    */
    // Cant get dym SM to work.
    
    extern __shared__ int shared_mem[];
    int * uStack = &shared_mem[0*ap.stackDepth[0]];
    int * vStack = &shared_mem[1*ap.stackDepth[0]];
    int * bStack = &shared_mem[2*ap.stackDepth[0]];
    int * origCurStack = &shared_mem[3*ap.stackDepth[0]];
    

    //__shared__ int origBCurrStack[MAXSTACKAUGPATH];
    int * origBCurrStack;
    int *stateStack;
    //__shared__ int stateStack[MAXSTACKAUGPATH];

    int u = ap.ddfsResult[0];
    int v = ap.ddfsResult[1];
    if (u == v) return;
    int stackTop = 0;
    int thisU = u;
    int thisV = v;
    int thisB = -1;
    int thisOrigCur = -1;
    int thisOrigBcur = -1;
    int success = false;
    bool initial = true;
    //printf("Start state %d u %d v %d b %d stackTop %d\n", 
    //        0, thisU, thisV, thisB, stackTop);
    int thisState = 0;
    pushStackVars(uStack, vStack, bStack, origCurStack, origBCurrStack, stateStack, &stackTop,&thisU, &thisV, &thisB, &thisOrigCur, &thisOrigBcur, &thisState);
    while(stackTop>0){
        popStackVars(uStack, vStack, bStack, origCurStack, origBCurrStack,stateStack, &stackTop,&thisU, &thisV, &thisB, &thisOrigCur, &thisOrigBcur, &thisState);
        
        int tentativeState;
        if (thisB == -2 )
            tentativeState = 3;
        else if (thisB == -3 )
            tentativeState = 4;
        else if (thisB == -4 )
            tentativeState = 5;
        else if (thisB == -1 && thisOrigCur == -1)
            tentativeState = 0;
        else if (thisB != -1 && thisOrigCur == -1)
            tentativeState = 1;
        else if (thisB != -1 && thisOrigCur != -1)
            tentativeState = 2;
        else
            assert(false);
        //if (thisState!=tentativeState)
        //    printf("%d %d\n",thisState,tentativeState);
        //assert(thisState==tentativeState);
        thisState = tentativeState;
        switch(thisState) {
            case 0 :
                //printf("Entered case 0\n");
                augumentPathSubroutine(bud,
                                        ap.offsets,
                                        ap.cols,
                                        ap.oddlvl,
                                        ap.evenlvl,
                                        ap.pred,
                                        ap.removed,
                                        ap.mate,
                                        ap.myBridge_a,
                                        ap.myBridge_b,
                                        ap.myBridge_c,
                                        ap.myBridge_d, 
                                        ap.color, 
                                        ap.removedVerticesQueue, 
                                        ap.removedVerticesQueueBack, 
                                        ap.budAtDDFSEncounter, 
                                        thisU, 
                                        thisV, 
                                        initial,
                                        uStack,
                                        vStack,
                                        bStack,
                                        origCurStack,
                                        origBCurrStack,
                                        stateStack,
                                        &stackTop);
                initial=false;
                break; //optional
            case 1 :
                //printf("Entered case 1\n");
                openingDfsSubroutineCall(
                                    ap.offsets,
                                    ap.cols,
                                    ap.oddlvl,
                                    ap.evenlvl,
                                    ap.pred,
                                    ap.color,
                                    ap.budAtDDFSEncounter, 
                                    ap.childsInDDFSTreePtr,
                                    thisU, 
                                    thisV,
                                    thisB, 
                                    uStack,
                                    vStack,
                                    bStack,
                                    origCurStack,
                                    origBCurrStack,
                                    stateStack,
                                    &stackTop);
                break; //optional
            case 2 :
                //printf("Entered case 2\n");
                openingDfsSubroutineCheck( 
                                    ap.childsInDDFSTreePtr,
                                    thisU, 
                                    thisV,
                                    thisB, 
                                    thisOrigCur,
                                    //thisOrigBcur,
                                    success,
                                    uStack,
                                    vStack,
                                    bStack,
                                    origCurStack,
                                    origBCurrStack,
                                    stateStack,
                                    &stackTop);

                break; //optional
            case 3 :
                //printf("Entered case 3\n");
                flip(ap.removed,ap.mate,ap.removedVerticesQueue,ap.removedVerticesQueueBack,thisU,thisV);
                break; //optional
            case 4 :
                //printf("Entered case 4\n");
                success = true;
                break; //optional
            case 5 :
                //printf("Entered case 5\n");
                success = false;
                break; //optional
            // you can have any number of case statements.
            default : //Optional
                break;
        }
    }

    auto minlvl = [&](int u)
    { return min(ap.evenlvl[u], ap.oddlvl[u]); };
    char Prop = 1;
    // Remove vertices.
    ap.foundPath[0] = true;
    //printf("removedVerticesQueueBack %d removedVerticesQueueFront %d\n",removedVerticesQueueBack[0],removedVerticesQueueFront[0]);
    while (ap.removedVerticesQueueBack[0]-ap.removedVerticesQueueFront[0])
    {
        //printf("removedVerticesQueueBack %d removedVerticesQueueFront %d\n",removedVerticesQueueBack[0],removedVerticesQueueFront[0]);
        int v = ap.removedVerticesQueue[ap.removedVerticesQueueFront[0]];
        ap.removedVerticesQueueFront[0]++;
        unsigned int start = ap.offsets[v];
        unsigned int end = ap.offsets[v + 1];
        unsigned int edgeIndex = start;
        edgeIndex = start;
        for(; edgeIndex < end; edgeIndex++) {
            if (ap.edgeStatus[edgeIndex] == Prop && minlvl(ap.cols[edgeIndex]) > minlvl(v) && 
                !ap.removed[ap.cols[edgeIndex]] && ++ap.removedPredecessorsSize[ap.cols[edgeIndex]] == ap.predecessor_count[ap.cols[edgeIndex]]){
                    //printf("removing %d post augpath\n",cols[edgeIndex]);
                    removeAndPushToQueue(ap.removed,ap.removedVerticesQueue,ap.removedVerticesQueueBack,ap.cols[edgeIndex]);
                }
        }
    }
    ap.removedVerticesQueueBack[0]=0;
    ap.removedVerticesQueueFront[0]=0;
    return;
}


__device__ void augumentPathIterativeSwitchDev(DSU_CU bud,
                                        unsigned int * offsets,
                                        unsigned int * cols,
                                        int * oddlvl,
                                        int * evenlvl,
                                        char * edgeStatus,
                                        bool * pred,
                                        unsigned int *childsInDDFSTreePtr,
                                        bool *removed,
                                        int * myBridge_a, 
                                        int * myBridge_b, 
                                        int * myBridge_c, 
                                        int * myBridge_d, 
                                        unsigned int * color,
                                        unsigned int * removedPredecessorsSize, 
                                        unsigned int * predecessor_count,
                                        int * removedVerticesQueue, 
                                        unsigned int * removedVerticesQueueFront, 
                                        unsigned int * removedVerticesQueueBack, 
                                        bool * foundPath,
                                        int * budAtDDFSEncounter, 
                                        int * mate,
                                        int u, int v, bool _initial){
    //printf("Entered augumentPathIterativeSwitch\n");
    __shared__ int uStack[MAXSTACKAUGPATH];
    __shared__ int vStack[MAXSTACKAUGPATH];
    __shared__ int bStack[MAXSTACKAUGPATH];
    __shared__ int origCurStack[MAXSTACKAUGPATH];
    __shared__ int origBCurrStack[MAXSTACKAUGPATH];
    __shared__ int stateStack[MAXSTACKAUGPATH];
    int stackTop = 0;
    int thisU = u;
    int thisV = v;
    int thisB = -1;
    int thisOrigCur = -1;
    int thisOrigBcur = -1;
    int success = false;
    bool initial = true;
    //printf("Start state %d u %d v %d b %d stackTop %d\n", 
    //        0, thisU, thisV, thisB, stackTop);
    int thisState = 0;
    pushStackVars(uStack, vStack, bStack, origCurStack, origBCurrStack, stateStack, &stackTop,&thisU, &thisV, &thisB, &thisOrigCur, &thisOrigBcur, &thisState);
    while(stackTop>0){
        popStackVars(uStack, vStack, bStack, origCurStack, origBCurrStack,stateStack, &stackTop,&thisU, &thisV, &thisB, &thisOrigCur, &thisOrigBcur, &thisState);
        switch(thisState) {
            case 0 :
                //printf("Entered case 0\n");
                augumentPathSubroutine(bud,
                                        offsets,
                                        cols,
                                        oddlvl,
                                        evenlvl,
                                        pred,
                                        removed,
                                        mate,
                                        myBridge_a,
                                        myBridge_b,
                                        myBridge_c,
                                        myBridge_d, 
                                        color, 
                                        removedVerticesQueue, 
                                        removedVerticesQueueBack, 
                                        budAtDDFSEncounter, 
                                        thisU, 
                                        thisV, 
                                        initial,
                                        uStack,
                                        vStack,
                                        bStack,
                                        origCurStack,
                                        origBCurrStack,
                                        stateStack,
                                        &stackTop);
                initial=false;
                break; //optional
            case 1 :
                //printf("Entered case 1\n");
                openingDfsSubroutineCall(
                                    offsets,
                                    cols,
                                    oddlvl,
                                    evenlvl,
                                    pred,
                                    color,
                                    budAtDDFSEncounter, 
                                    childsInDDFSTreePtr,
                                    thisU, 
                                    thisV,
                                    thisB, 
                                    uStack,
                                    vStack,
                                    bStack,
                                    origCurStack,
                                    origBCurrStack,
                                    stateStack,
                                    &stackTop);
                break; //optional
            case 2 :
                //printf("Entered case 2\n");
                openingDfsSubroutineCheck( 
                                    childsInDDFSTreePtr,
                                    thisU, 
                                    thisV, // thisOrigBcur passed here
                                    thisB, 
                                    thisOrigCur,
                                    //thisOrigBcur,
                                    success,
                                    uStack,
                                    vStack,
                                    bStack,
                                    origCurStack,
                                    origBCurrStack,
                                    stateStack,
                                    &stackTop);

                break; //optional
            case 3 :
                //printf("Entered case 3\n");
                flip(removed,mate,removedVerticesQueue,removedVerticesQueueBack,thisU,thisV);
                break; //optional
            case 4 :
                //printf("Entered case 4\n");
                childsInDDFSTreePtr[thisOrigBcur]=0;
                success = true;
                break; //optional
            case 5 :
                //printf("Entered case 5\n");
                childsInDDFSTreePtr[thisOrigBcur]=0;
                success = false;
                break; //optional
            // you can have any number of case statements.
            default : //Optional
                break;
        }
    }

    auto minlvl = [&](int u)
    { return min(evenlvl[u], oddlvl[u]); };
    char Prop = 1;
    // Remove vertices.
    foundPath[0] = true;
    //printf("removedVerticesQueueBack %d removedVerticesQueueFront %d\n",removedVerticesQueueBack[0],removedVerticesQueueFront[0]);
    while (removedVerticesQueueBack[0]-removedVerticesQueueFront[0])
    {
        //printf("removedVerticesQueueBack %d removedVerticesQueueFront %d\n",removedVerticesQueueBack[0],removedVerticesQueueFront[0]);
        int v = removedVerticesQueue[removedVerticesQueueFront[0]];
        removedVerticesQueueFront[0]++;
        unsigned int start = offsets[v];
        unsigned int end = offsets[v + 1];
        unsigned int edgeIndex = start;
        edgeIndex = start;
        for(; edgeIndex < end; edgeIndex++) {
            if (edgeStatus[edgeIndex] == Prop && minlvl(cols[edgeIndex]) > minlvl(v) && 
                !removed[cols[edgeIndex]] && ++removedPredecessorsSize[cols[edgeIndex]] == predecessor_count[cols[edgeIndex]]){
                    //printf("removing %d post augpath\n",cols[edgeIndex]);
                    removeAndPushToQueue(removed,removedVerticesQueue,removedVerticesQueueBack,cols[edgeIndex]);
                }
        }
    }
    removedVerticesQueueBack[0]=0;
    removedVerticesQueueFront[0]=0;
    return;
}

#endif