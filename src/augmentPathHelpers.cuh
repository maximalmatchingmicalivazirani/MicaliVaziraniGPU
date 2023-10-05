#ifndef AUGMENTPATHHELPERS
#define AUGMENTPATHHELPERS


//#define MAXSTACKAUGPATH 1500
#define MAXSTACKAUGPATH 2048
__constant__ int MAXSTACKAUGPATH_d;

template <typename T> __device__ void inline swapP(T& a, T& b)
{
    T c(a); a=b; b=c;
}

__device__ void removeAndPushToQueue(bool * removed,int * removedVerticesQueue, unsigned int * removedVerticesQueueBack, int u) {  
  //printf("gpu remove %d\n",u);
 removed[u] = true; removedVerticesQueue[removedVerticesQueueBack[0]++]=u;}

__device__ void flip(bool * removed, int * matching, int * removedVerticesQueue, unsigned int * removedVerticesQueueBack, int u, int v) {
    if(removed[u] || removed[v] || matching[u] == v) return;//flipping only unmatched edges
    removeAndPushToQueue(removed,removedVerticesQueue,removedVerticesQueueBack,u);
    removeAndPushToQueue(removed,removedVerticesQueue,removedVerticesQueueBack,v);
    matching[u] = v;
    matching[v] = u;
}

__device__ void popStackVars(int * uStack,
                            int * vStack,
                            int * bStack,
                            int * origCurrStack,
                            int * origBCurrStack,
                            int * stateStack,
                            int * stackTop,
                            int * thisU,
                            int * thisV,
                            int * thisB,
                            int * thisOrigCurr,
                            int * thisOrigBCurr,
                            int * thisState){
    stackTop[0]--;
    *thisU=uStack[*stackTop];
    *thisV=vStack[*stackTop];
    *thisB=bStack[*stackTop];
    *thisOrigCurr=origCurrStack[*stackTop];
    *thisOrigBCurr=origBCurrStack[*stackTop];
    //*thisState=stateStack[*stackTop];
    //printf("popped state %d stacktop %d",*thisState,stackTop[0]);
}

__device__ void pushStackVars(int * stack1,
                            int * stack2,
                            int * stack3,
                            int * stack4,
                            int * stack5,
                            int * stateStack,
                            int * stackTop,
                            int * stack1Val,
                            int * stack2Val,
                            int * stack3Val,
                            int * stack4Val,
                            int * stack5Val,
                            int * thisState){
    assert(stackTop[0]<MAXSTACKAUGPATH_d);
    stack1[*stackTop] = *stack1Val;
    stack2[*stackTop] = *stack2Val;
    stack3[*stackTop] = *stack3Val;
    stack4[*stackTop] = *stack4Val;
    //stack5[*stackTop] = *stack5Val;
    //stateStack[*stackTop] = *thisState;
    stackTop[0]++;
    //printf("pushed stack top %d state%d\n",stackTop[0],thisState[0]);
}

/*
Uses:
        u -- uses cur stack
        v -- uses bcur stack
        cur - not used
        bcur - not used
        b - not used
        origCur - not used
        origBCur - not used
*/

__device__ void augumentPathSubroutine(DSU_CU bud,
                                        unsigned int * offsets,
                                        unsigned int * cols,
                                        int * oddlvl,
                                        int * evenlvl,
                                        bool * pred,
                                        bool * removed,
                                        int * mate,
                                        int * myBridge_a, 
                                        int * myBridge_b, 
                                        int * myBridge_c, 
                                        int * myBridge_d, 
                                        unsigned int * color, 
                                        int * removedVerticesQueue, 
                                        unsigned int * removedVerticesQueueBack, 
                                        int * budAtDDFSEncounter, 
                                        int u, 
                                        int v, 
                                        bool initial,
                                        int * uStack,
                                        int * vStack,
                                        int * bStack,
                                        int * origCurStack,
                                        int * origBCurStack,
                                        int * stateStack,
                                        int * stackTop){
    //printf("augumentPath %d %d\n", u, v);
    auto minlvl = [&](int u)
    { return min(evenlvl[u], oddlvl[u]); };
    if(u == v) return;
    if(!initial && minlvl(u) == evenlvl[u]) { //simply follow predecessors
        // TMP
        unsigned int start = offsets[u];
        unsigned int end = offsets[u + 1];
        unsigned int edgeIndex;
        //  TODO USE PREDSIZE ARRAY!!
        int predSize = 0;
        for(edgeIndex=start; edgeIndex < end; edgeIndex++) {
            if (pred[edgeIndex])
                predSize++;
        }
        assert(predSize == 1); //u should be evenlevel (last minlevel edge is matched, so there is only one predecessor)
        // First predecessor of u
        int x = -1; //no need to flip edge since we know it's matched
        for(edgeIndex=start; edgeIndex < end; edgeIndex++) {
            if (pred[edgeIndex]){
                x = cols[edgeIndex];
                break;
            }
        }
        assert(x > -1);
        start = offsets[x];
        end = offsets[x + 1];
        int newU = -1;
        for(edgeIndex=start; edgeIndex < end; edgeIndex++) {
            if (pred[edgeIndex] && bud[cols[edgeIndex]] == bud[x]){
                newU = cols[edgeIndex];
                break;
            }
        }
        assert(newU > -1);
        u = newU;
        assert(!removed[u]);
        flip(removed,mate,removedVerticesQueue,removedVerticesQueueBack,x,u);
        int thisState = 0;
        int thisU = u;
        int thisV = v;
        int thisB = -1;
        int thisOrigCur = -1;
        int thisOrigBCur = -1;
        pushStackVars(uStack, vStack, bStack, origCurStack, origBCurStack, stateStack, stackTop,&thisU, &thisV, &thisB, &thisOrigCur, &thisOrigBCur, &thisState);
        //augumentPath(graph,dfsWL,color,removedVerticesQueue,removedVerticesQueueBack,budAtDDFSEncounter, u,v);
        // Push state 1
    }
    else { //through bridge
        // Start State
        int u3 = myBridge_a[u]; 
        int v3 = myBridge_b[u]; 
        int u2 = myBridge_c[u]; 
        int v2 = myBridge_d[u]; 
        if((color[u2]^1) == color[u] || color[v2] == color[u]) {
            swapP(u2, v2);
            swapP(u3,v3);
        }
        flip(removed,mate,removedVerticesQueue,removedVerticesQueueBack,u3,v3);
        /* Original order - Note they are pushed in reverse onto the LIFO stack
        // Push state 2
        bool openingDfsSucceed1 = openingDfs(graph,dfsWL,color,removedVerticesQueue,removedVerticesQueueBack,budAtDDFSEncounter, u3,u2,u);
        assert(openingDfsSucceed1);
        // End State

        // Push state 2
        int v4 = graph.bud.directParent[u];
        bool openingDfsSucceed2 = openingDfs(graph,dfsWL,color,removedVerticesQueue,removedVerticesQueueBack,budAtDDFSEncounter,v3,v2,v4);
        assert(openingDfsSucceed2);
        augumentPath(graph,dfsWL,color,removedVerticesQueue,removedVerticesQueueBack,budAtDDFSEncounter,v4,v);
        // End
        // Push state 1
        */
        int v4 = bud.directParent[u];
        {

            int thisU = v4;
            int thisV = v;
            int thisB = -1;
            int thisOrigCur = -1;
            int thisOrigBCur = -1;
            int thisState = 0;
            pushStackVars(uStack, vStack, bStack, origCurStack, origBCurStack, stateStack, stackTop,&thisU, &thisV, &thisB, &thisOrigCur, &thisOrigBCur, &thisState);
        }
        {
            int thisU = v3;
            int thisV = v2;
            int thisB = v4;
            int thisOrigCur = -1;
            int thisOrigBCur = -1;
            int thisState = 1;
            pushStackVars(uStack, vStack, bStack, origCurStack, origBCurStack, stateStack, stackTop,&thisU, &thisV, &thisB, &thisOrigCur, &thisOrigBCur, &thisState);
        }
        {
            int thisU = u3;
            int thisV = u2;
            int thisB = u;
            int thisOrigCur = -1;
            int thisOrigBCur = -1;
            int thisState = 1;
            pushStackVars(uStack, vStack, bStack, origCurStack, origBCurStack, stateStack, stackTop,&thisU, &thisV, &thisB, &thisOrigCur, &thisOrigBCur, &thisState);
        }
    }
}

/*
Uses:
        cur
        bcur
        b
        origCur - not used
        origBCur - not used
*/

__device__ void openingDfsSubroutineCall(
                                        unsigned int * offsets,
                                        unsigned int * cols,
                                        int * oddlvl,
                                        int * evenlvl,
                                        bool * pred,
                                        unsigned int * color,
                                        int * budAtDDFSEncounter, 
                                        unsigned int * childsInDDFSTreePtr,
                                        int cur, 
                                        int bcur, 
                                        int b,
                                        int * uStack,
                                        int * vStack,
                                        int * bStack,
                                        int * origCurrStack,
                                        int * origBCurrStack,
                                        int * stateStack,
                                        int * stackTop){
    if (bcur == b)
    {
        {
            int thisU = -1;
            int thisV = -1;
            int thisB = -3;
            int thisOrigCurr = -1;
            int thisOrigBCurr = -1;
            int thisState = 4;
            pushStackVars(uStack, vStack, bStack, origCurrStack, origBCurrStack, stateStack, stackTop,&thisU, &thisV, &thisB, &thisOrigCurr, &thisOrigBCurr, &thisState);
        }
        {
            int thisU = cur;
            int thisV = bcur;
            int thisB = -1;
            int thisOrigCurr = -1;
            int thisOrigBCurr = -1;
            int thisState = 0;
            pushStackVars(uStack, vStack, bStack, origCurrStack, origBCurrStack, stateStack, stackTop,&thisU, &thisV, &thisB, &thisOrigCurr, &thisOrigBCurr, &thisState);
        }
        return;
    }


    unsigned int start = offsets[bcur];
    unsigned int end = offsets[bcur + 1];
    if(childsInDDFSTreePtr[bcur] == 0)
        childsInDDFSTreePtr[bcur]=start;
    for(; childsInDDFSTreePtr[bcur] < end; childsInDDFSTreePtr[bcur]++) {
        unsigned int edgeIndex = childsInDDFSTreePtr[bcur];
        if (pred[edgeIndex] && budAtDDFSEncounter[edgeIndex] > -1){
            if (budAtDDFSEncounter[edgeIndex] == b || color[budAtDDFSEncounter[edgeIndex]] == color[bcur]){
                {
                    int thisU = cols[edgeIndex];
                    // int thisV = budAtDDFSEncounter[edgeIndex]; Never used
                    int thisV = bcur;
                    int thisB = b;
                    int thisOrigCurr = cur;
                    int thisOrigBCurr = bcur;
                    int thisState = 2;
                    pushStackVars(uStack, vStack, bStack, origCurrStack, origBCurrStack, stateStack, stackTop,&thisU, &thisV, &thisB, &thisOrigCurr, &thisOrigBCurr, &thisState);
                }
                {
                    int thisU = cols[edgeIndex];
                    int thisV = budAtDDFSEncounter[edgeIndex];
                    int thisB = b;
                    int thisOrigCurr = -1;
                    int thisOrigBCurr = -1;
                    int thisState = 1;
                    pushStackVars(uStack, vStack, bStack, origCurrStack, origBCurrStack, stateStack, stackTop,&thisU, &thisV, &thisB, &thisOrigCurr, &thisOrigBCurr, &thisState);
                }
                return;
            }
        }
    }
    {
        
        int thisU = -1;
        int thisV = -1;
        int thisB = -4;
        int thisOrigCurr = -1;
        int thisOrigBCurr = bcur;
        int thisState = 5;
        pushStackVars(uStack, vStack, bStack, origCurrStack, origBCurrStack, stateStack, stackTop,&thisU, &thisV, &thisB, &thisOrigCurr, &thisOrigBCurr, &thisState);
    }
    return;
}


/*
Uses:
        cur
        bcur - not used
        b
        origCur;
        origBCur; -- uses bcur stack

*/
__device__ void openingDfsSubroutineCheck(
                                        unsigned int * childsInDDFSTreePtr,
                                        int cur, 
                                        int origBCur, 
                                        //int bcur, 
                                        int b,
                                        int origCur, 
                                        bool success,
                                        int * uStack,
                                        int * vStack,
                                        int * bStack,
                                        int * origCurrStack,
                                        int * origBCurrStack,
                                        int * stateStack,
                                        int * stackTop){
  if (success)
  {
    {
        int thisU = -1;
        int thisV = -1;
        int thisB = -3;
        int thisOrigCurr = -1;
        int thisOrigBCurr = origBCur;
        int thisState = 4;
        pushStackVars(uStack, vStack, bStack, origCurrStack, origBCurrStack, stateStack, stackTop,&thisU, &thisV, &thisB, &thisOrigCurr, &thisOrigBCurr, &thisState);
    }
    {
        int thisU = origBCur;
        int thisV = cur;
        int thisB = -2;
        int thisOrigCurr = -2;
        int thisOrigBCurr = -2;
        int thisState = 3;
        pushStackVars(uStack, vStack, bStack, origCurrStack, origBCurrStack, stateStack, stackTop,&thisU, &thisV, &thisB, &thisOrigCurr, &thisOrigBCurr, &thisState);
    }
    {
        int thisU = origCur;
        int thisV = origBCur;
        int thisB = -1;
        int thisOrigCurr = -1;
        int thisOrigBCurr = -1;
        int thisState = 0;
        pushStackVars(uStack, vStack, bStack, origCurrStack, origBCurrStack, stateStack, stackTop,&thisU, &thisV, &thisB, &thisOrigCurr, &thisOrigBCurr, &thisState);
    }
    return;
  } else {
    // Continue previous call
    childsInDDFSTreePtr[origBCur]++;
    {
        int thisU = origCur;
        int thisV = origBCur;
        int thisB = b;
        int thisOrigCurr = -1;
        int thisOrigBCurr = -1;
        int thisState = 1;
        pushStackVars(uStack, vStack, bStack, origCurrStack, origBCurrStack, stateStack, stackTop,&thisU, &thisV, &thisB, &thisOrigCurr, &thisOrigBCurr, &thisState);
    }
    return;
  }
}



#endif