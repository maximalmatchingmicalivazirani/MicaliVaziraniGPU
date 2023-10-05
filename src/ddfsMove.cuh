#ifndef DDFSMOVE
#define DDFSMOVE
// READ ONLY
/*
bud
offsets
cols
removed
predecessors
color1
color2
*/
// WRITE ONLY
/*
support
budAtDDFSEncounter
*/
// READ-WRITE
/*
ddfsPredecessorsPtr
stack1
stack1Top
stack2
stack2Top
supportTop
color
globalColorCounter
*/
__device__ int ddfsMove(DSU_CU bud,                                           
                        unsigned int *offsets,
                        unsigned int *cols,
                        unsigned int *ddfsPredecessorsPtr,
                        bool *removed, 
                        bool *predecessors, 
                        unsigned int * stack1, 
                        unsigned int * stack2, 
                        unsigned int * stack1Top, 
                        unsigned int * stack2Top, 
                        unsigned int * support, 
                        unsigned int * supportTop, 
                        unsigned int * color, 
                        unsigned int *globalColorCounter,
                        int*budAtDDFSEncounter, 
                        const int color1, 
                        const int color2) {
//int ddfsMove(vector<int>& stack1, const int color1, vector<int>& stack2, const int color2, vector<int>& support) {
    
    //int u = stack1.back();
    int u = stack1[stack1Top[0]-1];
    unsigned int start = offsets[u];
    unsigned int end = offsets[u + 1];
    //printf("src %d start %d end %d ddfsPredecessorsPtr %d\n",u,start, end,ddfsPredecessorsPtr[u]);
    if (!ddfsPredecessorsPtr[u]){
        ddfsPredecessorsPtr[u]=start;
        //printf("setting ddfsPredecessorsPtr to start %d\n",ddfsPredecessorsPtr[u]);
    } else{
        //printf("picking up from %d\n",ddfsPredecessorsPtr[u]);
    }

    for(; ddfsPredecessorsPtr[u] < end; ddfsPredecessorsPtr[u]++) { // Delete Neighbors of startingVertex
        int edgeIndex=ddfsPredecessorsPtr[u];
        //printf("src %d dst %d ddfsPredecessorsPtr %d  pred %d \n",u,cols[edgeIndex], ddfsPredecessorsPtr[u], predecessors[edgeIndex]);
        if (predecessors[edgeIndex]) {
            //printf("PRED OF %d is %d ddfsPredecessorsPtr[%d] %d\n",u,cols[edgeIndex],u,ddfsPredecessorsPtr[u]);
            int a = cols[edgeIndex];
            int v = bud[a];
            assert(removed[a] == removed[v]);
            if(removed[a])
                continue;
            // Found an unvisited vertex of the desired level.
            if(color[v] == 0) {
                //stack1.push_back(v);
                stack1[stack1Top[0]++]=v;
                //support.push_back(v);
                support[supportTop[0]++]=v;

                //childsInDDFSTree[u].push_back({a,v});
                //budAtDDFSEncounter[u]=v;
                budAtDDFSEncounter[edgeIndex]=v;
                //childsInDDFSTree_values[childsInDDFSTreeTop[0]]=(uint64_t) a << 32 | v;
                color[v] = color1;
                return -1;
            }
            // Found a bottleneck.  Save the bud at this point in time.
            else if(v == stack2[stack2Top[0]-1]){
                //budAtDDFSEncounter[u]=v;
                budAtDDFSEncounter[edgeIndex]=v;
                //childsInDDFSTree_values[childsInDDFSTreeTop[0]]=(uint64_t) a << 32 | v;
            }
        }
    }

    // Desired level not available.
    // Decrement leading leg of ddfs.
    // Only backtrack 1 level at the most here.
    --stack1Top[0];
    if(stack1Top[0] == 0) {
        // Some sort of base case.
        //If found bottleneck, return it
        if(stack2Top[0] == 1) { //found bottleneck
            color[stack2[stack2Top[0]-1]] = 0;
            return stack2[stack2Top[0]-1];
        }
        //If didnt find bottleneck, pop off stack2
        // Pretty sure this is the backtracking part.
        //change colors
        assert(color[stack2[stack2Top[0]-1]] == color2);
        stack1[stack1Top[0]++]=stack2[stack2Top[0]-1];
        color[stack1[stack1Top[0]-1]] = color1;
        --stack2Top[0];
    }

    return -1;
}

#endif