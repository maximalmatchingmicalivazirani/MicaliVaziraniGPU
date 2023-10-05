#ifndef KERNELARGS
#define KERNELARGS
#include "argStructs.cuh"
#include "DSU.cuh"

struct KernelArgs
{
    KernelArgs(unsigned int * _bridgeIndices,
                                unsigned int *_numBridges,
                                unsigned int *_rows,
                                unsigned int *_cols,
                                DSU_CU & _bud, 
                                ddfsStruct & _ddfs,
                                updateLvlStruct & _updateLvl,
                                APStruct & _ap,
                                int & _depth) : 
                                bridgeIndices(_bridgeIndices),
                                numBridges(_numBridges),
                                rows(_rows),
                                cols(_cols),
                                bud(_bud),
                                ddfs(_ddfs),
                                updateLvl(_updateLvl),
                                ap(_ap),
                                depth(_depth)
                                {

                                }
    unsigned int* bridgeIndices; 
    unsigned int* numBridges; 
    unsigned int* rows;
    unsigned int* cols;
    DSU_CU bud;
    ddfsStruct ddfs;
    updateLvlStruct updateLvl;
    APStruct ap;
    int depth;
};
#endif