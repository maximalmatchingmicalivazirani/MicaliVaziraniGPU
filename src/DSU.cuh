#ifndef DSUCU
#define DSUCU
#include <thrust/sequence.h>
#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

template <typename T> __device__  void inline swap_test_device1(T& a, T& b)
{
    T c(a); a=b; b=c;
}

struct DSU_CU{
    /*
    DSU_CU(){}
    DSU_CU(int _n, int _m)
    {
        m = _m;
        n = _n;

        link_thrust_vec_h.resize(n);
        directParent_thrust_vec_h.resize(n);
        size_thrust_vec_h.resize(n);
        groupRoot_thrust_vec_h.resize(n);

        link_thrust_vec_d.resize(n);
        directParent_thrust_vec_d.resize(n);
        size_thrust_vec_d.resize(n);
        groupRoot_thrust_vec_d.resize(n);
    }
    void resize(int _n, int _m)
    {
        m = _m;
        n = _n;
        link_thrust_vec_h.resize(n);
        directParent_thrust_vec_h.resize(n);
        size_thrust_vec_h.resize(n);
        groupRoot_thrust_vec_h.resize(n);

        link_thrust_vec_d.resize(n);
        directParent_thrust_vec_d.resize(n);
        size_thrust_vec_d.resize(n);
        groupRoot_thrust_vec_d.resize(n);

        setDevice();

        reset(n);

    }
    void setHost(){
        link = thrust::raw_pointer_cast(&link_thrust_vec_h[0]);
        directParent = thrust::raw_pointer_cast(&directParent_thrust_vec_h[0]);
        size = thrust::raw_pointer_cast(&size_thrust_vec_h[0]);
        groupRoot = thrust::raw_pointer_cast(&groupRoot_thrust_vec_h[0]);
    }


    void setDevice(){
        link = thrust::raw_pointer_cast(&link_thrust_vec_d[0]);
        directParent = thrust::raw_pointer_cast(&directParent_thrust_vec_d[0]);
        size = thrust::raw_pointer_cast(&size_thrust_vec_d[0]);
        groupRoot = thrust::raw_pointer_cast(&groupRoot_thrust_vec_d[0]);
    }
    */

    int* link;
    int *directParent;
    int *size;
    int *groupRoot;
    int* link_stack;
    int* numStacks_d;
    int* maxStackDepth_d;

    unsigned int m; // Number of Edges
    unsigned int n; // Number of Vertices
    int numStacks; // Number of Vertices
    int maxStackDepth; // Number of Vertices

    /*
    thrust::host_vector<int> link_thrust_vec_h;
    thrust::host_vector<int> directParent_thrust_vec_h;
    thrust::host_vector<int> size_thrust_vec_h;
    thrust::host_vector<int> groupRoot_thrust_vec_h;


    thrust::device_vector<int> link_thrust_vec_d;
    thrust::device_vector<int> directParent_thrust_vec_d;
    thrust::device_vector<int> size_thrust_vec_d;
    thrust::device_vector<int> groupRoot_thrust_vec_d;
    */
    __host__ void reset();
    __device__ int find(int a);
    __device__ int findIteratively(int a);
    __device__ int findIterativelyReadOnly(int a);
    __device__ int findIterativelyReadOnly(int a, int *stack, int _MAXSTACKFIND);
    __device__ int operator[](const int& a);
    __device__ int operator()(const int& a);
    __device__ void linkTo(int a, int b);

};
/*
void DSU_CU::reset(int n){
    thrust::fill(size_thrust_vec_d.begin(), size_thrust_vec_d.end(), 1); // or 999999.f if you prefer
    thrust::fill(directParent_thrust_vec_d.begin(), directParent_thrust_vec_d.end(), -1); // or 999999.f if you prefer

    thrust::sequence(link_thrust_vec_d.begin(),link_thrust_vec_d.end(), 0, 1);
    thrust::sequence(groupRoot_thrust_vec_d.begin(),groupRoot_thrust_vec_d.end(), 0, 1);

    thrust::fill(size_thrust_vec_h.begin(), size_thrust_vec_h.end(), 1); // or 999999.f if you prefer
    thrust::fill(directParent_thrust_vec_h.begin(), directParent_thrust_vec_h.end(), -1); // or 999999.f if you prefer

    thrust::sequence(link_thrust_vec_h.begin(),link_thrust_vec_h.end(), 0, 1);
    thrust::sequence(groupRoot_thrust_vec_h.begin(),groupRoot_thrust_vec_h.end(), 0, 1);
}
*/

void DSU_CU::reset(){

    thrust::device_ptr<int> size_thrust_ptr=thrust::device_pointer_cast(size);
    thrust::device_ptr<int> directParent_thrust_ptr=thrust::device_pointer_cast(directParent);
    thrust::device_ptr<int> link_thrust_ptr=thrust::device_pointer_cast(link);
    thrust::device_ptr<int> groupRoot_thrust_ptr=thrust::device_pointer_cast(groupRoot);

    thrust::fill(size_thrust_ptr, size_thrust_ptr+n, 1); // or 999999.f if you prefer
    thrust::fill(directParent_thrust_ptr, directParent_thrust_ptr+n, -1); // or 999999.f if you prefer

    thrust::sequence(link_thrust_ptr,link_thrust_ptr+n, 0, 1);
    thrust::sequence(groupRoot_thrust_ptr,groupRoot_thrust_ptr+n, 0, 1);
}

//#define MAXSTACKFIND 2000
#define MAXSTACKFIND 4000
__constant__ int MAXSTACKDSU_d;
//__constant__ int MAXSTACKAUGPATH_d;

// Tail recursion
// Find(a)
// Base Case: link[a]==a, do nothing
// Recursive Case: link[a]!=a, assign link[a]=find(link[a]) and return link[a]

__device__ int DSU_CU::findIteratively(int a){
    #if GLOBALMEMORYDSU
    //printf("setting start of blockIdx %d in link stack to %d * block Idx %d\n",blockIdx.x,
    //maxStackDepth_d[0],blockIdx.x);
    int * stack = &link_stack[maxStackDepth_d[0]*blockIdx.x];
    #else
    __shared__ int stack[MAXSTACKFIND];
    #endif
    __shared__ int stackDepth;
    stackDepth = 0;
    // a is the current value of a
    // the back of the stack is index that current a will write into link[] when returning
    while(a != link[a]){
        #if GLOBALMEMORYDSU
        //assert(stackDepth+1<MAXSTACKDSU_d);
        #else
        //assert(stackDepth+1<MAXSTACKFIND);
        #endif
        stack[stackDepth++]=a;
        a=link[a];
    }
    while(stackDepth){
        link[stack[--stackDepth]]=a;
    }
    return a;
}

__device__ int DSU_CU::findIterativelyReadOnly(int a){
    // a is the current value of a
    // the back of the stack is index that current a will write into link[] when returning
    while(a != link[a]){
        a=link[a];
    }
    return a;
}

__device__ int DSU_CU::findIterativelyReadOnly(int a, int *stack, int _MAXSTACKFIND){
    __shared__ int stackDepth;
    stackDepth = 0;
    // a is the current value of a
    // the back of the stack is index that current a will write into link[] when returning
    while(a != link[a]){
        //assert(stackDepth+1<_MAXSTACKFIND);
        stack[stackDepth++]=a;
        a=link[a];
    }
    return a;
}

__device__ int DSU_CU::find(int a){
    return link[a] = (a == link[a] ? a : find(link[a]));
}

/*
__device__  int DSU_CU::operator[](const int& a){
    return groupRoot[find(a)];
}
__device__  void DSU_CU::linkTo(int a, int b){
    assert(directParent[a] == -1);
    assert(directParent[b] == -1);
    directParent[a] = b;
    a = find(a);
    b = find(b);
    int gr = groupRoot[b];
    assert(a != b);
    
    if(size[a] > size[b])
        swap_test_device1(a,b);
    link[b] = a;
    size[a] += size[b];
    groupRoot[a] = gr;
}
*/
__device__  int DSU_CU::operator[](const int& a){
    return groupRoot[findIteratively(a)];
}
__device__  int DSU_CU::operator()(const int& a){
    return groupRoot[findIterativelyReadOnly(a)];
}
__device__  void DSU_CU::linkTo(int a, int b){
    assert(directParent[a] == -1);
    assert(directParent[b] == -1);
    directParent[a] = b;
    a = findIteratively(a);
    b = findIteratively(b);
    int gr = groupRoot[b];
    assert(a != b);
    
    if(size[a] > size[b])
        swap_test_device1(a,b);
    link[b] = a;
    size[a] += size[b];
    groupRoot[a] = gr;
}

DSU_CU allocate_DSU(int _n, int _maxStackDepth,int _numStacks){

    DSU_CU dsu;
    dsu.maxStackDepth=_maxStackDepth;
    dsu.numStacks=_numStacks;
    dsu.n=_n;
    checkCudaErrors(cudaMalloc((void**) &dsu.directParent,sizeof(int)*dsu.n));
    checkCudaErrors(cudaMalloc((void**) &dsu.groupRoot,sizeof(int)*dsu.n));
    checkCudaErrors(cudaMalloc((void**) &dsu.link,sizeof(int)*dsu.n));
    checkCudaErrors(cudaMalloc((void**) &dsu.size,sizeof(int)*dsu.n));

    #if GLOBALMEMORYDSU
    printf("Allocating %d (%d x %d) memory for link_stack\n",dsu.maxStackDepth*dsu.numStacks,dsu.maxStackDepth,dsu.numStacks);
    printf("Allocating %d (%d x %d) memory for link_stack\n",_maxStackDepth*_numStacks,_maxStackDepth,_numStacks);

    checkCudaErrors(cudaMalloc((void**) &dsu.link_stack,sizeof(int)*dsu.maxStackDepth*dsu.numStacks));
    checkCudaErrors(cudaMalloc((void**) &dsu.maxStackDepth_d,sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &dsu.numStacks_d,sizeof(int)));
    cudaMemcpy(dsu.maxStackDepth_d, &dsu.maxStackDepth, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dsu.numStacks_d, &dsu.numStacks, sizeof(int), cudaMemcpyHostToDevice);
    #endif

    thrust::device_ptr<int> size_thrust_ptr=thrust::device_pointer_cast(dsu.size);
    thrust::device_ptr<int> directParent_thrust_ptr=thrust::device_pointer_cast(dsu.directParent);
    thrust::device_ptr<int> link_thrust_ptr=thrust::device_pointer_cast(dsu.link);
    thrust::device_ptr<int> groupRoot_thrust_ptr=thrust::device_pointer_cast(dsu.groupRoot);

    thrust::fill(size_thrust_ptr, size_thrust_ptr+dsu.n, 1); // or 999999.f if you prefer
    thrust::fill(directParent_thrust_ptr, directParent_thrust_ptr+dsu.n, -1); // or 999999.f if you prefer

    thrust::sequence(link_thrust_ptr,link_thrust_ptr+dsu.n, 0, 1);
    thrust::sequence(groupRoot_thrust_ptr,groupRoot_thrust_ptr+dsu.n, 0, 1);

    return dsu;
}
#endif
