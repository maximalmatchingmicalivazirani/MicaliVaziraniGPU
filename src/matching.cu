// complexity O(E sqrt(V) log*(V))
#include <bits/stdc++.h>
#include <chrono>
#include <unistd.h>

#define USE_DEVICE 1
#define TIME 1
#define SEPARATE2 1
#define PRINT 0
#define GLOBALMEMORYDSU 1
#define DEBUGDSU 0
#include "mmiof.h"
#include "matchgpu.h"

#include "CSRGraph.cuh"
#include "constants.cuh"
#include "ddfs.cuh"
#include "updateLvl.cuh"
#include "augmentPath.cuh"
#include "ConnectedComponents.cuh"


#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/zip_function.h>
#include <thrust/unique.h>
#include "kernelArgs.cuh"
#include "argStructs.cuh"
#include <omp.h>
#include <string>
#include <vector>



using namespace std;
#define st first
#define nd second
typedef pair<int, int> pii;

#include <stack>
#include <tuple>
std::vector<int> childsInDDFSTreePtr;
int iter = 0;
unsigned int globalCounter = 0;

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, char const* const func, char const* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const* const file, int const line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


template <class T, class A>
T join(const A &begin, const A &end, const T &t)
{
  T result;
  for (A it = begin;
       it != end;
       it++)
  {
    if (!result.empty())
      result.append(t);
    result.append(std::to_string(*it));
  }
  return result;
}

struct is_less_than_0
{
  __host__ __device__ int operator()(int &x)
  {
    if (x > -1)
      return x;
    else
      return -1;
  }
};


struct e_op_t
   { 
      int depth;
      __host__ __device__
       bool operator()(const thrust::tuple<char, int>& x)
       {
         return thrust::get<0>(x)==2 && (depth*2)+1==thrust::get<1>(x); // get<0> instead of x[0]
      }
   };

struct b_op_t
   { 
      unsigned int * rows;
      unsigned int * cols;
      __host__ __device__
       thrust::tuple<unsigned int, unsigned int> operator()(const unsigned int & x)
       {
         return thrust::make_tuple(rows[x],cols[x]); // get<0> instead of x[0]
      }
   };




template< typename T >
typename std::vector<T>::iterator 
   insert_sorted( std::vector<T> & vec, T const& item )
{
    return vec.insert
        ( 
            std::upper_bound( vec.begin(), vec.end(), item ),
            item 
        );
}


__global__ void setSuperSource_no_bloss_simple(unsigned int *CP_d, unsigned int *IC_d, unsigned int *f_d, int *S_d, float *sigma_d, int *m_d, int *search_tree_src_d, int n)
{
  int threadId = threadIdx.x + blockIdx.x * blockDim.x;
  if (threadId >= n)
    return;
  if (m_d[threadId] == -1)
  {
    f_d[threadId] = 1;
    sigma_d[threadId] = 1.0;
    S_d[threadId] = 0;
    search_tree_src_d[threadId] = threadId;
  }
} // end  setSuperSource

__global__ void spMvUgCscScKernel_all_edges(unsigned int *CP_d, unsigned int *IC_d, unsigned int *ft_d, unsigned int *f_d,
                                            float *sigma_d, int *pred_d, int *search_tree_src_d, int d, int n)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n)
  {
    // compute spmv
    ft_d[i] = 0;
    if (sigma_d[i] < 0.01)
    {
      int sum = 0;
      int k;
      int start = CP_d[i];
      int end = CP_d[i + 1];
      for (k = start; k < end; k++)
      {
        if (f_d[IC_d[k]])
        {
          pred_d[i] = IC_d[k];
          search_tree_src_d[i] = search_tree_src_d[IC_d[k]];
          sum += f_d[IC_d[k]];
        }
      }
      if (sum > 0.9)
      {
        ft_d[i] = sum;
      }
    }
  }
} // end spMvUgCscScKernel

__global__ void spMvUgCscScKernel_matched_edge(int *m_d, unsigned int *ft_d, unsigned int *f_d,
                                               float *sigma_d, int *pred_d, int *search_tree_src_d, int d, int n)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n)
  {
    // compute spmv
    ft_d[i] = 0;
    if (sigma_d[i] < 0.01)
    {
      int sum = 0;
      if (m_d[i] > -1)
      {
        if (f_d[m_d[i]])
        {
          pred_d[i] = m_d[i];
          search_tree_src_d[i] = search_tree_src_d[m_d[i]];
          sum += f_d[m_d[i]];
        }
      }
      if (sum > 0.9)
      {
        ft_d[i] = sum;
      }
    }
  }
} // end spMvUgCscScKernel

/**************************************************************************/
/*
 * assign vector ft_d to vector f_d,
 * check that the vector f_d  has at least one nonzero element
 * add the vector f to vector sigma.
 * compute the S vector.
 */
__global__ void bfsFunctionsKernel(unsigned int *f_d, unsigned int *ft_d, float *sigma_d, int *S_d, int *c,
                                   int n, int d)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n)
  {
    f_d[i] = 0;
    if (ft_d[i] > 0.9)
    {
      *c = 1;
      f_d[i] = ft_d[i];
      sigma_d[i] += ft_d[i];
      S_d[i] = d;
    }
  }
} // end  bfsFunctionsKernel

__global__ void setAllPaths_augmenting(unsigned int *CP_d, unsigned int *IC_d, int *m_d, int *S_in_d, float *sigma_d, int *search_tree_src_d, int n, uint64_t *BTypePair_list_d, unsigned int *BTypePair_list_counter_d)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n)
  {

    // printf("%d %d %d %d %f\n", i,S_in_d[i],search_tree_src_d[i],m_d[i],sigma_d[i]);
    if (sigma_d[i] > 0.1)
    {

      // One of these will be -1.
      int d = S_in_d[i];
      if (d % 2)
      {
        if (m_d[i] > -1)
        {
          int foundB = 1;
          foundB &= S_in_d[i] == S_in_d[m_d[i]];
          foundB &= search_tree_src_d[i] != search_tree_src_d[m_d[i]];
          foundB &= m_d[search_tree_src_d[i]] == -1;
          foundB &= m_d[search_tree_src_d[m_d[i]]] == -1;
          if (foundB)
          {
            // if(0 == atomicCAS(mutex, 0, 1)){
            uint32_t leastSignificantWord = i;
            uint32_t mostSignificantWord = m_d[i];

            uint64_t edgePair = (uint64_t)mostSignificantWord << 32 | leastSignificantWord;
            int topLocal = atomicAdd(BTypePair_list_counter_d, 1);
            if (topLocal >= n)
            {
              //#ifndef NDEBUG
              //printf("EXCEEDED N %u %u, S[%u]= %d, S[%u]= %d; search_tree_src_d[%d]=%d search_tree_src_d[%d]=%d\n", leastSignificantWord, mostSignificantWord, leastSignificantWord, S_in_d[leastSignificantWord], mostSignificantWord, S_in_d[mostSignificantWord], i, search_tree_src_d[i], m_d[i], search_tree_src_d[m_d[i]]);
              //#endif
            }
            else
            {
              BTypePair_list_d[topLocal] = edgePair;
            }
          }
        }
      }
      else
      {
        int k;
        int start = CP_d[i];
        int end = CP_d[i + 1];
        for (k = start; k < end; k++)
        {
          if (sigma_d[IC_d[k]] > 0.0 && S_in_d[i] == S_in_d[IC_d[k]])
          {
            int foundB = S_in_d[i] == S_in_d[IC_d[k]] &&
                         search_tree_src_d[i] != search_tree_src_d[IC_d[k]] &&
                         m_d[search_tree_src_d[i]] == -1 &&
                         m_d[search_tree_src_d[IC_d[k]]] == -1;
            if (foundB)
            {
              // if(0 == atomicCAS(mutex, 0, 1)){
              uint32_t leastSignificantWord = i;
              uint32_t mostSignificantWord = IC_d[k];
              uint64_t edgePair = (uint64_t)mostSignificantWord << 32 | leastSignificantWord;
              int topLocal = atomicAdd(BTypePair_list_counter_d, 1);
              if (topLocal >= n)
              {
                //#ifndef NDEBUG
                //printf("EXCEEDED N %u %u, S[%u]= %d, S[%u]= %d; search_tree_src_d[%d]=%d search_tree_src_d[%d]=%d\n", leastSignificantWord, mostSignificantWord, leastSignificantWord, S_in_d[leastSignificantWord], mostSignificantWord, S_in_d[mostSignificantWord], i, search_tree_src_d[i], m_d[i], search_tree_src_d[m_d[i]]);
                //#endif
              }
              else
              {
                BTypePair_list_d[topLocal] = edgePair;
              }
            }
          }
        }
      }
    }
  }
}

//==== Random greedy matching kernels ====
__global__ void grRequestEdgeList(uint64_t *BTypePair_list_d, int *search_tree_src_d, unsigned int *BTypePair_list_counter_d, int *requests, const int *match, const int nrVertices)
{

  // Let all blue vertices make requests.
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= BTypePair_list_counter_d[0] || i >= nrVertices)
    return;

  // const int2 indices = tex1Dfetch(neighbourRangesTexture, i);
  uint32_t curr_u = (uint32_t)BTypePair_list_d[i];
  uint32_t curr_v = (BTypePair_list_d[i] >> 32);
  int curr_u_root = search_tree_src_d[curr_u];
  int curr_v_root = search_tree_src_d[curr_v];
  // Look at all blue vertices and let them make requests.
  if (match[curr_u_root] == 0 && match[curr_v_root] == 1)
  {
    requests[curr_u_root] = curr_v;
  }
  else if (match[curr_v_root] == 0 && match[curr_u_root] == 1)
  {
    requests[curr_v_root] = curr_u;
  }
}

__global__ void grRespondEdgeList(uint64_t *BTypePair_list_d, int *search_tree_src_d, unsigned int *BTypePair_list_counter_d, int *requests, const int *match, const int nrVertices)
{
  // Let all blue vertices make requests.
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= BTypePair_list_counter_d[0] || i >= nrVertices)
    return;

  uint32_t curr_u = (uint32_t)BTypePair_list_d[i];
  uint32_t curr_v = (BTypePair_list_d[i] >> 32);
  int curr_u_root = search_tree_src_d[curr_u];
  int curr_v_root = search_tree_src_d[curr_v];
  // Look at all blue vertices and let them make requests.
  if (match[curr_u_root] == 0 && match[curr_v_root] == 1 && requests[curr_u_root] == curr_v)
  {
    requests[curr_v_root] = curr_u;
  }
  else if (match[curr_v_root] == 0 && match[curr_u_root] == 1 && requests[curr_v_root] == curr_u)
  {
    requests[curr_u_root] = curr_v;
  }
}

__global__ void gMatchEdgeList(uint64_t *BTypePair_disjoint_list_d, unsigned int *BTypePair_disjoint_list_counter_d, uint64_t *BTypePair_list_d, int *search_tree_src_d, unsigned int *BTypePair_list_counter_d,
                               int *match, const int *requests, const int nrVertices)
{

  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= BTypePair_list_counter_d[0] || i >= nrVertices)
    return;

  uint32_t curr_u = (uint32_t)BTypePair_list_d[i];
  uint32_t curr_v = (BTypePair_list_d[i] >> 32);
  int curr_u_root = search_tree_src_d[curr_u];
  int curr_v_root = search_tree_src_d[curr_v];

  const int r_u = requests[curr_u_root];
  const int r_v = requests[curr_v_root];

  if (r_u < nrVertices && r_v < nrVertices)
  {
    // This vertex has made a valid request.
    if (r_u == curr_v && r_v == curr_u && curr_u < curr_v)
    {
      // Match the vertices if the request was mutual.
      // match[i] = 4 + min(i, r);
      //  I need a pointer to the match for traversal.
      match[curr_u_root] = 4 + curr_v;
      match[curr_v_root] = 4 + curr_u;
      uint64_t edgePair = (uint64_t)curr_v << 32 | curr_u;
      int top = atomicAdd(BTypePair_disjoint_list_counter_d, 1);
      BTypePair_disjoint_list_d[top] = edgePair;
    }
  }
}

__global__ void lift_path_parallel(int *m_d, int *pred_d, uint64_t *BTypePair_disjoint_list_d, unsigned int *BTypePair_disjoint_list_counter_d)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= BTypePair_disjoint_list_counter_d[0])
    return;
  int curr_v = (int)(BTypePair_disjoint_list_d[i] >> 32);
  int curr_u = (int)(uint32_t)BTypePair_disjoint_list_d[i];

  const int a = curr_u;
  const int b = curr_v;
  int last_a;
  int last_b;
  int a_length = 0;
  int curr_a = a;
  bool matchFirst = m_d[a] == b;
  if (matchFirst)
  {
    while (curr_a != -1)
    {
      last_a = curr_a;
      if (a_length % 2)
      {
        curr_a = pred_d[curr_a];
      }
      else
      {
        curr_a = pred_d[curr_a];
        m_d[curr_a] = last_a;
        m_d[last_a] = curr_a;
      }
      a_length++;
    }
  }
  else
  {
    while (curr_a != -1)
    {
      last_a = curr_a;
      if (a_length % 2)
      {
        curr_a = pred_d[curr_a];
        m_d[curr_a] = last_a;
        m_d[last_a] = curr_a;
      }
      else
      {
        curr_a = pred_d[curr_a];
      }
      a_length++;
    }
  }
  int b_length = 0;
  int curr_b = b;
  if (matchFirst)
  {
    while (curr_b != -1)
    {
      last_b = curr_b;
      if (b_length % 2)
      {
        curr_b = pred_d[curr_b];
      }
      else
      {
        curr_b = pred_d[curr_b];
        m_d[curr_b] = last_b;
        m_d[last_b] = curr_b;
      }
      b_length++;
    }
  }
  else
  {
    while (curr_b != -1)
    {
      last_b = curr_b;
      if (b_length % 2)
      {
        curr_b = pred_d[curr_b];
        m_d[curr_b] = last_b;
        m_d[last_b] = curr_b;
      }
      else
      {
        curr_b = pred_d[curr_b];
      }
      b_length++;
    }
  }
  if (!matchFirst)
  {
    m_d[a] = b;
    m_d[b] = a;
  }
}


struct GreedyMatcher
{

  GreedyMatcher(CSRGraph &_csr) : csr(_csr)
  {
    m_h.resize(csr.n);
    c_h.resize(1, 1);
    m_d.resize(csr.n, 0);
    req_d.resize(csr.n, 0);
    c_d.resize(1, 1);
    cudaMallocHost((void**)&c_Pinned, sizeof(int)); // host pinned
  }
  int maxMatch(std::vector<int> &mate)
  {
    unsigned int *rows_d_ptr = thrust::raw_pointer_cast(csr.offsets_d.data());
    unsigned int *cols_d_ptr = thrust::raw_pointer_cast(csr.cols_d.data());
    //int *m_d_ptr = thrust::raw_pointer_cast(m_d.data());
    int *m_d_ptr = thrust::raw_pointer_cast(csr.mate_d.data());
    int *req_d_ptr = thrust::raw_pointer_cast(req_d.data());
    int *c_ptr = thrust::raw_pointer_cast(c_d.data());
    srand(1);
    /*computing MM */
    int matchround = 0;
    int dimGrid = (csr.n + THREADS_PER_BLOCK) / THREADS_PER_BLOCK;
    memset(c_Pinned, 1, sizeof(int));
    while (c_Pinned[0] && ++matchround < NR_MAX_MATCH_ROUNDS)
    {
      //printf("match round %d\n", matchround);
      //c_h[0] = 0;
      //c_d = c_h;
      memset(c_Pinned, 0, sizeof(int));
      gaSelect<<<dimGrid, THREADS_PER_BLOCK>>>(m_d_ptr, c_ptr, csr.n, rand());
      grRequest<<<dimGrid, THREADS_PER_BLOCK>>>(rows_d_ptr, cols_d_ptr, req_d_ptr, m_d_ptr, csr.n);
      grRespond<<<dimGrid, THREADS_PER_BLOCK>>>(rows_d_ptr, cols_d_ptr, req_d_ptr, m_d_ptr, csr.n);
      gMatch<<<dimGrid, THREADS_PER_BLOCK>>>(m_d_ptr, req_d_ptr, csr.n);
      cudaMemcpy(c_Pinned, thrust::raw_pointer_cast(c_d.data()), sizeof(int), cudaMemcpyDeviceToHost);
      //c_h = c_d;
    }
    using namespace thrust::placeholders;
    thrust::for_each(csr.mate_d.begin(), csr.mate_d.end(), _1 -= 4);
    thrust::transform(csr.mate_d.begin(), csr.mate_d.end(), csr.mate_d.begin(), is_less_than_0()); // in-place transformation
    //m_h = csr.mate_d;
    //for (int i = 0; i < m_h.size(); ++i)
    //    printf("%d %d\n", i, m_h[i]);
    //for (int i = 0; i < m_h.size(); ++i)
    //  mate[i] = m_h[i];
    int numAugmented = thrust::count_if(csr.mate_d.begin(), csr.mate_d.end(), _1 > -1);
    return numAugmented / 2;
  }

  CSRGraph &csr;

  thrust::host_vector<int> m_h;
  thrust::device_vector<int> c_h;

  thrust::device_vector<int> m_d;
  thrust::device_vector<int> req_d;
  thrust::device_vector<int> c_d;
  int * c_Pinned;
};

struct BFS
{
  BFS(CSRGraph &_csr, GreedyMatcher &_gm) : csr(_csr), gm(_gm)
  {

    f_h.resize(csr.n);
    ft_h.resize(csr.n);
    S_h.resize(csr.n);
    pred_h.resize(csr.n);
    sigma_h.resize(csr.n);
    search_tree_src_h.resize(csr.n, -1);
    c_h.resize(1, 1);

    f_d.resize(csr.n, 0);
    ft_d.resize(csr.n, 0);
    S_d.resize(csr.n, 0);
    pred_d.resize(csr.n, -1);
    sigma_d.resize(csr.n, 0.0);
    search_tree_src_d.resize(csr.n, -1);
    c_d.resize(1, 1);

    BTypePair_list_counter_d.resize(1, 0);
    BTypePair_list_d.resize(csr.n, 0);
    BTypePair_disjoint_list_counter_d.resize(1, 0);
    BTypePair_disjoint_list_d.resize(csr.n, 0);
    m2_d.resize(csr.n, 0);
  }

  int augmentNaivePaths(std::vector<int> &mate)
  {

    int numAugmented = 0;

    unsigned int *rows_d_ptr = thrust::raw_pointer_cast(csr.offsets_d.data());
    unsigned int *cols_d_ptr = thrust::raw_pointer_cast(csr.cols_d.data());

    unsigned int *f_d_ptr = thrust::raw_pointer_cast(f_d.data());
    unsigned int *ft_d_ptr = thrust::raw_pointer_cast(ft_d.data());

    int *S_d_ptr = thrust::raw_pointer_cast(S_d.data());
    float *sigma_d_ptr = thrust::raw_pointer_cast(sigma_d.data());
    int *search_tree_src_d_ptr = thrust::raw_pointer_cast(search_tree_src_d.data());
    int *pred_d_ptr = thrust::raw_pointer_cast(pred_d.data());

    int *m_d_ptr = thrust::raw_pointer_cast(csr.mate_d.data());
    //int *m_d_ptr = thrust::raw_pointer_cast(gm.m_d.data());
    int *c_d_ptr = thrust::raw_pointer_cast(c_d.data());
    int *c_d_m_ptr = thrust::raw_pointer_cast(gm.c_d.data());

    uint64_t *BTypePair_list_d_ptr = thrust::raw_pointer_cast(BTypePair_list_d.data());
    unsigned int *BTypePair_list_counter_d_ptr = thrust::raw_pointer_cast(BTypePair_list_counter_d.data());
    uint64_t *BTypePair_disjoint_list_d_ptr = thrust::raw_pointer_cast(BTypePair_disjoint_list_d.data());
    unsigned int *BTypePair_disjoint_list_counter_d_ptr = thrust::raw_pointer_cast(BTypePair_disjoint_list_counter_d.data());

    int *m2_d_ptr = thrust::raw_pointer_cast(m2_d.data());
    int *req_d_ptr = thrust::raw_pointer_cast(gm.req_d.data());
    // initialize all of the edges of the super source to
    //  f_d[r] = 1;
    //  sigma_d[r] = 1.0;
    int dimGrid = (csr.n + THREADS_PER_BLOCK) / THREADS_PER_BLOCK;

    do
    {

      setSuperSource_no_bloss_simple<<<dimGrid, THREADS_PER_BLOCK>>>(rows_d_ptr, cols_d_ptr, f_d_ptr, S_d_ptr, sigma_d_ptr, m_d_ptr, search_tree_src_d_ptr, csr.n);
      int d = 0;
      c_h[0] = 1;
      while (c_h[0])
      {
        d = d + 1;
        c_h[0] = 0;
        c_d = c_h;
        if (d % 2)
        {
          spMvUgCscScKernel_all_edges<<<dimGrid, THREADS_PER_BLOCK>>>(rows_d_ptr, cols_d_ptr, ft_d_ptr, f_d_ptr, sigma_d_ptr, pred_d_ptr, search_tree_src_d_ptr, d, csr.n);
        }
        else
        {
          spMvUgCscScKernel_matched_edge<<<dimGrid, THREADS_PER_BLOCK>>>(m_d_ptr, ft_d_ptr, f_d_ptr, sigma_d_ptr, pred_d_ptr, search_tree_src_d_ptr, d, csr.n);
        }
        bfsFunctionsKernel<<<dimGrid, THREADS_PER_BLOCK>>>(f_d_ptr, ft_d_ptr, sigma_d_ptr, S_d_ptr, c_d_ptr, csr.n, d);
        c_h = c_d;
      }
      setAllPaths_augmenting<<<dimGrid, THREADS_PER_BLOCK>>>(rows_d_ptr, cols_d_ptr, m_d_ptr, S_d_ptr, sigma_d_ptr, search_tree_src_d_ptr, csr.n, BTypePair_list_d_ptr, BTypePair_list_counter_d_ptr);
      BTypePair_list_counter_h = BTypePair_list_counter_d;
      // printf("Non-disjoint %d paths\n", BTypePair_list_counter_h[0]);
      cudaMemset(req_d_ptr, 0, sizeof(*req_d_ptr) * csr.n);

      srand(1);
      /*computing MM */
      int matchround = 0;
      gm.c_h[0] = 1;
      while (gm.c_h[0] && ++matchround < NR_MAX_MATCH_ROUNDS)
      {
        // printf("match round %d\n", matchround);
        gm.c_h[0] = 0;
        gm.c_d = gm.c_h;
        gaSelect<<<dimGrid, THREADS_PER_BLOCK>>>(m2_d_ptr, c_d_m_ptr, csr.n, rand());
        grRequestEdgeList<<<dimGrid, THREADS_PER_BLOCK>>>(BTypePair_list_d_ptr, search_tree_src_d_ptr, BTypePair_list_counter_d_ptr, req_d_ptr, m2_d_ptr, csr.n);
        grRespondEdgeList<<<dimGrid, THREADS_PER_BLOCK>>>(BTypePair_list_d_ptr, search_tree_src_d_ptr, BTypePair_list_counter_d_ptr, req_d_ptr, m2_d_ptr, csr.n);
        gMatchEdgeList<<<dimGrid, THREADS_PER_BLOCK>>>(BTypePair_disjoint_list_d_ptr, BTypePair_disjoint_list_counter_d_ptr, BTypePair_list_d_ptr, search_tree_src_d_ptr, BTypePair_list_counter_d_ptr, m2_d_ptr, req_d_ptr, csr.n);
        cudaMemset(req_d_ptr, 0, sizeof(*req_d_ptr) * csr.n);
        gm.c_h = gm.c_d;
      }
      BTypePair_disjoint_list_counter_h = BTypePair_disjoint_list_counter_d;
      // printf("Disjoint %d paths\n", BTypePair_disjoint_list_counter_h[0]);

      lift_path_parallel<<<dimGrid, THREADS_PER_BLOCK>>>(m_d_ptr, pred_d_ptr, BTypePair_disjoint_list_d_ptr, BTypePair_disjoint_list_counter_d_ptr);
      numAugmented += BTypePair_disjoint_list_counter_h[0];

      cudaMemset(f_d_ptr, 0, sizeof(*f_d_ptr) * csr.n);
      cudaMemset(ft_d_ptr, 0, sizeof(*f_d_ptr) * csr.n);
      cudaMemset(S_d_ptr, -1, sizeof(*S_d_ptr) * csr.n);

      cudaMemset(sigma_d_ptr, 0, sizeof(*sigma_d_ptr) * csr.n);
      cudaMemset(pred_d_ptr, -1, sizeof(*pred_d_ptr) * csr.n);
      cudaMemset(search_tree_src_d_ptr, -1, sizeof(*search_tree_src_d_ptr) * csr.n);

      cudaMemset(m2_d_ptr, 0, sizeof(*m2_d_ptr) * csr.n);
      cudaMemset(BTypePair_list_counter_d_ptr, 0, sizeof(*BTypePair_list_counter_d_ptr));
      cudaMemset(BTypePair_disjoint_list_counter_d_ptr, 0, sizeof(*BTypePair_disjoint_list_counter_d_ptr));

    } while (BTypePair_disjoint_list_counter_h[0] > 0);
    /*
    gm.m_h = gm.m_d;
    for (int i = 0; i < gm.m_h.size(); ++i)
      mate[i] = gm.m_h[i];
    */
    return numAugmented;
  }

  CSRGraph &csr;
  GreedyMatcher &gm;

  thrust::host_vector<unsigned int> f_h;
  thrust::host_vector<unsigned int> ft_h;
  thrust::host_vector<int> S_h;
  thrust::host_vector<int> pred_h;
  thrust::host_vector<float> sigma_h;
  thrust::host_vector<int> search_tree_src_h;
  thrust::host_vector<int> c_h;

  thrust::host_vector<unsigned int> BTypePair_list_counter_h;
  thrust::host_vector<uint64_t> BTypePair_list_h;
  thrust::host_vector<unsigned int> BTypePair_disjoint_list_counter_h;
  thrust::host_vector<uint64_t> BTypePair_disjoint_list_h;

  thrust::device_vector<unsigned int> f_d;
  thrust::device_vector<unsigned int> ft_d;
  thrust::device_vector<int> S_d;
  thrust::device_vector<int> pred_d;
  thrust::device_vector<float> sigma_d;
  thrust::device_vector<int> search_tree_src_d;
  thrust::device_vector<int> c_d;

  thrust::device_vector<unsigned int> BTypePair_list_counter_d;
  thrust::device_vector<uint64_t> BTypePair_list_d;
  thrust::device_vector<unsigned int> BTypePair_disjoint_list_counter_d;
  thrust::device_vector<uint64_t> BTypePair_disjoint_list_d;
  thrust::device_vector<int> m2_d;
};

// disjoint set union data structure
struct DSU
{
  vector<int> link;
  vector<int> directParent;
  vector<int> size;
  vector<int> groupRoot;

  void reset(int n)
  {
    link = vector<int>(n);
    size = vector<int>(n, 1);
    iota(link.begin(), link.end(), 0);
    groupRoot = link;
    directParent = vector<int>(n, -1);
  }

  int find(int a)
  {
    return link[a] = (a == link[a] ? a : find(link[a]));
  }

  int operator[](const int &a)
  {
    return groupRoot[find(a)];
  }

  void linkTo(int a, int b)
  {
    assert(directParent[a] == -1);
    assert(directParent[b] == -1);
    directParent[a] = b;
    a = find(a);
    b = find(b);
    int gr = groupRoot[b];
    assert(a != b);

    if (size[a] > size[b])
      swap(a, b);
    link[b] = a;
    size[a] += size[b];
    groupRoot[a] = gr;
  }
};


struct Edge
{
  int to;
  int other;
  EdgeType type;
  Edge(int _to, int _other, EdgeType _type = NotScanned) : to(_to), other(_other), type(_type) {}
};


int n;                      // IN: nuber of vertices
vector<vector<Edge>> graph; // IN: graph as neighbours list
vector<int> mate;           // OUT: vertex which is matched with given, or -1 is unmatched

vector<vector<int>> predecessors;
vector<int> ddfsPredecessorsPtr;
vector<int> removed;
vector<int> evenlvl, oddlvl;
DSU bud;
DSU bud2;

int globalColorCounter; // resets to 1 after each iteration
// colors for bridges are numbered (2,3), (4,5), (6,7) ...
// to check if vertices belong to same petal check if color1/2 == color2/2
// color^1 is the other color for single ddfs run

vector<int> color;
vector<vector<pair<int, int>>> childsInDDFSTree; //{x, bud[x] at the moment when ddfs started}; may also contain other color vertices which previously were its childs
vector<pair<pii, pii>> myBridge;                 // bridge, bud[bridge]


std::vector<int> mate2;
std::vector<int> removed2;
queue<int> removedVerticesQueue2;

int minlvl(int u) { return min(evenlvl[u], oddlvl[u]); }

int tenacity(pii edge)
{
  if (mate[edge.st] == edge.nd)
    return oddlvl[edge.st] + oddlvl[edge.nd] + 1;
  return evenlvl[edge.st] + evenlvl[edge.nd] + 1;
}

/*
tries to move color1 down, updating colors, stacks and childs in ddfs tree
also adds each visited vertex to support of this bridge
*/
int ddfsMove(vector<int> &stack1, const int color1, vector<int> &stack2, const int color2, vector<int> &support)
{
  int u = stack1.back();
  for (; ddfsPredecessorsPtr[u] < predecessors[u].size(); ddfsPredecessorsPtr[u]++)
  {
    int a = predecessors[u][ddfsPredecessorsPtr[u]];
    int v = bud[a];
    assert(removed[a] == removed[v]);
    if (removed[a])
      continue;
    if (color[v] == 0)
    {
      stack1.push_back(v);
      support.push_back(v);
      childsInDDFSTree[u].push_back({a, v});
      color[v] = color1;
      return -1;
    }
    else if (v == stack2.back())
      childsInDDFSTree[u].push_back({a, v});
  }
  stack1.pop_back();

  if (stack1.size() == 0)
  {
    if (stack2.size() == 1)
    { // found bottleneck
      color[stack2.back()] = 0;
      return stack2.back();
    }
    // change colors
    assert(color[stack2.back()] == color2);
    stack1.push_back(stack2.back());
    color[stack1.back()] = color1;
    stack2.pop_back();
  }
  return -1;
}

// returns {r0, g0} or {bottleneck, bottleneck}
pair<int, int> ddfs(pii e, vector<int> &out_support)
{
  vector<int> Sr = {bud[e.st]}, Sg = {bud[e.nd]};
  if (Sr[0] == Sg[0])
    return {Sr[0], Sg[0]};

  out_support = {Sr[0], Sg[0]};
  int newRed = color[Sr[0]] = ++globalColorCounter, newGreen = color[Sg[0]] = ++globalColorCounter;
  assert(newRed == (newGreen ^ 1));

  for (;;)
  {
    // if found two disjoint paths
    if (minlvl(Sr.back()) == 0 && minlvl(Sg.back()) == 0)
      return {Sr.back(), Sg.back()};

    int b;
    //printf("stack1[%d]=%d ml %d stack2[%d-1]=%d ml %d \n",Sr.size(),Sr.back(),minlvl(Sr.back()),Sg.size(),Sg.back(),minlvl(Sg.back()));
    if (minlvl(Sr.back()) >= minlvl(Sg.back())){
      //printf("ENTERED IF\n");
      b = ddfsMove(Sr, newRed, Sg, newGreen, out_support);
    }else{
      //printf("ENTERED ELSE\n");
      b = ddfsMove(Sg, newGreen, Sr, newRed, out_support);
    }
    if (b != -1)
      return {b, b};
  }
}

queue<int> removedVerticesQueue;

void removeAndPushToQueue(int u)
{
  //printf("remove %d\n",u);
  removed[u] = 1;
  removedVerticesQueue.push(u);
}

void removeAndPushToQueue2(int u)
{
  removed2[u] = 1;
  removedVerticesQueue2.push(u);
}

void flip(int u, int v)
{
  if (removed[u] || removed[v] || mate[u] == v)
    return; // flipping only unmatched edges
  //printf("inside flip %d %d\n",u,v);
  removeAndPushToQueue(u);
  removeAndPushToQueue(v);
  mate[u] = v;
  mate[v] = u;
}

void flip2(int u, int v)
{
  if (removed2[u] || removed2[v] || mate2[u] == v)
    return; // flipping only unmatched edges
  removeAndPushToQueue2(u);
  removeAndPushToQueue2(v);
  mate2[u] = v;
  mate2[v] = u;
}

void augumentPath(int a, int b, bool initial = false);
std::vector< std::pair< int, std::tuple <int,int,int,int,int> > > augumentPath2(int a, int b, bool initial = false);

bool openingDfs(int cur, int bcur, int b)
{
  if (bcur == b)
  {
    augumentPath(cur, bcur);
    return true;
  }
  for (auto a : childsInDDFSTree[bcur])
  {
    if ((a.nd == b || color[a.nd] == color[bcur]) && openingDfs(a.st, a.nd, b))
    {
      augumentPath(cur, bcur);
      flip(bcur, a.st);
      return true;
    }
  }
  return false;
}

std::vector< std::pair< int, std::tuple <int,int,int,int,int> > >  openingDfs2Call(int cur, int bcur, int b, int origCur, int origBCur, int * childsInDDFSTreePtr, bool * success)
{
  std::vector< std::pair< int, std::tuple <int,int,int,int,int> > > states;
  if (bcur == b)
  {
    //augumentPath2(cur, bcur);
    states.push_back(std::pair<int,std::tuple<int,int,int,int,int>>(0,std::tuple<int,int,int,int,int>(cur, bcur,-1,-1,-1)));
    states.push_back(std::pair<int,std::tuple<int,int,int,int,int>>(4,std::tuple<int,int,int,int,int>(-1, -1,-1,-1,-1)));
    //success[0]=true;
    return states;
    //return true;
  }

  for (; childsInDDFSTreePtr[bcur] < childsInDDFSTree[bcur].size(); childsInDDFSTreePtr[bcur]++)
  {
    auto a = childsInDDFSTree[bcur][childsInDDFSTreePtr[bcur]];
  //for (auto a : childsInDDFSTree[bcur])
  //{
    if (a.nd == b || color[a.nd] == color[bcur]) 
    {
      states.push_back(std::pair<int,std::tuple<int,int,int,int,int>>(1,std::tuple<int,int,int,int,int>(a.st, a.nd, b,-1,-1)));
      states.push_back(std::pair<int,std::tuple<int,int,int,int,int>>(2,std::tuple<int,int,int,int,int>(a.st, a.nd, b,cur, bcur)));
      return states;
      //openingDfs2Call(a.st, a.nd, b);
      //openingDfs2Check(a.st, a.nd, b,cur, bcur);
    }
  }
  states.push_back(std::pair<int,std::tuple<int,int,int,int,int>>(5,std::tuple<int,int,int,int,int>(-1, -1,-1,-1,-1)));
  return states;
}


std::vector< std::pair< int, std::tuple <int,int,int,int,int> > >  openingDfs2Check(int cur, int bcur, int b, int origCur, int origBCur, int * childsInDDFSTreePtr, bool * success)
{
  std::vector< std::pair< int, std::tuple <int,int,int,int,int> > > states;
  if (success[0])
  {
    //augumentPath2(origCur, origBCur);
    //flip2(origBCur, cur);
    states.push_back(std::pair<int,std::tuple<int,int,int,int,int>>(0,std::tuple<int,int,int,int,int>(origCur, origBCur,-1,-1,-1)));
    states.push_back(std::pair<int,std::tuple<int,int,int,int,int>>(3,std::tuple<int,int,int,int,int>(origBCur, cur,-1,-1,-1)));
    states.push_back(std::pair<int,std::tuple<int,int,int,int,int>>(4,std::tuple<int,int,int,int,int>(-1, -1,-1,-1,-1)));
    return states;
  } else {
    // Continue previous call
    childsInDDFSTreePtr[origBCur]++;
    states.push_back(std::pair<int,std::tuple<int,int,int,int,int>>(1,std::tuple<int,int,int,int,int>(origCur, origBCur, b,-1,-1)));
    return states;
  }
}

void augumentPath(int u, int v, bool initial)
{
  //printf("cpu augumentPath %d %d\n",u,v);
  if (u == v)
    return;
  if (!initial && minlvl(u) == evenlvl[u])
  {                                      // simply follow predecessors
    assert(predecessors[u].size() == 1); // u should be evenlevel (last minlevel edge is matched, so there is only one predecessor)
    int x = predecessors[u][0];          // no need to flip edge since we know it's matched

    int idx = 0;
    while (bud[predecessors[x][idx]] != bud[x])
    {
      idx++;
      assert(idx < (int)predecessors[x].size());
    }
    u = predecessors[x][idx];
    assert(!removed[u]);
    flip(x, u);
    augumentPath(u, v);
  }
  else
  { // through bridge
    auto u3 = myBridge[u].st.st, v3 = myBridge[u].st.nd, u2 = myBridge[u].nd.st, v2 = myBridge[u].nd.nd;
    if ((color[u2] ^ 1) == color[u] || color[v2] == color[u])
    {
      swap(u2, v2);
      swap(u3, v3);
    }

    flip(u3, v3);
    bool openingDfsSucceed1 = openingDfs(u3, u2, u);
    assert(openingDfsSucceed1);

    int v4 = bud.directParent[u];
    bool openingDfsSucceed2 = openingDfs(v3, v2, v4);
    assert(openingDfsSucceed2);
    augumentPath(v4, v);
  }
}

std::vector< std::pair< int, std::tuple <int,int,int,int,int> > > augumentPath2(int u, int v, bool initial)
{
  if (u == v)
    return std::vector< std::pair< int, std::tuple <int,int,int,int,int> > >{};
  if (!initial && minlvl(u) == evenlvl[u])
  {                                      // simply follow predecessors
    assert(predecessors[u].size() == 1); // u should be evenlevel (last minlevel edge is matched, so there is only one predecessor)
    int x = predecessors[u][0];          // no need to flip edge since we know it's matched

    int idx = 0;
    while (bud[predecessors[x][idx]] != bud[x])
    {
      idx++;
      assert(idx < (int)predecessors[x].size());
    }
    u = predecessors[x][idx];
    assert(!removed2[u]);
    flip2(x, u);
    std::vector< std::pair< int, std::tuple <int,int,int,int,int> > > states;
    states.push_back(std::pair<int,std::tuple<int,int,int,int,int>>(0,std::tuple<int,int,int,int,int>(u,v,-1,-1,-1)));
    return states;
    //augumentPath2(u, v);
  }
  else
  { // through bridge
    auto u3 = myBridge[u].st.st, v3 = myBridge[u].st.nd, u2 = myBridge[u].nd.st, v2 = myBridge[u].nd.nd;
    if ((color[u2] ^ 1) == color[u] || color[v2] == color[u])
    {
      swap(u2, v2);
      swap(u3, v3);
    }

    flip2(u3, v3);
    
    std::vector< std::pair< int, std::tuple <int,int,int,int,int> > > states;
    states.push_back(std::pair<int,std::tuple<int,int,int,int,int>>(1,std::tuple<int,int,int,int,int>(u3,u2,u,-1,-1)));
    //bool openingDfsSucceed1 = openingDfs2(u3, u2, u);
    //assert(openingDfsSucceed1);

    int v4 = bud.directParent[u];
    states.push_back(std::pair<int,std::tuple<int,int,int,int,int>>(1,std::tuple<int,int,int,int,int>(v3,v2,v4,-1,-1)));

    //bool openingDfsSucceed2 = openingDfs2(v3, v2, v4);
    //assert(openingDfsSucceed2);
    //augumentPath2(v4, v);
    states.push_back(std::pair<int,std::tuple<int,int,int,int,int>>(0,std::tuple<int,int,int,int,int>(v4,v,-1,-1,-1)));

    return states;

  }
}

// Side effects: changes removed, mate, and removedVerticesQueue
void augumentPathStack(int initU, int initV, bool initial)
{
  // Method, arguments
  std::stack< std::pair< int, std::tuple <int,int,int,int,int> > > stack;
  bool success = false;
  int u;
  int v;
  int b;
  int origCur;
  int origBcur;
  std::vector< std::pair< int, std::tuple <int,int,int,int,int> > > newstates = augumentPath2(initU,initV,true);
  for (std::vector<std::pair< int, std::tuple <int,int,int,int,int> >>::reverse_iterator i = newstates.rbegin(); 
          i != newstates.rend(); ++i ) { 
    stack.push(*i);
  } 

  int MethodNum;
  std::tuple <int,int,int,int,int> state;
  while(stack.size()){
    MethodNum = stack.top().first;
    state = stack.top().second;
    stack.pop();
    switch(MethodNum) {
      case 0 :
          u = std::get<0>(state);
          v  = std::get<1>(state);
          newstates = augumentPath2(u,v,false);
          for (std::vector<std::pair< int, std::tuple <int,int,int,int,int> >>::reverse_iterator i = newstates.rbegin(); 
                  i != newstates.rend(); ++i ) { 
            stack.push(*i);
          } 
          break; //optional
      case 1 :
          u = std::get<0>(state);
          v  = std::get<1>(state);
          b  = std::get<2>(state);
          origCur  = std::get<3>(state);
          origBcur  = std::get<4>(state);
          newstates = openingDfs2Call(u, v, b, origCur, origBcur, &childsInDDFSTreePtr[0], &success);
          for (std::vector<std::pair< int, std::tuple <int,int,int,int,int> >>::reverse_iterator i = newstates.rbegin(); 
                  i != newstates.rend(); ++i ) { 
            stack.push(*i);
          } 
          break; //optional
      case 2 :
          u = std::get<0>(state);
          v  = std::get<1>(state);
          b  = std::get<2>(state);
          origCur  = std::get<3>(state);
          origBcur  = std::get<4>(state);
          newstates = openingDfs2Check(u, v, b, origCur, origBcur, &childsInDDFSTreePtr[0], &success);
          for (std::vector<std::pair< int, std::tuple <int,int,int,int,int> >>::reverse_iterator i = newstates.rbegin(); 
                  i != newstates.rend(); ++i ) { 
            stack.push(*i);
          } 
          break; //optional
      case 3 :
          u = std::get<0>(state);
          v  = std::get<1>(state);
          flip2(u, v);
          break; //optional
      case 4 :
          success = true;
          break; //optional
      case 5 :
          success = false;
          break; //optional
      // you can have any number of case statements.
      default : //Optional
          break;
    }
  }
}


bool bfs() {
    vector<vector<int> > verticesAtLevel(n);
    vector<vector<pii> > bridges(2*n+2);
    vector<int> removedPredecessorsSize(n);

    auto setLvl = [&](int u, int lev) {
        if(lev&1) oddlvl[u] = lev; else evenlvl[u] = lev;
        verticesAtLevel[lev].push_back(u);
    };

    for(int u=0;u<n;u++)
        if(mate[u] == -1)
            setLvl(u,0);

    bool foundPath = false;  
    for(int i=0;i<n && !foundPath;i++) {
        // This can be performed in parallel
        // each edge e only belongs to one vertex in the frontier.
        // The only race condition is if two outgoing edges in the frontier point to the same vertex v.
        // Then it is possible they would both add themselves to v's predecessors.
        // Not sure if this is a problem.
        // If it is, it can be solved by claiming vertices atomically.

        // Since the level is at most reduced to i+1, there are no race conditions due to
        // order of operations between the if(lvl>=i+1) and else

        // Note: this is a ms-bfs approach since all vertices at the level are grown.
        for(auto u : verticesAtLevel[i]) {
            for(auto& e:graph[u]) {
                if(e.type == NotScanned && (oddlvl[u] == i) == (mate[u] == e.to)) {
                    if(minlvl(e.to) >= i+1) {
                        e.type = Prop;
                        graph[e.to][e.other].type = Prop;

                        if(minlvl(e.to) > i+1)
                            setLvl(e.to,i+1);
                        predecessors[e.to].push_back(u);
                    }
                    else {
                        e.type = Bridge;
                        graph[e.to][e.other].type = Bridge;
                        if(tenacity({u,e.to}) < INF) {
                            bridges[tenacity({u,e.to})].push_back({u,e.to});
                        }
                    }
                }
            }
        }
        
        // This loop should be parallelized using a dynamic worklist.  Bridges should be popped of the stack
        // and ddfs performed using a copy of the graph.  There is no cooperation between the searchers in this version.
        for(auto b : bridges[2*i+1]) {
            if(removed[bud[b.st]] || removed[bud[b.nd]])
                continue;
            vector<int> support;
            auto ddfsResult = ddfs(b,support);
            pair<pii,pii> curBridge = {b,{bud[b.st], bud[b.nd]}};
            /*even when we found two disjoint paths, we create fake petal, with bud in the end of second path
            the support of this bridge will be these two pathes and some other vertices, which have bases on this paths, so we will remove them and this will not affect corectness
            using this trick, we can simply call augumentPath on these two ends - the first end is just above fake bud, so it will augument exactly the path we need
            the only problem is that some vertices in this support will be uncorrectly classified as inner/outer, so we need to pass initial=true flag to fix this case*/
            for(auto v:support) {
                if(v == ddfsResult.second) continue; //skip bud
                myBridge[v] = curBridge;
                bud.linkTo(v,ddfsResult.second);

                //this part of code is only needed when bottleneck found, but it doesn't mess up anything when called on two paths 
                setLvl(v,2*i+1-minlvl(v));
                for(auto f : graph[v])
                    if(evenlvl[v] > oddlvl[v] && f.type == Bridge && tenacity({v,f.to}) < INF && mate[v] != f.to)
                        bridges[tenacity({v,f.to})].push_back({v,f.to});
            }

            if(ddfsResult.first != ddfsResult.second) {
                augumentPath(ddfsResult.first,ddfsResult.second,true);
                foundPath = true;
                while(!removedVerticesQueue.empty()) {
                    int v = removedVerticesQueue.front();
                    removedVerticesQueue.pop();
                    for(auto e : graph[v])
                        if(e.type == Prop && minlvl(e.to) > minlvl(v) && !removed[e.to] && ++removedPredecessorsSize[e.to] == predecessors[e.to].size())
                            removeAndPushToQueue(e.to);
                }
            }
        }
    }
    return foundPath;
}


bool bfsSorted()
{
  vector<vector<int>> verticesAtLevel(n);
  vector<vector<pii>> bridges(2 * n + 2);
  vector<int> removedPredecessorsSize(n);
  vector<int> removedPredecessorsSize2(n);
  childsInDDFSTreePtr.clear();
  childsInDDFSTreePtr.resize(n);

  auto setLvl = [&](int u, int lev)
  {
    if (lev & 1)
      oddlvl[u] = lev;
    else
      evenlvl[u] = lev;
    //verticesAtLevel[lev].push_back(u);
    insert_sorted(verticesAtLevel[lev],u);
  };

  for (int u = 0; u < n; u++)
    if (mate[u] == -1)
      setLvl(u, 0);

  bool foundPath = false;
  for (int i = 0; i < n && !foundPath; i++)
  {
    // This can be performed in parallel
    // each edge e only belongs to one vertex in the frontier.
    // The only race condition is if two outgoing edges in the frontier point to the same vertex v.
    // Then it is possible they would both add themselves to v's predecessors.
    // Not sure if this is a problem.
    // If it is, it can be solved by claiming vertices atomically.

    // Since the level is at most reduced to i+1, there are no race conditions due to
    // order of operations between the if(lvl>=i+1) and else

    // Note: this is a ms-bfs approach since all vertices at the level are grown.
    for (auto u : verticesAtLevel[i])
    {
      for (auto &e : graph[u])
      {
        if (e.type == NotScanned && (oddlvl[u] == i) == (mate[u] == e.to))
        {
          if (minlvl(e.to) >= i + 1)
          {
            e.type = Prop;
            graph[e.to][e.other].type = Prop;

            if (minlvl(e.to) > i + 1)
              setLvl(e.to, i + 1);
            predecessors[e.to].push_back(u);
          }
          else
          {
            e.type = Bridge;
            graph[e.to][e.other].type = Bridge;
            if (tenacity({u, e.to}) < INF)
            {
              bridges[tenacity({u, e.to})].push_back({u, e.to});
            }
          }
        }
      }
    }

    

    // This loop should be parallelized using a dynamic worklist.  Bridges should be popped of the stack
    // and ddfs performed using a copy of the graph.  The race condition lies in the bud array.
    // The way to extract parallelism is to check if

    // removed, bud, predecessors, oddlvl, and evenlvl are constant during a ddfs call.
    // color,stack1,stack2,support,childsInDDFSTree are not, and will be private to a ddfs call.
    // int nthreads, tid;
     // #pragma omp parallel for\
        shared(removed,bud,predecessors,oddlvl,evenlvl,i,removedVerticesQueue)\
        firstprivate(globalColorCounter ,color, childsInDDFSTree, ddfsPredecessorsPtr)\
        private(nthreads, tid)
    for (auto b : bridges[2 * i + 1])
    {
      if (removed[bud[b.st]] || removed[bud[b.nd]])
        continue;
      vector<int> support;

      // Race conditions exist if some threads are contracting bud's
      auto ddfsResult = ddfs(b, support);
      /*
      tid = omp_get_thread_num();
      nthreads = omp_get_num_threads();
      std::string str=join(support.begin(), support.end(), std::string(","));
      vector<int> supportmate;
      for (int k = 0; k < support.size();++k)
          supportmate.push_back(mate[support[k]]);
      std::string str2=join(supportmate.begin(), supportmate.end(), std::string(","));
      printf("Hello World Thread %d / %d %s %s %s; %s %s %s\n", tid, nthreads, std::to_string(ddfsResult.first).c_str(),std::to_string(ddfsResult.second).c_str(),str.c_str(),std::to_string(mate[ddfsResult.first]).c_str(),std::to_string(mate[ddfsResult.second]).c_str(),str2.c_str());
      */
      pair<pii, pii> curBridge = {b, {bud[b.st], bud[b.nd]}};
      /*even when we found two disjoint paths, we create fake petal, with bud in the end of second path
      the support of this bridge will be these two pathes and some other vertices, which have bases on this paths, so we will remove them and this will not affect corectness
      using this trick, we can simply call augumentPath on these two ends - the first end is just above fake bud, so it will augument exactly the path we need
      the only problem is that some vertices in this support will be uncorrectly classified as inner/outer, so we need to pass initial=true flag to fix this case*/
      for (auto v : support)
      {
        if (v == ddfsResult.second)
          continue; // skip bud
        myBridge[v] = curBridge;
        bud.linkTo(v, ddfsResult.second);

        // this part of code is only needed when bottleneck found, but it doesn't mess up anything when called on two paths
        setLvl(v, 2 * i + 1 - minlvl(v));
        for (auto f : graph[v])
          if (evenlvl[v] > oddlvl[v] && f.type == Bridge && tenacity({v, f.to}) < INF && mate[v] != f.to)
            bridges[tenacity({v, f.to})].push_back({v, f.to});
      }
      
      /*
      mate2 = mate;
      assert(mate2 == mate);
      removed2 = removed;
      removedVerticesQueue2 = removedVerticesQueue;
      */
      if (ddfsResult.first != ddfsResult.second)
      {
        augumentPath(ddfsResult.first, ddfsResult.second, true);
        foundPath = true;
        while (!removedVerticesQueue.empty())
        {
          int v = removedVerticesQueue.front();
          removedVerticesQueue.pop();
          for (auto e : graph[v])
            if (e.type == Prop && minlvl(e.to) > minlvl(v) && !removed[e.to] && ++removedPredecessorsSize[e.to] == predecessors[e.to].size())
              removeAndPushToQueue(e.to);
        }
      }
      /*
      childsInDDFSTreePtr.clear();
      childsInDDFSTreePtr.resize(n,0);

      if (ddfsResult.first != ddfsResult.second)
      {
        //augumentPath2(ddfsResult.first, ddfsResult.second, true);
        augumentPathStack(ddfsResult.first, ddfsResult.second, true);
        foundPath = true;
        while (!removedVerticesQueue2.empty())
        {
          int v = removedVerticesQueue2.front();
          removedVerticesQueue2.pop();
          for (auto e : graph[v])
            if (e.type == Prop && minlvl(e.to) > minlvl(v) && !removed2[e.to] && ++removedPredecessorsSize2[e.to] == predecessors[e.to].size())
              removeAndPushToQueue2(e.to);
        }
      }
      //for (int t = 0; t < mate.size(); ++t)
      //  printf("%d %d %d\n", t, mate[t],mate2[t]);
      assert(mate2 == mate);
      assert(removed2 == removed);
      assert(removedVerticesQueue2 == removedVerticesQueue);
      */
    }
  }
  return foundPath;
}


void bfsIterationVectors(std::vector<std::vector<int>> & verticesAtLevel, 
                        vector<vector<pii>> & bridges,
                        int i){

    auto setLvl = [&](int u, int lev)
    {
      if (lev & 1)
        oddlvl[u] = lev;
      else
        evenlvl[u] = lev;
      //verticesAtLevel[lev].push_back(u);
      insert_sorted(verticesAtLevel[lev],u);
    };
    // Note: this is a ms-bfs approach since all vertices at the level are grown.
    for (auto u : verticesAtLevel[i])
    {
      for (auto &e : graph[u])
      {
        if (e.type == NotScanned && (oddlvl[u] == i) == (mate[u] == e.to))
        {
          if (minlvl(e.to) >= i + 1)
          {
            e.type = Prop;
            graph[e.to][e.other].type = Prop;
            if (minlvl(e.to) > i + 1)
              setLvl(e.to, i + 1);
            //predecessors[e.to].push_back(u);
            insert_sorted(predecessors[e.to],u);
          }
          else
          {
            e.type = Bridge;
            graph[e.to][e.other].type = Bridge;
            if (tenacity({u, e.to}) < INF)
            {
              //bridges[tenacity({u, e.to})].push_back({u, e.to});
              insert_sorted(bridges[tenacity({u, e.to})],{u, e.to});

            }
          }
        }
      }
    }
}

void bfsIterationArraysHost(CSRGraph & csr,
                        int i){
    auto getLvl = [&](int u, int lev)
    {
      if(lev%2) return csr.oddlvl_h[u]; else return csr.evenlvl_h[u];
    };

    auto setLvl = [&](int u, int lev)
    {
      if (lev & 1)
        csr.oddlvl_h[u] = lev;
      else
        csr.evenlvl_h[u] = lev;
      //verticesAtLevel[lev].push_back(u);
    };

    auto tenacity = [&](int u, int v)
    {
    if (csr.mate_h[u] == v)
        return csr.oddlvl_h[u] + csr.oddlvl_h[v] + 1;
      return csr.evenlvl_h[u] + csr.evenlvl_h[v] + 1;
    };

    for (int vertex = 0; vertex < n; vertex++){
        if(i != getLvl(vertex, i)) continue;

        unsigned int start = csr.offsets_h[vertex];
        unsigned int end = csr.offsets_h[vertex + 1];
        unsigned int edgeIndex=start;
        for(; edgeIndex < end; edgeIndex++) { // Delete Neighbors of startingVertex
            
            //printf("depth %d  minlvl(%d)=%d src %d dst %d start %d end %d edgeIndex %d edgeStatus %d evenlvl %d oddlvl %d matching %d\n",i, csr.cols_h[edgeIndex],minlvl(csr.col_h[edgeIndex]),vertex,csr.col_h[edgeIndex],start,end,edgeIndex,graph.edgeStatus[edgeIndex],graph.evenlvl[vertex],graph.oddlvl[vertex],graph.matching[vertex]);
            if (csr.edgeStatus_h[edgeIndex] == NotScanned && (csr.oddlvl_h[vertex] == i) == (csr.mate_h[vertex] == csr.cols_h[edgeIndex])) {
                if(minlvl(csr.cols_h[edgeIndex]) >= i+1) {
                    csr.edgeStatus_h[edgeIndex] = Prop;
                    unsigned int startOther = csr.offsets_h[csr.cols_h[edgeIndex]];
                    unsigned int endOther = csr.offsets_h[csr.cols_h[edgeIndex] + 1];
                    unsigned int edgeIndexOther;
                    //printf("BID %d prop edge %d - %d \n", blockIdx.x, vertex, graph.dst[edgeIndex]);
                    for(edgeIndexOther=startOther; edgeIndexOther < endOther; edgeIndexOther++) { // Delete Neighbors of startingVertex
                        if(vertex==csr.cols_h[edgeIndexOther]){
                            csr.edgeStatus_h[edgeIndexOther] = Prop;
                            break;
                        }
                    }
                    if(minlvl(csr.cols_h[edgeIndex]) > i+1)
                        setLvl(csr.cols_h[edgeIndex], i+1);
                    csr.predecessors_h[edgeIndexOther]=true;
                }
                else{
                    csr.edgeStatus_h[edgeIndex] = Bridge;
                    //printf("BID %d bridge edge %d - %d tenacity %d \n", blockIdx.x, vertex,graph.dst[edgeIndex],tenacity(graph,vertex,graph.dst[edgeIndex]));
                    unsigned int startOther = csr.offsets_h[csr.cols_h[edgeIndex]];
                    unsigned int endOther = csr.offsets_h[csr.cols_h[edgeIndex] + 1];
                    unsigned int edgeIndexOther;
                    for(edgeIndexOther=startOther; edgeIndexOther < endOther; edgeIndexOther++) { // Delete Neighbors of startingVertex
                        if(vertex==csr.cols_h[edgeIndexOther]){
                            csr.edgeStatus_h[edgeIndexOther] = Bridge;
                            break;
                        }
                    }
                    if(tenacity(vertex,csr.cols_h[edgeIndex]) < INF) {
                        csr.bridgeTenacity_h[edgeIndex] = tenacity(vertex,csr.cols_h[edgeIndex]);
                    }
                }
            }
        }
    }
}


__global__ void setSources(
                            int *oddlvl,
                            int *evenlvl,
                            int *mate,
                            int n,
                            int i) {
    auto setLvl = [&](int u, int lev)
    {
      if (lev & 1)
        oddlvl[u] = lev;
      else
        evenlvl[u] = lev;
      //verticesAtLevel[lev].push_back(u);
    };
    unsigned int vertex = threadIdx.x + blockIdx.x*(blockDim.x);
    if(vertex >= n) return;

    if (mate[vertex]==-1){
        setLvl(vertex,0);
    }
}

__global__ void bfsIterationArraysDevice_serial(
                                          int *oddlvl,
                                          int *evenlvl,
                                          int *mate,
                                          unsigned int *offsets,
                                          unsigned int *cols,
                                          char *edgeStatus,
                                          bool *predecessors,
                                          int *bridgeTenacity,
                                          int n,
                                          int i){
    auto getLvl = [&](int u, int lev)
    {
      if(lev & 1) return oddlvl[u]; else return evenlvl[u];
    };

    auto minlvl = [&](int u)
    { return min(evenlvl[u], oddlvl[u]); };

    auto setLvl = [&](int u, int lev)
    {
      if (lev & 1)
        oddlvl[u] = lev;
      else
        evenlvl[u] = lev;
      //verticesAtLevel[lev].push_back(u);
    };

    auto tenacity = [&](int u, int v)
    {
    if (mate[u] == v)
        return oddlvl[u] + oddlvl[v] + 1;
      return evenlvl[u] + evenlvl[v] + 1;
    };
    int iter = 0;

    for (int vertex = 0; vertex < n; vertex++){
        // Race conditions could prevent parallel vertices from continuing.
        // The vector approach first extracts vertices which are at the level
        // then operates on all them regardless.
        if(i != getLvl(vertex, i)) continue;

        unsigned int start = offsets[vertex];
        unsigned int end = offsets[vertex + 1];
        unsigned int edgeIndex=start;
        for(; edgeIndex < end; edgeIndex++) {
            ++iter;
             // Delete Neighbors of startingVertex
            //printf("edgeStatus[edgeIndex]%d == NotScanned %d && evenlvl[vertex]%d (oddlvl[vertex]%d == i %d) == (mate[vertex] %d == cols[edgeIndex] %d)\n",
            //edgeStatus[edgeIndex],NotScanned,evenlvl[vertex],oddlvl[vertex],i,mate[vertex],cols[edgeIndex]);
            if (edgeStatus[edgeIndex] == NotScanned && (oddlvl[vertex] == i) == (mate[vertex] == cols[edgeIndex])) {
                if(minlvl(cols[edgeIndex]) >= i+1) {
                    edgeStatus[edgeIndex] = Prop;
                    unsigned int startOther = offsets[cols[edgeIndex]];
                    unsigned int endOther = offsets[cols[edgeIndex] + 1];
                    unsigned int edgeIndexOther;
                    //printf("prop edge %d - %d \n", vertex, cols[edgeIndex]);
                    for(edgeIndexOther=startOther; edgeIndexOther < endOther; edgeIndexOther++) { // Delete Neighbors of startingVertex
                        if(vertex==cols[edgeIndexOther]){
                            edgeStatus[edgeIndexOther] = Prop;
                            break;
                        }
                    }
                    if(minlvl(cols[edgeIndex]) > i+1)
                        setLvl(cols[edgeIndex], i+1);
                    predecessors[edgeIndexOther]=true;
                }
                else{
                    edgeStatus[edgeIndex] = Bridge;
                    //printf("bridge edge %d - %d tenacity %d \n", vertex,cols[edgeIndex],tenacity(vertex,cols[edgeIndex]));
                    unsigned int startOther = offsets[cols[edgeIndex]];
                    unsigned int endOther = offsets[cols[edgeIndex] + 1];
                    unsigned int edgeIndexOther;
                    for(edgeIndexOther=startOther; edgeIndexOther < endOther; edgeIndexOther++) { // Delete Neighbors of startingVertex
                        if(vertex==cols[edgeIndexOther]){
                            edgeStatus[edgeIndexOther] = Bridge;
                            break;
                        }
                    }
                    if(tenacity(vertex,cols[edgeIndex]) < INF) {
                        bridgeTenacity[edgeIndex] = tenacity(vertex,cols[edgeIndex]);
                    }
                }
            } 
        }
    }
}


__global__ void bfsIterationArraysDevice_serial(
                                          int *verticesInLevel,
                                          int *oddlvl,
                                          int *evenlvl,
                                          int *mate,
                                          unsigned int *offsets,
                                          unsigned int *cols,
                                          char *edgeStatus,
                                          bool *predecessors,
                                          unsigned int *predecessor_count,
                                          int *bridgeTenacity,
                                          int n,
                                          int i){
    auto getLvl = [&](int u, int lev)
    {
      if(lev & 1) return oddlvl[u]; else return evenlvl[u];
    };

    auto minlvl = [&](int u)
    { return min(evenlvl[u], oddlvl[u]); };

    auto setLvl = [&](int u, int lev)
    {
      if (lev & 1)
        oddlvl[u] = lev;
      else
        evenlvl[u] = lev;
      //verticesAtLevel[lev].push_back(u);
    };

    auto tenacity = [&](int u, int v)
    {
    if (mate[u] == v)
        return oddlvl[u] + oddlvl[v] + 1;
      return evenlvl[u] + evenlvl[v] + 1;
    };
    int iter = 0;

    for (int vertexIndex = 0; vertexIndex < n; vertexIndex++){
        int vertex = verticesInLevel[vertexIndex];
        // Race conditions could prevent parallel vertices from continuing.
        // The vector approach first extracts vertices which are at the level
        // then operates on all them regardless.
        if(i != getLvl(vertex, i)) continue;

        unsigned int start = offsets[vertex];
        unsigned int end = offsets[vertex + 1];
        unsigned int edgeIndex=start;
        for(; edgeIndex < end; edgeIndex++) {
            ++iter;
             // Delete Neighbors of startingVertex
            //printf("edgeStatus[edgeIndex]%d == NotScanned %d && evenlvl[vertex]%d (oddlvl[vertex]%d == i %d) == (mate[vertex] %d == cols[edgeIndex] %d)\n",
            //edgeStatus[edgeIndex],NotScanned,evenlvl[vertex],oddlvl[vertex],i,mate[vertex],cols[edgeIndex]);
            if (edgeStatus[edgeIndex] == NotScanned && (oddlvl[vertex] == i) == (mate[vertex] == cols[edgeIndex])) {
                if(minlvl(cols[edgeIndex]) >= i+1) {
                    edgeStatus[edgeIndex] = Prop;
                    unsigned int startOther = offsets[cols[edgeIndex]];
                    unsigned int endOther = offsets[cols[edgeIndex] + 1];
                    unsigned int edgeIndexOther;
                    //printf("prop edge %d - %d \n", vertex, cols[edgeIndex]);
                    for(edgeIndexOther=startOther; edgeIndexOther < endOther; edgeIndexOther++) { // Delete Neighbors of startingVertex
                        if(vertex==cols[edgeIndexOther]){
                            edgeStatus[edgeIndexOther] = Prop;
                            break;
                        }
                    }
                    if(minlvl(cols[edgeIndex]) > i+1)
                        setLvl(cols[edgeIndex], i+1);
                    predecessors[edgeIndexOther]=true;
                    atomicAdd(&predecessor_count[cols[edgeIndex]],(unsigned int)1);
                }
                else{
                    edgeStatus[edgeIndex] = Bridge;
                    //printf("bridge edge %d - %d tenacity %d \n", vertex,cols[edgeIndex],tenacity(vertex,cols[edgeIndex]));
                    unsigned int startOther = offsets[cols[edgeIndex]];
                    unsigned int endOther = offsets[cols[edgeIndex] + 1];
                    unsigned int edgeIndexOther;
                    for(edgeIndexOther=startOther; edgeIndexOther < endOther; edgeIndexOther++) { // Delete Neighbors of startingVertex
                        if(vertex==cols[edgeIndexOther]){
                            edgeStatus[edgeIndexOther] = Bridge;
                            break;
                        }
                    }
                    if(tenacity(vertex,cols[edgeIndex]) < INF) {
                        bridgeTenacity[edgeIndex] = tenacity(vertex,cols[edgeIndex]);
                    }
                }
            } 
        }
    }
}


__global__ void bfsIterationArraysDevice_parallel(
                                          int *verticesInLevel,
                                          int *oddlvl,
                                          int *evenlvl,
                                          int *mate,
                                          unsigned int *offsets,
                                          unsigned int *cols,
                                          char *edgeStatus,
                                          bool *predecessors,
                                          unsigned int *predecessor_count,
                                          int *bridgeTenacity,
                                          unsigned int *n,
                                          unsigned int *blc,
                                          bool * nonempty,
                                          int i,
                                          int max){

    auto minlvl = [&](int u)
    { return min(evenlvl[u], oddlvl[u]); };

    auto setLvl = [&](int u, int lev)
    {
      if (lev & 1)
        oddlvl[u] = lev;
      else
        evenlvl[u] = lev;
      //verticesAtLevel[lev].push_back(u);
    };

    auto tenacity = [&](int u, int v)
    {
    if (mate[u] == v)
        return oddlvl[u] + oddlvl[v] + 1;
      return evenlvl[u] + evenlvl[v] + 1;
    };
    int iter = 0;

    // start
    int vertexIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(vertexIndex >= n[0]) return;
    if (vertexIndex==0) blc[0]=0;
    int vertex = verticesInLevel[vertexIndex];
    
    // Race conditions could prevent parallel vertices from continuing.
    // The vector approach first extracts vertices which are at the level
    // then operates on all them regardless.


    unsigned int start = offsets[vertex];
    unsigned int end = offsets[vertex + 1];
    unsigned int edgeIndex=start;
    for(; edgeIndex < end; edgeIndex++) {
        ++iter;
          // Delete Neighbors of startingVertex
        //printf("edgeStatus[edgeIndex]%d == NotScanned %d && evenlvl[vertex]%d (oddlvl[vertex]%d == i %d) == (mate[vertex] %d == cols[edgeIndex] %d)\n",
        //edgeStatus[edgeIndex],NotScanned,evenlvl[vertex],oddlvl[vertex],i,mate[vertex],cols[edgeIndex]);
        if (edgeStatus[edgeIndex] == NotScanned && (oddlvl[vertex] == i) == (mate[vertex] == cols[edgeIndex])) {
            if(minlvl(cols[edgeIndex]) >= i+1) {
                edgeStatus[edgeIndex] = Prop;
                unsigned int startOther = offsets[cols[edgeIndex]];
                unsigned int endOther = offsets[cols[edgeIndex] + 1];
                unsigned int edgeIndexOther;
                //printf("prop edge %d - %d \n", vertex, cols[edgeIndex]);
                for(edgeIndexOther=startOther; edgeIndexOther < endOther; edgeIndexOther++) { // Delete Neighbors of startingVertex
                    if(vertex==cols[edgeIndexOther]){
                        edgeStatus[edgeIndexOther] = Prop;
                        break;
                    }
                }
                if(minlvl(cols[edgeIndex]) > i+1){
                  if ((i+1)<max)
                    nonempty[i+1]=true;
                  setLvl(cols[edgeIndex], i+1);
                }
                predecessors[edgeIndexOther]=true;
                atomicAdd(&predecessor_count[cols[edgeIndex]],(unsigned int)1);
            }
            else{
                edgeStatus[edgeIndex] = Bridge;
                //printf("bridge edge %d - %d tenacity %d \n", vertex,cols[edgeIndex],tenacity(vertex,cols[edgeIndex]));
                unsigned int startOther = offsets[cols[edgeIndex]];
                unsigned int endOther = offsets[cols[edgeIndex] + 1];
                unsigned int edgeIndexOther;
                for(edgeIndexOther=startOther; edgeIndexOther < endOther; edgeIndexOther++) { // Delete Neighbors of startingVertex
                    if(vertex==cols[edgeIndexOther]){
                        edgeStatus[edgeIndexOther] = Bridge;
                        break;
                    }
                }
                if(tenacity(vertex,cols[edgeIndex]) < INF) {
                    bridgeTenacity[edgeIndex] = tenacity(vertex,cols[edgeIndex]);
                }
            }
        } 
    }
}

__global__ void updateLvlAndTenacityPassStruct_serial(
                                          DSU_CU bud,
                                          updateLvlStruct updateLvl,
                                          bool *removed,
                                          int i,
                                          int u,
                                          int v){
    if (removed[bud[u]] || removed[bud[v]])
      return;
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
    for (int vertexIndex = 0; vertexIndex < n; vertexIndex++){
        int vertex = updateLvl.support[vertexIndex];
        if (vertex == updateLvl.ddfsResult[1])
          continue; // skip bud
        //myBridge[vertex] = curBridge;
        updateLvl.myBridge_a[vertex]=updateLvl.curBridge[0];
        updateLvl.myBridge_b[vertex]=updateLvl.curBridge[1];
        updateLvl.myBridge_c[vertex]=updateLvl.curBridge[2];
        updateLvl.myBridge_d[vertex]=updateLvl.curBridge[3];
        
        bud.linkTo(vertex, updateLvl.ddfsResult[1]);
        
        // this part of code is only needed when bottleneck found, but it doesn't mess up anything when called on two paths
        setLvl(vertex, 2 * i + 1 - minlvl(vertex));
        unsigned int start = updateLvl.offsets[vertex];
        unsigned int end = updateLvl.offsets[vertex + 1];
        unsigned int edgeIndex=start;
        for(; edgeIndex < end; edgeIndex++) {
        if(updateLvl.evenlvl[vertex] > updateLvl.oddlvl[vertex] && updateLvl.edgeStatus[edgeIndex] == Bridge && 
            tenacity(vertex,updateLvl.cols[edgeIndex]) < INF && updateLvl.mate[vertex] != updateLvl.cols[edgeIndex]) {
            
            updateLvl.bridgeTenacity[edgeIndex] = tenacity(vertex,updateLvl.cols[edgeIndex]);
          }
        }
    }
}


__device__ void updateLvlAndTenacityPassStruct_serialDev(
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
    for (int vertexIndex = 0; vertexIndex < n; vertexIndex++){
        int vertex = updateLvl.support[vertexIndex];
        if (vertex == updateLvl.ddfsResult[1])
          continue; // skip bud
        //myBridge[vertex] = curBridge;
        updateLvl.myBridge_a[vertex]=updateLvl.curBridge[0];
        updateLvl.myBridge_b[vertex]=updateLvl.curBridge[1];
        updateLvl.myBridge_c[vertex]=updateLvl.curBridge[2];
        updateLvl.myBridge_d[vertex]=updateLvl.curBridge[3];
        
        bud.linkTo(vertex, updateLvl.ddfsResult[1]);
        
        // this part of code is only needed when bottleneck found, but it doesn't mess up anything when called on two paths
        setLvl(vertex, 2 * i + 1 - minlvl(vertex));
        if (2 * i + 1 - minlvl(vertex) < updateLvl.n[0])
          updateLvl.nonEmpty[2 * i + 1 - minlvl(vertex)]=true;
        unsigned int start = updateLvl.offsets[vertex];
        unsigned int end = updateLvl.offsets[vertex + 1];
        unsigned int edgeIndex=start;
        for(; edgeIndex < end; edgeIndex++) {
        if(updateLvl.evenlvl[vertex] > updateLvl.oddlvl[vertex] && updateLvl.edgeStatus[edgeIndex] == Bridge && 
            tenacity(vertex,updateLvl.cols[edgeIndex]) < INF && updateLvl.mate[vertex] != updateLvl.cols[edgeIndex]) {
            
            updateLvl.bridgeTenacity[edgeIndex] = tenacity(vertex,updateLvl.cols[edgeIndex]);
          }
        }
    }
}


__global__ void updateLvlAndTenacity_serial(
                                          unsigned int *support,
                                          int *oddlvl,
                                          int *evenlvl,
                                          int *mate,
                                          unsigned int *offsets,
                                          unsigned int *cols,
                                          char *edgeStatus,
                                          int *bridgeTenacity,
                                          int *myBridge_a,
                                          int *myBridge_b,
                                          int *myBridge_c,
                                          int *myBridge_d,
                                          DSU_CU bud,
                                          pii ddfsResult,
                                          pair<pii, pii> curBridge,
                                          unsigned int *size,
                                          int i){
    int n = size[0];

    auto minlvl = [&](int u)
    { return min(evenlvl[u], oddlvl[u]); };

    auto setLvl = [&](int u, int lev)
    {
      if (lev & 1)
        oddlvl[u] = lev;
      else
        evenlvl[u] = lev;
      //verticesAtLevel[lev].push_back(u);
    };

    auto tenacity = [&](int u, int v)
    {
    if (mate[u] == v)
        return oddlvl[u] + oddlvl[v] + 1;
      return evenlvl[u] + evenlvl[v] + 1;
    };
    for (int vertexIndex = 0; vertexIndex < n; vertexIndex++){
        int vertex = support[vertexIndex];
        if (vertex == ddfsResult.second)
          continue; // skip bud
        //myBridge[vertex] = curBridge;
        myBridge_a[vertex]=curBridge.first.first;
        myBridge_b[vertex]=curBridge.first.second;
        myBridge_c[vertex]=curBridge.second.first;
        myBridge_d[vertex]=curBridge.second.second;
        
        bud.linkTo(vertex, ddfsResult.second);
        
        // this part of code is only needed when bottleneck found, but it doesn't mess up anything when called on two paths
        setLvl(vertex, 2 * i + 1 - minlvl(vertex));
        unsigned int start = offsets[vertex];
        unsigned int end = offsets[vertex + 1];
        unsigned int edgeIndex=start;
        for(; edgeIndex < end; edgeIndex++) {
        if(evenlvl[vertex] > oddlvl[vertex] && edgeStatus[edgeIndex] == Bridge && 
            tenacity(vertex,cols[edgeIndex]) < INF && mate[vertex] != cols[edgeIndex]) {
            
            bridgeTenacity[edgeIndex] = tenacity(vertex,cols[edgeIndex]);
          }
        }
    }
}


__device__ void updateLvlAndTenacity_serialDev(
                                          unsigned int *support,
                                          int *oddlvl,
                                          int *evenlvl,
                                          int *mate,
                                          unsigned int *offsets,
                                          unsigned int *cols,
                                          char *edgeStatus,
                                          int *bridgeTenacity,
                                          int *myBridge_a,
                                          int *myBridge_b,
                                          int *myBridge_c,
                                          int *myBridge_d,
                                          DSU_CU bud,
                                          unsigned int * ddfsResult,
                                          unsigned int * curBridge,
                                          unsigned int *size,
                                          int i){
    int n = size[0];

    auto minlvl = [&](int u)
    { return min(evenlvl[u], oddlvl[u]); };

    auto setLvl = [&](int u, int lev)
    {
      if (lev & 1)
        oddlvl[u] = lev;
      else
        evenlvl[u] = lev;
      //verticesAtLevel[lev].push_back(u);
    };

    auto tenacity = [&](int u, int v)
    {
    if (mate[u] == v)
        return oddlvl[u] + oddlvl[v] + 1;
      return evenlvl[u] + evenlvl[v] + 1;
    };
    for (int vertexIndex = 0; vertexIndex < n; vertexIndex++){
        int vertex = support[vertexIndex];
        if (vertex == ddfsResult[1])
          continue; // skip bud
        //myBridge[vertex] = curBridge;
        myBridge_a[vertex]=curBridge[0];
        myBridge_b[vertex]=curBridge[1];
        myBridge_c[vertex]=curBridge[2];
        myBridge_d[vertex]=curBridge[3];
        
        bud.linkTo(vertex, ddfsResult[1]);
        
        // this part of code is only needed when bottleneck found, but it doesn't mess up anything when called on two paths
        setLvl(vertex, 2 * i + 1 - minlvl(vertex));
        unsigned int start = offsets[vertex];
        unsigned int end = offsets[vertex + 1];
        unsigned int edgeIndex=start;
        for(; edgeIndex < end; edgeIndex++) {
        if(evenlvl[vertex] > oddlvl[vertex] && edgeStatus[edgeIndex] == Bridge && 
            tenacity(vertex,cols[edgeIndex]) < INF && mate[vertex] != cols[edgeIndex]) {
            
            bridgeTenacity[edgeIndex] = tenacity(vertex,cols[edgeIndex]);
          }
        }
    }
}



__global__ void extract_bridges_parallel(
                                          int *oddlvl,
                                          int *evenlvl,
                                          int *mate,
                                          unsigned int *offsets,
                                          unsigned int *cols,
                                          char *edgeStatus,
                                          int *bridgeTenacity,
                                          unsigned int *bridgeList_counter,
                                          unsigned int *bridgeList,
                                          //unsigned int *bridgeList_u,
                                          //unsigned int* bridgeList_v,
                                          int n,
                                          int depth){

    unsigned int vertex = threadIdx.x + blockIdx.x*(blockDim.x);
    if(vertex >= n) return;


    unsigned int start = offsets[vertex];
    unsigned int end = offsets[vertex + 1];
    unsigned int edgeIndex=start;
    for(; edgeIndex < end; edgeIndex++) { // Delete Neighbors of startingVertex
        if (edgeStatus[edgeIndex] == Bridge && bridgeTenacity[edgeIndex] == (2*depth)+1) {
            unsigned int top = atomicAdd(bridgeList_counter,1);
            //uint64_t edgePair = (uint64_t) vertex << 32 | dst[edgeIndex];
                //printf("Adding bridge %d %d ten %d 2*depth+1 %d top %d\n", vertex, dst[edgeIndex],bridgeTenacity[edgeIndex],2*depth+1,top);
            //bridgeList_u[top] = vertex;
            //bridgeList_v[top] = cols[edgeIndex];
            bridgeList[top] = edgeIndex;
        }
    }
}


__global__ void extract_vertices_in_level(
                                          int *lvlarray,
                                          int *verticesInLevelList,
                                          unsigned int *verticesInLevelList_counter,
                                          bool * nonEmpty,
                                          int n,
                                          int depth){

    unsigned int vertex = threadIdx.x + blockIdx.x*(blockDim.x);
    if(vertex >= n) return;
    if (vertex==0) nonEmpty[depth]=false;
    if(lvlarray[vertex]==depth){
        unsigned int top = atomicAdd(verticesInLevelList_counter,1);
        //uint64_t edgePair = (uint64_t) vertex << 32 | dst[edgeIndex];
            //printf("Adding bridge %d %d ten %d 2*depth+1 %d top %d\n", vertex, dst[edgeIndex],bridgeTenacity[edgeIndex],2*depth+1,top);
        //bridgeList_u[top] = vertex;
        //bridgeList_v[top] = cols[edgeIndex];
        verticesInLevelList[top] = vertex;
    }
}



__global__ void extract_bridges_parallel_edge_centric(
                                          int *oddlvl,
                                          int *evenlvl,
                                          int *mate,
                                          unsigned int *offsets,
                                          unsigned int *cols,
                                          char *edgeStatus,
                                          int *bridgeTenacity,
                                          unsigned int *bridgeList_counter,
                                          unsigned int *bridgeList,
                                          //unsigned int *bridgeList_u,
                                          //unsigned int* bridgeList_v,
                                          int m,
                                          int depth){

    unsigned int edgeIndex = threadIdx.x + blockIdx.x*(blockDim.x);
    if(edgeIndex >= m) return;
    if (edgeStatus[edgeIndex] == Bridge && bridgeTenacity[edgeIndex] == (2*depth)+1) {
        unsigned int top = atomicAdd(bridgeList_counter,1);
        //uint64_t edgePair = (uint64_t) vertex << 32 | dst[edgeIndex];
            //printf("Adding bridge %d %d ten %d 2*depth+1 %d top %d\n", vertex, dst[edgeIndex],bridgeTenacity[edgeIndex],2*depth+1,top);
        //bridgeList_u[top] = vertex;
        //bridgeList_v[top] = cols[edgeIndex];
        bridgeList[top] = edgeIndex;
    }
}


__global__ void BU_first_kernel(          DSU_CU bud,
                                          unsigned int *offsets,
                                          unsigned int *rows,
                                          unsigned int *cols,
                                          int *mate,
                                          bool *predecessors,
                                          unsigned int *bridgeList_counter,
                                          unsigned int *bridgeList,
                                          unsigned int *bu_bfs_key,
                                          unsigned int* bu_bfs_val,
                                          unsigned long long int* bu_bfs_top,
                                          unsigned int *bu_bfs_key_root,
                                          unsigned int* bu_bfs_val_root,
                                          unsigned long long int* bu_bfs_top_root,
                                          unsigned long long int numBridges,
                                          unsigned long long int maxPairs){
    unsigned int bridgeIndex = blockIdx.x*blockDim.x + threadIdx.x;
    if (bridgeIndex >= numBridges) return;
    unsigned int vertex1 = rows[bridgeList[bridgeIndex]];
    unsigned int vertex2 = cols[bridgeList[bridgeIndex]];
    unsigned int start = offsets[vertex1];
    unsigned int end = offsets[vertex1 + 1];
    unsigned int edgeIndex=start;
    for(; edgeIndex < end; edgeIndex++) {
      if (predecessors[edgeIndex]){
        unsigned int col = bud(cols[edgeIndex]);
        if (mate[col]==-1){
          unsigned long long int top = atomicAdd(bu_bfs_top_root,(unsigned long long int)1);
          assert(top<maxPairs);
          bu_bfs_key_root[top]=bridgeList[bridgeIndex];
          bu_bfs_val_root[top]=col;
        } else {
          unsigned long long int top = atomicAdd(bu_bfs_top,(unsigned long long int)1);
          assert(top<maxPairs);
          bu_bfs_key[top]=bridgeList[bridgeIndex];
          bu_bfs_val[top]=col;
        }
      }
    }
    start = offsets[vertex2];
    end = offsets[vertex2 + 1];
    edgeIndex=start;
    for(; edgeIndex < end; edgeIndex++) {
      if (predecessors[edgeIndex]){
        unsigned int col = bud(cols[edgeIndex]);
        if (mate[col]==-1){
          unsigned long long int top = atomicAdd(bu_bfs_top_root,(unsigned long long int)1);
          assert(top<maxPairs);
          bu_bfs_key_root[top]=bridgeList[bridgeIndex];
          bu_bfs_val_root[top]=col;
        } else {
          unsigned long long int top = atomicAdd(bu_bfs_top,(unsigned long long int)1);
          assert(top<maxPairs);
          bu_bfs_key[top]=bridgeList[bridgeIndex];
          bu_bfs_val[top]=col;
        }
      }
    }
}


__global__ void BU_kernel(                DSU_CU bud,
                                          unsigned int *offsets,
                                          unsigned int *rows,
                                          unsigned int *cols,
                                          int *mate,
                                          bool *predecessors,
                                          unsigned int *bu_bfs_key,
                                          unsigned int* bu_bfs_val,
                                          unsigned long long int* bu_bfs_top,
                                          unsigned int *bu_bfs_key_buffer,
                                          unsigned int* bu_bfs_val_buffer,
                                          unsigned long long int* bu_bfs_top_buffer,
                                          unsigned int *bu_bfs_key_root,
                                          unsigned int* bu_bfs_val_root,
                                          unsigned long long int* bu_bfs_top_root,
                                          unsigned long long int numPairs,
                                          unsigned long long int maxPairs){
    unsigned int bridgeIndex = blockIdx.x*blockDim.x + threadIdx.x;
    if (bridgeIndex >= numPairs) return;
    unsigned int key = bu_bfs_key[bridgeIndex];
    unsigned int vertex = bu_bfs_val[bridgeIndex];
    unsigned int start = offsets[vertex];
    unsigned int end = offsets[vertex + 1];
    unsigned int edgeIndex=start;
    for(; edgeIndex < end; edgeIndex++) {
      if (predecessors[edgeIndex]){
        unsigned int col = bud(cols[edgeIndex]);
        if (mate[col]==-1){
          unsigned long long int top = atomicAdd(bu_bfs_top_root,(unsigned long long int)1);
          assert(top<maxPairs);
          bu_bfs_key_root[top]=key;
          bu_bfs_val_root[top]=col;
        } else {
          unsigned long long int top = atomicAdd(bu_bfs_top_buffer,(unsigned long long int)1);
          assert(top<maxPairs);
          bu_bfs_key_buffer[top]=key;
          bu_bfs_val_buffer[top]=col;
        }
      }
    }

}


__global__ void BU_first_kernel(          
                                          unsigned int *offsets,
                                          unsigned int *rows,
                                          unsigned int *cols,
                                          int *mate,
                                          bool *predecessors,
                                          unsigned int *bridgeList_counter,
                                          unsigned int *bridgeList,
                                          unsigned int *bu_bfs_key,
                                          unsigned int* bu_bfs_val,
                                          unsigned long long int* bu_bfs_top,
                                          unsigned int *bu_bfs_key_root,
                                          unsigned int* bu_bfs_val_root,
                                          unsigned long long int* bu_bfs_top_root,
                                          unsigned long long int maxPairs){

    unsigned int bridgeIndex = blockIdx.x;
    unsigned int vertex1 = rows[bridgeList[bridgeIndex]];
    unsigned int vertex2 = cols[bridgeList[bridgeIndex]];
    unsigned int start = offsets[vertex1];
    unsigned int end = offsets[vertex1 + 1];
    unsigned int edgeIndex=start;
    for(; edgeIndex < end; edgeIndex++) {
      if (predecessors[edgeIndex]){
        unsigned int col = cols[edgeIndex];
        if (mate[col]==-1){
          unsigned long long int top = atomicAdd(bu_bfs_top_root,(unsigned long long int)1);
          assert(top<maxPairs);
          bu_bfs_key_root[top]=bridgeList[bridgeIndex];
          bu_bfs_val_root[top]=col;
        } else {
          unsigned long long int top = atomicAdd(bu_bfs_top,(unsigned long long int)1);
          assert(top<maxPairs);
          bu_bfs_key[top]=bridgeList[bridgeIndex];
          bu_bfs_val[top]=col;
        }
      }
    }
    start = offsets[vertex2];
    end = offsets[vertex2 + 1];
    edgeIndex=start;
    for(; edgeIndex < end; edgeIndex++) {
      if (predecessors[edgeIndex]){
        unsigned int col = cols[edgeIndex];
        if (mate[col]==-1){
          unsigned long long int top = atomicAdd(bu_bfs_top_root,(unsigned long long int)1);
          assert(top<maxPairs);
          bu_bfs_key_root[top]=bridgeList[bridgeIndex];
          bu_bfs_val_root[top]=col;
        } else {
          unsigned long long int top = atomicAdd(bu_bfs_top,(unsigned long long int)1);
          assert(top<maxPairs);
          bu_bfs_key[top]=bridgeList[bridgeIndex];
          bu_bfs_val[top]=col;
        }
      }
    }
}


__global__ void BU_kernel(                
                                          unsigned int *offsets,
                                          unsigned int *rows,
                                          unsigned int *cols,
                                          int *mate,
                                          bool *predecessors,
                                          unsigned int *bu_bfs_key,
                                          unsigned int* bu_bfs_val,
                                          unsigned long long int* bu_bfs_top,
                                          unsigned int *bu_bfs_key_buffer,
                                          unsigned int* bu_bfs_val_buffer,
                                          unsigned long long int* bu_bfs_top_buffer,
                                          unsigned int *bu_bfs_key_root,
                                          unsigned int* bu_bfs_val_root,
                                          unsigned long long int* bu_bfs_top_root,
                                          unsigned long long int maxPairs){

    unsigned int bridgeIndex = blockIdx.x;
    unsigned int key = bu_bfs_key[bridgeIndex];
    unsigned int vertex = bu_bfs_val[bridgeIndex];
    unsigned int start = offsets[vertex];
    unsigned int end = offsets[vertex + 1];
    unsigned int edgeIndex=start;
    for(; edgeIndex < end; edgeIndex++) {
      if (predecessors[edgeIndex]){
        unsigned int col = cols[edgeIndex];
        if (mate[col]==-1){
          unsigned long long int top = atomicAdd(bu_bfs_top_root,(unsigned long long int)1);
          assert(top<maxPairs);
          bu_bfs_key_root[top]=key;
          bu_bfs_val_root[top]=col;
        } else {
          unsigned long long int top = atomicAdd(bu_bfs_top_buffer,(unsigned long long int)1);
          assert(top<maxPairs);
          bu_bfs_key_buffer[top]=key;
          bu_bfs_val_buffer[top]=col;
        }
      }
    }

}

__global__ void LA_kernel_init(                       
              unsigned int * claimed_A,
              unsigned int * bu_bfs_key_root,
              signed long numUnique){
  unsigned int edgeIndex = threadIdx.x + blockDim.x*blockIdx.x;
  if (edgeIndex >= numUnique) return;
  auto x = bu_bfs_key_root[edgeIndex];
  claimed_A[x]=x;
}


__global__ void LA_kernel(                       
              unsigned int * claimed_A,
              unsigned int * claimed_B,
              unsigned int * bu_bfs_key_root,
              unsigned int * bu_bfs_val_root,
              signed long numUnique,
              bool * claimedNew){
  unsigned int edgeIndex = threadIdx.x + blockDim.x*blockIdx.x;
  if (edgeIndex >= numUnique) return;
  auto x = bu_bfs_key_root[edgeIndex];
  assert(claimed_A[x] != INF);
  //if (claimed_A[x] == INF)
  //  printf("MASSIVE ERROR\n");
  //if (claimed_A[x] == INF) return;
  auto y = bu_bfs_val_root[edgeIndex];
  if (claimed_B[y]!=claimed_A[x]){
    //printf("claimed_A[%d]=%d Claiming claimed_B[%d]=%d\n",x,claimed_A[x],y,claimed_B[y]);
    claimed_B[y]=claimed_A[x];
    claimedNew[0]=true;
  }
}


__global__ void LA_kernelAtomic(                       
              unsigned int * claimed_A,
              unsigned int * smallest_Ever_Seen,
              unsigned int * bu_bfs_key_root,
              unsigned int * bu_bfs_val_root,
              signed long numUnique,
              bool * claimedNew){
  unsigned int edgeIndex = threadIdx.x + blockDim.x*blockIdx.x;
  if (edgeIndex >= numUnique) return;
  auto x = bu_bfs_key_root[edgeIndex];
  assert(claimed_A[x] != INF);
  //if (claimed_A[x] == INF)
  //  printf("MASSIVE ERROR\n");
  //if (claimed_A[x] == INF) return;
  auto y = bu_bfs_val_root[edgeIndex];
  if (claimed_A[x]!=atomicMin(&smallest_Ever_Seen[y],claimed_A[x])){
    //auto ses =atomicMin(&smallest_Ever_Seen[y],smallest_Ever_Seen[x]);
    //printf("claimed_A[%d]=%d Claiming claimed_B[%d]=%d\n",x,claimed_A[x],y,claimed_B[y]);
    claimedNew[0]=true;
  }
}


//returns {r0, g0} or {bottleneck, bottleneck} packed into uint64_t
__global__ void ddfs(DSU_CU bud, 
                      unsigned int *offsets,
                      unsigned int *cols,
                      unsigned int *ddfsPredecessorsPtr,
                      bool *removed, 
                      bool *predecessors, 
                      int *oddlvl,
                      int *evenlvl,
                      unsigned int * stack1, 
                      unsigned int * stack2, 
                      unsigned int * stack1Top, 
                      unsigned int * stack2Top, 
                      unsigned int * support, 
                      unsigned int * supportTop, 
                      unsigned int * color, 
                      unsigned int *globalColorCounter, 
                      int*budAtDDFSEncounter, 
                      unsigned int *ddfsResult, 
                      unsigned int * curBridge,
                      int src, int dst) {
    if (removed[bud[src]] || removed[bud[dst]])
      return;
    stack1Top[0]=0;
    stack2Top[0]=0;
    supportTop[0]=0;

    auto minlvl = [&](int u)
    { return min(evenlvl[u], oddlvl[u]); };
    
    stack1[stack1Top[0]++]=bud[src];
    stack2[stack2Top[0]++]=bud[dst];
    //vector<int> Sr = {bud[src]}, Sg = {bud[dst]};
    //if(Sr[0] == Sg[0])
    //    return {Sr[0],Sg[0]};
    if (stack1[0]==stack2[0]){
        //printf("stack1[0]=%d == stack2[0]=%d\n",stack1[0],stack2[0]);
        //return (uint64_t) stack1[0] << 32 | stack2[0];
        ddfsResult[0]=stack1[0];
        ddfsResult[1]=stack2[0];
        curBridge[0] = src;
        curBridge[1] = dst;
        curBridge[2] = bud[src];
        curBridge[3] = bud[dst];
        return;

    }
    //out_support = {Sr[0], Sg[0]};
    support[supportTop[0]++]=stack1[0];
    support[supportTop[0]++]=stack2[0];

    //int newRed = color[Sr[0]] = ++globalColorCounter, newGreen = color[Sg[0]] = ++globalColorCounter;
    //assert(newRed == (newGreen^1));
    int newRed = color[stack1[0]] = ++globalColorCounter[0], newGreen = color[stack2[0]] = ++globalColorCounter[0];
    assert(newRed == (newGreen^1));
    

    for(;;) {
        //printf("IN FOR\n");
        //if found two disjoint paths
        //if(minlvl(Sr.back()) == 0 && minlvl(Sg.back()) == 0)
        if(minlvl(stack1[stack1Top[0]-1]) == 0 && minlvl(stack2[stack2Top[0]-1]) == 0){

            //printf("stack1[%d]=%d\n",stack1Top[0]-1,stack1[stack1Top[0]-1]);
            //printf("stack2[%d]=%d\n",stack2Top[0]-1,stack2[stack2Top[0]-1]);
            //printf("minlvl(graph,stack1[stack1Top[0]]) == 0 && minlvl(graph,stack2[stack2Top[0]]) == 0\n");
            ddfsResult[0]=stack1[stack1Top[0]-1];
            ddfsResult[1]=stack2[stack2Top[0]-1];
            curBridge[0] = src;
            curBridge[1] = dst;
            curBridge[2] = bud[src];
            curBridge[3] = bud[dst];
            return;
        }
        int b;
        //if(minlvl(Sr.back()) >= minlvl(Sg.back()))
        //printf("stack1[%d-1]=%d ml %d stack2[%d-1]=%d ml %d \n",stack1Top[0],stack1[stack1Top[0]-1],minlvl(stack1[stack1Top[0]-1]),stack2Top[0],stack2[stack2Top[0]-1],minlvl(stack2[stack2Top[0]-1]));
        if(minlvl(stack1[stack1Top[0]-1]) >= minlvl(stack2[stack2Top[0]-1])){
            //printf("ENTERED IF\n");
            b = ddfsMove(bud,offsets,cols,ddfsPredecessorsPtr,removed,predecessors,stack1,stack2,stack1Top,stack2Top,support,supportTop,color,globalColorCounter,budAtDDFSEncounter,newRed, newGreen);
        } else{
            //printf("ENTERED ELSE\n");
            b = ddfsMove(bud,offsets,cols,ddfsPredecessorsPtr,removed,predecessors,stack2,stack1,stack2Top,stack1Top,support,supportTop,color,globalColorCounter,budAtDDFSEncounter,newGreen, newRed);
        }
        if(b != -1){
            //return {b,b};
            //printf("B!=-1\n");
            //return (uint64_t) b << 32 | b;
            ddfsResult[0]=b;
            ddfsResult[1]=b;
            curBridge[0] = src;
            curBridge[1] = dst;
            curBridge[2] = bud[src];
            curBridge[3] = bud[dst];
            return;
        }
    }
}


//returns {r0, g0} or {bottleneck, bottleneck} packed into uint64_t
__global__ void ddfsStructGlobal(DSU_CU bud, 
                                ddfsStruct ddfs,
                                int src, int dst) {
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
            return;
        }
        int b;
        //if(minlvl(Sr.back()) >= minlvl(Sg.back()))
        //printf("stack1[%d-1]=%d ml %d stack2[%d-1]=%d ml %d \n",stack1Top[0],stack1[stack1Top[0]-1],minlvl(stack1[stack1Top[0]-1]),stack2Top[0],stack2[stack2Top[0]-1],minlvl(stack2[stack2Top[0]-1]));
        if(minlvl(ddfs.stack1[ddfs.stack1Top[0]-1]) >= minlvl(ddfs.stack2[ddfs.stack2Top[0]-1])){
            //printf("ENTERED IF\n");
            b = ddfsMove(bud,ddfs.offsets,ddfs.cols,ddfs.ddfsPredecessorsPtr,ddfs.removed,ddfs.predecessors,ddfs.stack1,
                        ddfs.stack2,ddfs.stack1Top,ddfs.stack2Top,ddfs.support,ddfs.supportTop,ddfs.color,ddfs.globalColorCounter,
                        ddfs.budAtDDFSEncounter,newRed, newGreen);
        } else{
            //printf("ENTERED ELSE\n");
            b = ddfsMove(bud,ddfs.offsets,ddfs.cols,ddfs.ddfsPredecessorsPtr,ddfs.removed,ddfs.predecessors,ddfs.stack2,
            ddfs.stack1,ddfs.stack2Top,ddfs.stack1Top,ddfs.support,ddfs.supportTop,ddfs.color,ddfs.globalColorCounter,
            ddfs.budAtDDFSEncounter,newGreen, newRed);
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
            return;
        }
    }
}




__device__ void ddfsDev(DSU_CU bud, 
                      unsigned int *offsets,
                      unsigned int *cols,
                      unsigned int *ddfsPredecessorsPtr,
                      bool *removed, 
                      bool *predecessors, 
                      int *oddlvl,
                      int *evenlvl,
                      unsigned int * stack1, 
                      unsigned int * stack2, 
                      unsigned int * stack1Top, 
                      unsigned int * stack2Top, 
                      unsigned int * support, 
                      unsigned int * supportTop, 
                      unsigned int * color, 
                      unsigned int *globalColorCounter, 
                      int*budAtDDFSEncounter, 
                      unsigned int *ddfsResult, 
                      unsigned int * curBridge,
                      int src, int dst) {
    if (removed[bud[src]] || removed[bud[dst]])
      return;
    stack1Top[0]=0;
    stack2Top[0]=0;
    supportTop[0]=0;

    auto minlvl = [&](int u)
    { return min(evenlvl[u], oddlvl[u]); };
    
    stack1[stack1Top[0]++]=bud[src];
    stack2[stack2Top[0]++]=bud[dst];
    //vector<int> Sr = {bud[src]}, Sg = {bud[dst]};
    //if(Sr[0] == Sg[0])
    //    return {Sr[0],Sg[0]};
    if (stack1[0]==stack2[0]){
        //printf("stack1[0]=%d == stack2[0]=%d\n",stack1[0],stack2[0]);
        //return (uint64_t) stack1[0] << 32 | stack2[0];
        ddfsResult[0]=stack1[0];
        ddfsResult[1]=stack2[0];
        curBridge[0] = src;
        curBridge[1] = dst;
        curBridge[2] = bud[src];
        curBridge[3] = bud[dst];
        return;

    }
    //out_support = {Sr[0], Sg[0]};
    support[supportTop[0]++]=stack1[0];
    support[supportTop[0]++]=stack2[0];

    //int newRed = color[Sr[0]] = ++globalColorCounter, newGreen = color[Sg[0]] = ++globalColorCounter;
    //assert(newRed == (newGreen^1));
    int newRed = color[stack1[0]] = ++globalColorCounter[0], newGreen = color[stack2[0]] = ++globalColorCounter[0];
    assert(newRed == (newGreen^1));
    

    for(;;) {
        //printf("IN FOR\n");
        //if found two disjoint paths
        //if(minlvl(Sr.back()) == 0 && minlvl(Sg.back()) == 0)
        if(minlvl(stack1[stack1Top[0]-1]) == 0 && minlvl(stack2[stack2Top[0]-1]) == 0){

            //printf("stack1[%d]=%d\n",stack1Top[0]-1,stack1[stack1Top[0]-1]);
            //printf("stack2[%d]=%d\n",stack2Top[0]-1,stack2[stack2Top[0]-1]);
            //printf("minlvl(graph,stack1[stack1Top[0]]) == 0 && minlvl(graph,stack2[stack2Top[0]]) == 0\n");
            ddfsResult[0]=stack1[stack1Top[0]-1];
            ddfsResult[1]=stack2[stack2Top[0]-1];
            curBridge[0] = src;
            curBridge[1] = dst;
            curBridge[2] = bud[src];
            curBridge[3] = bud[dst];
            return;
        }
        int b;
        //if(minlvl(Sr.back()) >= minlvl(Sg.back()))
        //printf("stack1[%d-1]=%d ml %d stack2[%d-1]=%d ml %d \n",stack1Top[0],stack1[stack1Top[0]-1],minlvl(stack1[stack1Top[0]-1]),stack2Top[0],stack2[stack2Top[0]-1],minlvl(stack2[stack2Top[0]-1]));
        if(minlvl(stack1[stack1Top[0]-1]) >= minlvl(stack2[stack2Top[0]-1])){
            //printf("ENTERED IF\n");
            b = ddfsMove(bud,offsets,cols,ddfsPredecessorsPtr,removed,predecessors,stack1,stack2,stack1Top,stack2Top,support,supportTop,color,globalColorCounter,budAtDDFSEncounter,newRed, newGreen);
        } else{
            //printf("ENTERED ELSE\n");
            b = ddfsMove(bud,offsets,cols,ddfsPredecessorsPtr,removed,predecessors,stack2,stack1,stack2Top,stack1Top,support,supportTop,color,globalColorCounter,budAtDDFSEncounter,newGreen, newRed);
        }
        if(b != -1){
            //return {b,b};
            //printf("B!=-1\n");
            //return (uint64_t) b << 32 | b;
            ddfsResult[0]=b;
            ddfsResult[1]=b;
            curBridge[0] = src;
            curBridge[1] = dst;
            curBridge[2] = bud[src];
            curBridge[3] = bud[dst];
            return;
        }
    }
}

__device__ void ddfsStructDev(DSU_CU bud,
                              ddfsStruct ddfs,
                      int src, int dst) {
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
            return;
        }
        int b;
        //if(minlvl(Sr.back()) >= minlvl(Sg.back()))
        //printf("stack1[%d-1]=%d ml %d stack2[%d-1]=%d ml %d \n",stack1Top[0],stack1[stack1Top[0]-1],minlvl(stack1[stack1Top[0]-1]),stack2Top[0],stack2[stack2Top[0]-1],minlvl(stack2[stack2Top[0]-1]));
        if(minlvl(ddfs.stack1[ddfs.stack1Top[0]-1]) >= minlvl(ddfs.stack2[ddfs.stack2Top[0]-1])){
            //printf("ENTERED IF\n");
            b = ddfsMove(bud,ddfs.offsets,ddfs.cols,ddfs.ddfsPredecessorsPtr,ddfs.removed,ddfs.predecessors,ddfs.stack1,
                          ddfs.stack2,ddfs.stack1Top,ddfs.stack2Top,ddfs.support,ddfs.supportTop,ddfs.color,ddfs.globalColorCounter,ddfs.budAtDDFSEncounter,newRed, newGreen);
        } else{
            //printf("ENTERED ELSE\n");
            b = ddfsMove(bud,ddfs.offsets,ddfs.cols,ddfs.ddfsPredecessorsPtr,ddfs.removed,ddfs.predecessors,ddfs.stack2,
                          ddfs.stack1,ddfs.stack2Top,ddfs.stack1Top,ddfs.support,ddfs.supportTop,ddfs.color,ddfs.globalColorCounter,ddfs.budAtDDFSEncounter,newGreen, newRed);
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
            return;
        }
    }
}


/*
https://forums.developer.nvidia.com/t/dynamic-sm-with-dynamic-parallelism/197328/8

Problem configuring with > 48K.

*/

__global__ void processBridgesPassStructs(
                                unsigned int * bridgeIndices,
                                unsigned int *numBridges,
                                unsigned int *rows,
                                unsigned int *cols,

                                // Common args
                                DSU_CU bud, 
                                ddfsStruct ddfs,
                                updateLvlStruct updateLvl,
                                APStruct ap,
                                int depth){
  for (int i = 0; i < numBridges[0]; ++i){
    auto u = rows[bridgeIndices[i]];
    auto v = cols[bridgeIndices[i]];
      if (ddfs.removed[bud[u]] || ddfs.removed[bud[v]])
        continue;
      ddfsStructDev(bud,ddfs,
                    u,v);
      updateLvlAndTenacityPassStruct_serialDev(                                      
                                      bud,
                                      updateLvl,
                                      depth);
      augumentPathIterativeSwitchPassStructDev(bud,ap);
  }
}

/*
__global__ void processBridgesPassStructsTimed(
                                unsigned int * bridgeIndices,
                                unsigned int *numBridges,
                                unsigned int *rows,
                                unsigned int *cols,

                                // Common args
                                DSU_CU bud, 
                                ddfsStruct ddfs,
                                updateLvlStruct updateLvl,
                                APStruct ap,
                                int depth,
                                unsigned int * processedBridgeCount,
                                unsigned int * count,
                                long long int * time1,
                                long long int * time2,
                                long long int * time3){
  long long int start = 0; 
  long long int stopK1 = 0;
  long long int stopK2 = 0;
  long long int stopK3 = 0;
  __shared__ int i;
  __shared__ int u;
  __shared__ int v;
  __shared__ bool removed;
  i = 0;
  __syncthreads();
  for (; i < numBridges[0];){    
    __syncthreads();
    if (threadIdx.x == 0){
      u = rows[bridgeIndices[i]];
      v = cols[bridgeIndices[i]];
      removed = (ddfs.removed[bud[u]] || ddfs.removed[bud[v]]);
    }
    __syncthreads();
    if (removed){
      if (threadIdx.x==0) i++;
      continue;
    }
    if (threadIdx.x == 0){
      processedBridgeCount[depth]++;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
      ddfsStructDev(bud,ddfs,
                    u,v,&count[depth]);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stopK1));
    }
    __syncthreads();
    if (threadIdx.x == 0){
      updateLvlAndTenacityPassStruct_serialDev(                                      
                                      bud,
                                      updateLvl,
                                      depth);
    }
    __syncthreads();
    if (threadIdx.x == 0){
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stopK2));
      augumentPathIterativeSwitchPassStructDev(bud,ap);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stopK3));
      time1[depth]+=stopK1-start;
      time2[depth]+=stopK2-stopK1;
      time3[depth]+=stopK3-stopK2;
    }
    __syncthreads();
    if (threadIdx.x==0) i++;
  }
}
*/


__global__ void processBridgesPassStructsTimed(
                                unsigned int * bridgeIndices,
                                unsigned int *numBridges,
                                unsigned int *rows,
                                unsigned int *cols,

                                // Common args
                                DSU_CU bud, 
                                ddfsStruct ddfs,
                                updateLvlStruct updateLvl,
                                APStruct ap,
                                int depth,
                                unsigned int * processedBridgeCount,
                                unsigned int * count,
                                long long int * time1,
                                long long int * time2,
                                long long int * time3){
  long long int start = 0; 
  long long int stopK1 = 0;
  long long int stopK2 = 0;
  long long int stopK3 = 0;
  int u;
  int v;
  __shared__ bool removed;
  for (int i = 0; i < numBridges[0];++i){   
    __syncthreads();
    if (threadIdx.x==0){
      u = rows[bridgeIndices[i]];
      v = cols[bridgeIndices[i]];
      removed = (ddfs.removed[bud[u]] || ddfs.removed[bud[v]]);
    }
    __syncthreads();
    if (removed){
      continue;
    }
    __syncthreads();
    if (threadIdx.x==0){
      processedBridgeCount[depth]++;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
      ddfsStructDev(bud,ddfs,
                    u,v,&count[depth]);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stopK1));
    }
    __syncthreads();
    //if (threadIdx.x==0){
      /*
      updateLvlAndTenacityPassStruct_serialDev(                                      
                                      bud,
                                      updateLvl,
                                      depth);
                                      */
      updateLvlAndTenacityPassStruct_parallelDev(                                      
                                      bud,
                                      updateLvl,
                                      depth);
    //}
    __syncthreads();
    if (threadIdx.x==0){
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stopK2));
      augumentPathIterativeSwitchPassStructDev(bud,ap);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stopK3));
      time1[depth]+=stopK1-start;
      time2[depth]+=stopK2-stopK1;
      time3[depth]+=stopK3-stopK2;
    }
  }
}


__global__ void processBridgesPassStructsParallelTimed(
                                unsigned int * connectedComponentOffsets,
                                unsigned int * connectedComponentColumns,
                                unsigned int * connectedComponentTop,
                                unsigned int numConnectedComponents,
                                //
                                unsigned int *numBridges,
                                unsigned int *rows,
                                unsigned int *cols,

                                // Common args
                                DSU_CU bud, 
                                ddfsStruct ddfs,
                                updateLvlStruct updateLvl,
                                APStruct ap,
                                int depth,
                                unsigned int * processedBridgeCount,
                                unsigned int * count,
                                long long int * time1,
                                long long int * time2,
                                long long int * time3){
  long long int start = 0; 
  long long int stopK1 = 0;
  long long int stopK2 = 0;
  long long int stopK3 = 0;
  int u;
  int v;
  __shared__ bool removed;
  __shared__ unsigned int connectedComponentIndex;

  if (threadIdx.x==0){
    connectedComponentIndex=atomicAdd(connectedComponentTop,1);
  }
  __syncthreads();
  if (threadIdx.x==0)
  assert(numBridges[0]==connectedComponentOffsets[numConnectedComponents]);
  while(connectedComponentIndex<numConnectedComponents){
    unsigned int startCC = connectedComponentOffsets[connectedComponentIndex];
    unsigned int endCC = connectedComponentOffsets[connectedComponentIndex+1];
    for (; startCC < endCC; ++startCC){   
      __syncthreads();
      if (threadIdx.x==0){
        u = rows[connectedComponentColumns[startCC]];
        v = cols[connectedComponentColumns[startCC]];
        removed = (ddfs.removed[bud[u]] || ddfs.removed[bud[v]]);
      }
      __syncthreads();
      if (removed){
        continue;
      }
      __syncthreads();
      if (threadIdx.x==0){
        processedBridgeCount[depth]++;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
        ddfsStructDev(bud,ddfs,
                      u,v,&count[depth],blockIdx.x);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stopK1));
      }
      __syncthreads();
      //if (threadIdx.x==0){
        /*
        updateLvlAndTenacityPassStruct_serialDev(                                      
                                        bud,
                                        updateLvl,
                                        depth);
                                        */
        updateLvlAndTenacityPassStruct_parallelDev(                                      
                                        bud,
                                        updateLvl,
                                        depth,blockIdx.x);
      //}
      __syncthreads();
      if (threadIdx.x==0){
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stopK2));
        augumentPathIterativeSwitchPassStructDev(bud,ap,blockIdx.x);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stopK3));
        time1[depth]+=stopK1-start;
        time2[depth]+=stopK2-stopK1;
        time3[depth]+=stopK3-stopK2;
      }
    }
    __syncthreads();
    if (threadIdx.x==0){
      connectedComponentIndex=atomicAdd(connectedComponentTop,1);
    }
    __syncthreads();
  }
}



void processBridgesPassStructsGlobal(
                                unsigned int maxSM,
                                unsigned int * bridgeIndices,
                                unsigned int *numBridges,
                                unsigned int *rows,
                                unsigned int *cols,
                                // Common args
                                DSU_CU bud, 
                                ddfsStruct ddfs,
                                updateLvlStruct updateLvl,
                                APStruct ap,
                                int depth){
  for (int i = 0; i < numBridges[0]; ++i){
    auto u = rows[bridgeIndices[i]];
    auto v = cols[bridgeIndices[i]];
    /*
    if (ddfs.removed[bud[u]] || ddfs.removed[bud[v]])
      continue;
    */
    ddfsStructGlobal<<<1,1>>>(bud,ddfs,
                  u,v);
    updateLvlAndTenacityPassStruct_serial<<<1,1>>>(                                      
                                    bud,
                                    updateLvl,
                                    ddfs.removed,
                                    depth,
                                    u,v);
    augumentPathIterativeSwitchPassStructGlobal<<<1,1,maxSM>>>(bud,ap,ddfs.removed,u,v);
  }
}

__global__ void processBridges(
                                int * bridgeIndices,
                                int numBridges,
                                // Common args
                                DSU_CU bud, 
                                unsigned int *offsets,
                                unsigned int *rows,
                                unsigned int *cols,
                                unsigned int *ddfsPredecessorsPtr,
                                bool *removed, 
                                bool *predecessors, 
                                int *oddlvl,
                                int *evenlvl,
                                // DDFS Args
                                unsigned int * stack1, 
                                unsigned int * stack2, 
                                unsigned int * stack1Top, 
                                unsigned int * stack2Top, 
                                unsigned int * support, 
                                unsigned int * supportTop, 
                                unsigned int * color, 
                                unsigned int *globalColorCounter, 
                                int*budAtDDFSEncounter, 
                                unsigned int *ddfsResult, 
                                unsigned int * curBridge,
                                // UpdateTen Args
                                int depth,
                                char *edgeStatus,
                                int *bridgeTenacity,
                                int *myBridge_a,
                                int *myBridge_b,
                                int *myBridge_c,
                                int *myBridge_d,
                                // AUgPath args
                                unsigned int *childsInDDFSTreePtr,
                                unsigned int * removedPredecessorsSize, 
                                unsigned int * predecessor_count,
                                int * removedVerticesQueue, 
                                unsigned int * removedVerticesQueueFront, 
                                unsigned int * removedVerticesQueueBack, 
                                bool * foundPath,
                                int * mate){
  for (int i = 0; i < numBridges; ++i){
    auto u = rows[bridgeIndices[i]];
    auto v = cols[bridgeIndices[i]];
    ddfsDev(bud,
        offsets,
        cols,
        ddfsPredecessorsPtr,
        removed,
        predecessors,
        oddlvl,
        evenlvl,
        stack1,
        stack2,
        stack1Top,
        stack2Top,
        support,
        supportTop,
        color,
        globalColorCounter,
        budAtDDFSEncounter,
        ddfsResult,
        curBridge,
        u,v);
      updateLvlAndTenacity_serialDev(support,
                                      oddlvl,
                                      evenlvl,
                                      mate,
                                      offsets,
                                      cols,
                                      edgeStatus,
                                      bridgeTenacity,
                                      myBridge_a,
                                      myBridge_b,
                                      myBridge_c,
                                      myBridge_d,
                                      bud,
                                      ddfsResult,
                                      curBridge,
                                      supportTop,
                                      depth);

    if (ddfsResult[0] != ddfsResult[1])
    {
        augumentPathIterativeSwitchDev(bud,
                      offsets,
                      cols,
                      oddlvl,
                      evenlvl,
                      edgeStatus,
                      predecessors,
                      childsInDDFSTreePtr,
                      removed,
                      myBridge_a,
                      myBridge_b,
                      myBridge_c,
                      myBridge_d,
                      color,
                      removedPredecessorsSize,
                      predecessor_count,
                      removedVerticesQueue,
                      removedVerticesQueueFront,
                      removedVerticesQueueBack,
                      foundPath,
                      budAtDDFSEncounter,
                      mate,
                      ddfsResult[0], ddfsResult[1],true);

    }
  }
}

bool bfsPassStructsDevGPUOnly(CSRGraph & csr)
{

  ddfsStruct ddfsArgsStruct(csr);
  updateLvlStruct updateLvlArgsStruct(csr);
  APStruct APArgsStruct(csr);

  int dimGridSetSrc = (csr.n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;

  /* Method 1 - Set Sources */
  setSources<<<dimGridSetSrc,THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(csr.oddlvl_d.data()),
                                        thrust::raw_pointer_cast(csr.evenlvl_d.data()),
                                        thrust::raw_pointer_cast(csr.mate_d.data()),
                                        csr.n,0);

  bool foundPath = false;
  bool nonEmpty = true;
  //csr.foundPath_h[0]=false;
  //csr.foundPath_d=csr.foundPath_h;
  memset(csr.h_foundPathPinned, false, sizeof(bool));
  cudaMemset(thrust::raw_pointer_cast(csr.foundPath_d.data()), false, sizeof(bool));
  for (int i = 0; i < n && !foundPath && nonEmpty; i++)
  {
    cudaMemset(thrust::raw_pointer_cast(csr.bridgeList_counter_d.data()), 0, sizeof(unsigned int));
    cudaMemset(thrust::raw_pointer_cast(csr.verticesInLevel_counter_d.data()), 0, sizeof(unsigned int));
    // This can be performed in parallel
    // each edge e only belongs to one vertex in the frontier.
    // The only race condition is if two outgoing edges in the frontier point to the same vertex v.
    // Then it is possible they would both add themselves to v's predecessors.
    // Not sure if this is a problem.
    // If it is, it can be solved by claiming vertices atomically.

    // Since the level is at most reduced to i+1, there are no race conditions due to
    // order of operations between the if(lvl>=i+1) and else
    /*
    bfsIterationArraysHost(csr,i);
    */

    // storage for the nonzero indices
    // compute indices of nonzero elements
    // Extracts vertices in the level.

    /* Method 2 - Graft M-ALT-BFS */
    /*
    typedef thrust::device_vector<int>::iterator IndexIterator;
    using namespace thrust::placeholders;
    IndexIterator indices_end = thrust::copy_if(thrust::make_counting_iterator(0),
                                                thrust::make_counting_iterator((int)csr.n),
                                                
                                                csr.verticesInLevel_d.begin(),
                                                _1 == i);
    */
    //csr.verticesInLevel_counter_h[0]=0;
    //csr.verticesInLevel_counter_d=csr.verticesInLevel_counter_h;
    extract_vertices_in_level<<<dimGridSetSrc,THREADS_PER_BLOCK>>>(                            
                            (i & 1) ? thrust::raw_pointer_cast(csr.oddlvl_d.data()) : 
                            thrust::raw_pointer_cast(csr.evenlvl_d.data()),
                            thrust::raw_pointer_cast(csr.verticesInLevel_d.data()),
                            thrust::raw_pointer_cast(csr.verticesInLevel_counter_d.data()),
                            thrust::raw_pointer_cast(csr.nonempty_d.data()),
                            csr.n,
                            i);
   /*
    csr.verticesInLevel_counter_h=csr.verticesInLevel_counter_d;
    unsigned int numV = csr.verticesInLevel_counter_h[0];
    int dimGridBFS = (numV + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    */
    bfsIterationArraysDevice_parallel<<<dimGridSetSrc,THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(csr.verticesInLevel_d.data()),
                                          thrust::raw_pointer_cast(csr.oddlvl_d.data()),
                                          thrust::raw_pointer_cast(csr.evenlvl_d.data()),
                                          thrust::raw_pointer_cast(csr.mate_d.data()),
                                          thrust::raw_pointer_cast(csr.offsets_d.data()),
                                          thrust::raw_pointer_cast(csr.cols_d.data()),
                                          thrust::raw_pointer_cast(csr.edgeStatus_d.data()),
                                          thrust::raw_pointer_cast(csr.predecessors_d.data()),
                                          thrust::raw_pointer_cast(csr.predecessors_count_d.data()),
                                          thrust::raw_pointer_cast(csr.bridgeTenacity_d.data()),
                                          thrust::raw_pointer_cast(csr.verticesInLevel_counter_d.data()),
                                          thrust::raw_pointer_cast(csr.bridgeList_counter_d.data()),
                                          thrust::raw_pointer_cast(csr.nonempty_d.data()),
                                          i,csr.n);

    int dimGridBridge = (2*csr.m + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    extract_bridges_parallel_edge_centric<<<dimGridBridge,THREADS_PER_BLOCK>>>(
                            thrust::raw_pointer_cast(csr.oddlvl_d.data()),
                            thrust::raw_pointer_cast(csr.evenlvl_d.data()),
                            thrust::raw_pointer_cast(csr.mate_d.data()),
                            thrust::raw_pointer_cast(csr.offsets_d.data()),
                            thrust::raw_pointer_cast(csr.cols_d.data()),
                            thrust::raw_pointer_cast(csr.edgeStatus_d.data()),
                            thrust::raw_pointer_cast(csr.bridgeTenacity_d.data()),
                            thrust::raw_pointer_cast(csr.bridgeList_counter_d.data()),
                            thrust::raw_pointer_cast(csr.bridgeList_d.data()),
                            2*csr.m,
                            i);

    cudaMemcpy(csr.h_numBridges, thrust::raw_pointer_cast(csr.bridgeList_counter_d.data()), sizeof(unsigned int), cudaMemcpyDeviceToHost);
    csr.numConnectedComponents=0;
    bool separate = csr.h_numBridges[0] > 0 && i < csr.depthCutoff;
    //printf("h_numBridges %llu i %d < csr.depthCutoff %d = %d\n",csr.h_numBridges[0],i,csr.depthCutoff,separate);
    if(separate){
      std::chrono::time_point<std::chrono::steady_clock> m_StartTime = std::chrono::steady_clock::now();
      bool predsRemain = true;
      bool loopEntered = false;

      cudaMemset(thrust::raw_pointer_cast(csr.bu_bfs_Top_d.data()), 0, sizeof(unsigned long long int));
      cudaMemset(thrust::raw_pointer_cast(csr.bu_bfs_buffer_Top_d.data()), 0, sizeof(unsigned long long int));
      cudaMemset(thrust::raw_pointer_cast(csr.bu_bfs_Top_root_d.data()), 0, sizeof(unsigned long long int));
      int dimGridFK = (csr.h_numBridges[0] + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
      assert(dimGridFK<65535);
      //printf("Calling BU_first_kernel %d < %d; %d\n",dimGridFK,65535,THREADS_PER_BLOCK);

      BU_first_kernel<<<dimGridFK,THREADS_PER_BLOCK>>>(
                          csr.bud,
                          thrust::raw_pointer_cast(csr.offsets_d.data()),
                          thrust::raw_pointer_cast(csr.rows_d.data()),
                          thrust::raw_pointer_cast(csr.cols_d.data()),
                          thrust::raw_pointer_cast(csr.mate_d.data()),
                          thrust::raw_pointer_cast(csr.predecessors_d.data()),
                          thrust::raw_pointer_cast(csr.bridgeList_counter_d.data()),
                          thrust::raw_pointer_cast(csr.bridgeList_d.data()),
                          thrust::raw_pointer_cast(csr.bu_bfs_key_d.data()),
                          thrust::raw_pointer_cast(csr.bu_bfs_val_d.data()),
                          thrust::raw_pointer_cast(csr.bu_bfs_Top_d.data()),
                          thrust::raw_pointer_cast(csr.bu_bfs_key_root_d.data()),
                          thrust::raw_pointer_cast(csr.bu_bfs_val_root_d.data()),
                          thrust::raw_pointer_cast(csr.bu_bfs_Top_root_d.data()),
                          csr.h_numBridges[0],
                          csr.maxPairs);
      cudaMemcpy(csr.bu_bfs_Top_Pinned, thrust::raw_pointer_cast(csr.bu_bfs_Top_d.data()), sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
      //printf("INTPAIRS %llu\n",csr.bu_bfs_Top_Pinned[0]);
      predsRemain=csr.bu_bfs_Top_Pinned[0]>0;
      bool parity = false;
      while(predsRemain){
        parity=!parity;
        if (parity){
          cudaMemset(thrust::raw_pointer_cast(csr.bu_bfs_buffer_Top_d.data()), 0, sizeof(unsigned long long int));
          int dimGridBUK = (csr.bu_bfs_Top_Pinned[0] + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
          assert(dimGridBUK<65535);
          //printf("Calling BU_first_kernel %d < %d; %d\n",dimGridBUK,65535,THREADS_PER_BLOCK);
          BU_kernel<<<dimGridBUK,THREADS_PER_BLOCK>>>(
                              csr.bud,
                              thrust::raw_pointer_cast(csr.offsets_d.data()),
                              thrust::raw_pointer_cast(csr.rows_d.data()),
                              thrust::raw_pointer_cast(csr.cols_d.data()),
                              thrust::raw_pointer_cast(csr.mate_d.data()),
                              thrust::raw_pointer_cast(csr.predecessors_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_key_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_val_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_Top_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_key_buffer_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_val_buffer_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_buffer_Top_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_key_root_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_val_root_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_Top_root_d.data()),
                              csr.bu_bfs_Top_Pinned[0],
                              csr.maxPairs);
          cudaMemcpy(csr.bu_bfs_buffer_Top_Pinned, thrust::raw_pointer_cast(csr.bu_bfs_buffer_Top_d.data()), sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
          //printf("NK %u MAX %u\n",csr.bu_bfs_buffer_Top_h[0],2*csr.m);
          //printf("INTPAIRS %llu\n",csr.bu_bfs_buffer_Top_Pinned[0]);

          predsRemain=csr.bu_bfs_buffer_Top_Pinned[0]>0;
        } else {
          cudaMemset(thrust::raw_pointer_cast(csr.bu_bfs_Top_d.data()), 0, sizeof(unsigned long long int));
          int dimGridBUK = (csr.bu_bfs_buffer_Top_Pinned[0] + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
          //printf("Calling BU_first_kernel %d < %d; %d\n",dimGridBUK,65535,THREADS_PER_BLOCK);
          assert(dimGridBUK<65535);
          BU_kernel<<<dimGridBUK,THREADS_PER_BLOCK>>>(
                              csr.bud,
                              thrust::raw_pointer_cast(csr.offsets_d.data()),
                              thrust::raw_pointer_cast(csr.rows_d.data()),
                              thrust::raw_pointer_cast(csr.cols_d.data()),
                              thrust::raw_pointer_cast(csr.mate_d.data()),
                              thrust::raw_pointer_cast(csr.predecessors_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_key_buffer_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_val_buffer_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_buffer_Top_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_key_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_val_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_Top_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_key_root_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_val_root_d.data()),
                              thrust::raw_pointer_cast(csr.bu_bfs_Top_root_d.data()),
                              csr.bu_bfs_buffer_Top_Pinned[0],
                              csr.maxPairs);
          cudaMemcpy(csr.bu_bfs_Top_Pinned, thrust::raw_pointer_cast(csr.bu_bfs_Top_d.data()), sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
          //printf("INTPAIRS %llu\n",csr.bu_bfs_Top_Pinned[0]);
          //printf("NK %u MAX %u\n",csr.bu_bfs_Top_h[0],2*csr.m);
          predsRemain=csr.bu_bfs_Top_Pinned[0]>0;
        }
      }
      cudaMemcpy(csr.bu_bfs_Top_Root_Pinned, thrust::raw_pointer_cast(csr.bu_bfs_Top_root_d.data()), sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
      assert(csr.h_numBridges[0]<csr.bu_bfs_Top_Root_Pinned[0]);
      typedef unsigned int KeyType;
      typedef unsigned int ValueType;
      thrust::sort_by_key(thrust::device, csr.bu_bfs_key_root_d.begin(), csr.bu_bfs_key_root_d.begin()+(unsigned int)csr.bu_bfs_Top_Root_Pinned[0], csr.bu_bfs_val_root_d.begin());
      // Combine the two vectors into a single zip iterator
      thrust::zip_iterator<thrust::tuple<thrust::device_vector<KeyType>::iterator, thrust::device_vector<ValueType>::iterator>> zip_begin(
          thrust::make_tuple(csr.bu_bfs_key_root_d.begin(), csr.bu_bfs_val_root_d.begin())
      );
      thrust::zip_iterator<thrust::tuple<thrust::device_vector<KeyType>::iterator, thrust::device_vector<ValueType>::iterator>> zip_end(
          thrust::make_tuple(csr.bu_bfs_key_root_d.begin()+(unsigned int)csr.bu_bfs_Top_Root_Pinned[0], csr.bu_bfs_val_root_d.begin()+(unsigned int)csr.bu_bfs_Top_Root_Pinned[0])
      );

      // Remove duplicates based on both keys and values
      auto new_end = thrust::unique(zip_begin, zip_end,
          [] __device__ (const thrust::tuple<KeyType, ValueType>& a, const thrust::tuple<KeyType, ValueType>& b) {
              return (thrust::get<0>(a) == thrust::get<0>(b)) && (thrust::get<1>(a) == thrust::get<1>(b));
          }
      );

      auto numUnique=new_end-zip_begin;
      
      int lit = 0;
      int dimGridLA = (numUnique + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
      thrust::fill(csr.claimed_A_d.begin(), csr.claimed_A_d.end(), INF); // or 999999.f if you prefer
      //thrust::fill(csr.claimed_B_d.begin(), csr.claimed_B_d.end(), INF); // or 999999.f if you prefer
      thrust::fill(csr.smallest_Ever_Seen_d.begin(), csr.smallest_Ever_Seen_d.end(), INF); // or 999999.f if you prefer
      
      thrust::fill(csr.claimedNew_d.begin(), csr.claimedNew_d.end(), false); // or 999999.f if you prefer
      LA_kernel_init<<<dimGridLA,THREADS_PER_BLOCK>>>(                            
          thrust::raw_pointer_cast(csr.claimed_A_d.data()),
          thrust::raw_pointer_cast(csr.bu_bfs_key_root_d.data()),
          numUnique);
      /*
      csr.claimed_Track_d=csr.claimed_A_d;

      // Sort the vector (if not already sorted)
      thrust::sort(csr.claimed_Track_d.begin(), csr.claimed_Track_d.end());

      // Use thrust::unique to remove duplicates
      auto first_new_end_un = thrust::unique(csr.claimed_Track_d.begin(), csr.claimed_Track_d.end());

      // Calculate the number of unique elements
      auto first_num_unique_cc = thrust::distance(csr.claimed_Track_d.begin(), first_new_end_un) - 1;
      printf("FIRST link it %d num cc %d numUnique %d\n",lit, first_num_unique_cc, numUnique);
      assert(first_num_unique_cc<=numUnique);
      */
      csr.h_claimedNewPinned[0]=true;
      bool parityLA = false;
      while(csr.h_claimedNewPinned[0]){
        //printf("Enerted loop\n");
        csr.h_claimedNewPinned[0]=false;
        cudaMemset(thrust::raw_pointer_cast(csr.claimedNew_d.data()), false, sizeof(bool));
        parityLA=!parityLA;
        if(parityLA){
          //printf("Enerted if\n");
          LA_kernelAtomic<<<dimGridLA,THREADS_PER_BLOCK>>>(                            
              thrust::raw_pointer_cast(csr.claimed_A_d.data()),
              thrust::raw_pointer_cast(csr.smallest_Ever_Seen_d.data()),
              thrust::raw_pointer_cast(csr.bu_bfs_key_root_d.data()),
              thrust::raw_pointer_cast(csr.bu_bfs_val_root_d.data()),
              numUnique,
              thrust::raw_pointer_cast(csr.claimedNew_d.data()));
        } else {
          //printf("Enerted else\n");
          LA_kernel<<<dimGridLA,THREADS_PER_BLOCK>>>(   
              thrust::raw_pointer_cast(csr.smallest_Ever_Seen_d.data()),
              thrust::raw_pointer_cast(csr.claimed_A_d.data()),
              thrust::raw_pointer_cast(csr.bu_bfs_val_root_d.data()),
              thrust::raw_pointer_cast(csr.bu_bfs_key_root_d.data()),
              numUnique,
              thrust::raw_pointer_cast(csr.claimedNew_d.data()));
          /*
          csr.claimed_Track_d=csr.claimed_A_d;
          // Sort the vector (if not already sorted)
          thrust::sort(csr.claimed_Track_d.begin(), csr.claimed_Track_d.end());

          // Use thrust::unique to remove duplicates
          auto new_end_un = thrust::unique(csr.claimed_Track_d.begin(), csr.claimed_Track_d.end());

          // Calculate the number of unique elements
          auto num_unique_cc = thrust::distance(csr.claimed_Track_d.begin(), new_end_un) - 1;
          printf("link it %d num cc %d numUnique %d\n",lit++, num_unique_cc, numUnique);
          */
        }
        cudaMemcpy(csr.h_claimedNewPinned, 
                  thrust::raw_pointer_cast(csr.claimedNew_d.data()), 
                  sizeof(bool), 
                  cudaMemcpyDeviceToHost);
      }


      setGroupRoot<<<dimGridFK,THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(csr.claimed_A_d.data()),                                 
                                              thrust::raw_pointer_cast(csr.bridgeList_d.data()),
                                              thrust::raw_pointer_cast(csr.bridgeList_groupRoot_d.data()),
                                              thrust::raw_pointer_cast(csr.bridgeList_counter_d.data()));
    
      #if DEBUGDSU

      //printf("FIN LINKING level %d\n",i);
      // Sort the vector (if not already sorted)
      thrust::sort(csr.claimed_A_d.begin(), csr.claimed_A_d.end());

      // Use thrust::unique to remove duplicates
      auto new_end_un = thrust::unique(csr.claimed_A_d.begin(), csr.claimed_A_d.end());

      // Calculate the number of unique elements
      auto num_unique_cc = thrust::distance(csr.claimed_A_d.begin(), new_end_un) - 1;

      // Could be done in another stream
      //csr.budCC.reset();
      thrust::fill(csr.VertexClaimedByBridge_d.begin(), csr.VertexClaimedByBridge_d.end(), INF); // or 999999.f if you prefer

      linkAllEdges<<<1,1>>>(csr.budCC,
                            thrust::raw_pointer_cast(csr.VertexClaimedByBridge_d.data()),
                            thrust::raw_pointer_cast(csr.bu_bfs_key_root_d.data()),
                            thrust::raw_pointer_cast(csr.bu_bfs_val_root_d.data()),
                            thrust::raw_pointer_cast(csr.bu_bfs_Top_root_d.data()));

      setGroupRoot<<<dimGridFK,THREADS_PER_BLOCK>>>(csr.budCC,                                      
                                              thrust::raw_pointer_cast(csr.bridgeList_d.data()),
                                              thrust::raw_pointer_cast(csr.bridgeList_groupRoot_d.data()),
                                              thrust::raw_pointer_cast(csr.bridgeList_counter_d.data()));
      

      setGroupRoot<<<dimGridFK,THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(csr.claimed_A_d.data()),                                 
                                              thrust::raw_pointer_cast(csr.bridgeList_d.data()),
                                              thrust::raw_pointer_cast(csr.bridgeList_groupRoot2_d.data()),
                                              thrust::raw_pointer_cast(csr.bridgeList_counter_d.data()));
      
      csr.createBridgeOffsets();

      assert(csr.numConnectedComponents==num_unique_cc);
      //assert(csr.bridgeList_groupRoot_d==csr.bridgeList_groupRoot2_d);
      #else

      csr.createBridgeOffsets();

      #endif


      std::chrono::time_point<std::chrono::steady_clock> m_EndTime = std::chrono::steady_clock::now();
      double elapsedSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(m_EndTime - m_StartTime).count();
      #if PRINT
      printf("ROW %u %u %u %f %f %llu %llu %u\n",globalCounter++, iter-1, i,(double)(csr.h_numBridges[0])/(double)csr.numConnectedComponents, elapsedSeconds, csr.h_numBridges[0],csr.bu_bfs_Top_Root_Pinned[0],csr.numConnectedComponents);
      #endif
    }            


    //printf("NB %d\n",csr.bridgeList_counter_h[0]);
    #if USE_DEVICE
    #if TIME
    if (separate){
    cudaMemset(thrust::raw_pointer_cast(csr.connectedComponentTop_d.data()), 0, sizeof(unsigned int));
    processBridgesPassStructsParallelTimed<<<csr.numStacks,64,csr.maxSM>>>(                                        
      thrust::raw_pointer_cast(csr.offsets_bridges_d.data()),
      thrust::raw_pointer_cast(csr.bridgeList_d.data()),
      thrust::raw_pointer_cast(csr.connectedComponentTop_d.data()),
      csr.numConnectedComponents,
      thrust::raw_pointer_cast(csr.bridgeList_counter_d.data()),
      thrust::raw_pointer_cast(csr.rows_d.data()),
      thrust::raw_pointer_cast(csr.cols_d.data()),
      csr.bud,
      ddfsArgsStruct,
      updateLvlArgsStruct,
      APArgsStruct,
      i,
      thrust::raw_pointer_cast(csr.count_d.data()),
      thrust::raw_pointer_cast(csr.verticesTraversed_d.data()),
      thrust::raw_pointer_cast(csr.K1Time_d.data()),
      thrust::raw_pointer_cast(csr.K2Time_d.data()),
      thrust::raw_pointer_cast(csr.K3Time_d.data())
    );
    } else {
    processBridgesPassStructsTimed<<<1,512,csr.maxSM>>>(                                        
      thrust::raw_pointer_cast(csr.bridgeList_d.data()),
      thrust::raw_pointer_cast(csr.bridgeList_counter_d.data()),
      thrust::raw_pointer_cast(csr.rows_d.data()),
      thrust::raw_pointer_cast(csr.cols_d.data()),
      csr.bud,
      ddfsArgsStruct,
      updateLvlArgsStruct,
      APArgsStruct,
      i,
      thrust::raw_pointer_cast(csr.count_d.data()),
      thrust::raw_pointer_cast(csr.verticesTraversed_d.data()),
      thrust::raw_pointer_cast(csr.K1Time_d.data()),
      thrust::raw_pointer_cast(csr.K2Time_d.data()),
      thrust::raw_pointer_cast(csr.K3Time_d.data())
    );
    }
    #else
    processBridgesPassStructs<<<1,1,csr.maxSM>>>(                                        
      thrust::raw_pointer_cast(csr.bridgeList_d.data()),
      thrust::raw_pointer_cast(csr.bridgeList_counter_d.data()),
      thrust::raw_pointer_cast(csr.rows_d.data()),
      thrust::raw_pointer_cast(csr.cols_d.data()),
      csr.bud,
      ddfsArgsStruct,
      updateLvlArgsStruct,
      APArgsStruct,
      i
    );
    #endif
    #else
    cudaMemcpy(csr.h_numBridges, thrust::raw_pointer_cast(csr.bridgeList_counter_d.data()), sizeof(unsigned int), cudaMemcpyDeviceToHost);
    csr.bridgeList_h=csr.bridgeList_d;
    processBridgesPassStructsGlobal(             csr.maxSM,                           
      thrust::raw_pointer_cast(csr.bridgeList_h.data()),
      csr.h_numBridges,
      thrust::raw_pointer_cast(csr.rows_h.data()),
      thrust::raw_pointer_cast(csr.cols_h.data()),
      csr.bud,
      ddfsArgsStruct,
      updateLvlArgsStruct,
      APArgsStruct,
      i
    );
    #endif
    //csr.foundPath_h=csr.foundPath_d;
    //foundPath=csr.foundPath_h[0];
    cudaMemcpy(csr.h_foundPathPinned, thrust::raw_pointer_cast(csr.foundPath_d.data()), sizeof(bool), cudaMemcpyDeviceToHost);
    foundPath=csr.h_foundPathPinned[0];
    if (i+1<n)
      cudaMemcpy(csr.h_nonemptyPinned, &(thrust::raw_pointer_cast(csr.nonempty_d.data()))[i+1], sizeof(bool), cudaMemcpyDeviceToHost);
    nonEmpty=csr.h_nonemptyPinned[0];
    //memcpy(&foundPath, csr.h_foundPathPinned, sizeof(bool));
  }
  return csr.h_foundPathPinned[0];
}

bool bfsPassStructsDev(CSRGraph & csr)
{

  ddfsStruct ddfsArgsStruct(csr);
  updateLvlStruct updateLvlArgsStruct(csr);
  APStruct APArgsStruct(csr);

  vector<vector<int>> verticesAtLevel(n);
  vector<vector<pii>> bridges(2 * n + 2);
  vector<int> predecessorsSize(n);
  vector<int> removedPredecessorsSize(n);
  vector<int> removedPredecessorsSize2(n);
  childsInDDFSTreePtr.clear();
  childsInDDFSTreePtr.resize(n);
  auto setLvl = [&](int u, int lev)
  {
    if (lev & 1)
      oddlvl[u] = lev;
    else
      evenlvl[u] = lev;
    //verticesAtLevel[lev].push_back(u);
    insert_sorted(verticesAtLevel[lev],u);
  };

  /* Method 1 - Set Sources */
  for (int u = 0; u < n; u++)
    if (mate[u] == -1)
      setLvl(u, 0);

  int dimGridBFS = (csr.n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;

    /*
    csr.oddlvl_h=oddlvl;
    csr.oddlvl_d=csr.oddlvl_h;
    csr.evenlvl_h=evenlvl;
    csr.evenlvl_d=csr.evenlvl_h;
    */
  /* Method 1 - Set Sources */
  setSources<<<dimGridBFS,THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(csr.oddlvl_d.data()),
                                        thrust::raw_pointer_cast(csr.evenlvl_d.data()),
                                        thrust::raw_pointer_cast(csr.mate_d.data()),
                                        csr.n,0);

  bool foundPath = false;
  csr.foundPath_h[0]=false;
  csr.foundPath_d=csr.foundPath_h;
  for (int i = 0; i < n && !foundPath; i++)
  {

    // This can be performed in parallel
    // each edge e only belongs to one vertex in the frontier.
    // The only race condition is if two outgoing edges in the frontier point to the same vertex v.
    // Then it is possible they would both add themselves to v's predecessors.
    // Not sure if this is a problem.
    // If it is, it can be solved by claiming vertices atomically.

    // Since the level is at most reduced to i+1, there are no race conditions due to
    // order of operations between the if(lvl>=i+1) and else
    /*
    bfsIterationArraysHost(csr,i);
    */

    // storage for the nonzero indices
    // compute indices of nonzero elements
    // Extracts vertices in the level.

    /* Method 2 - Graft M-ALT-BFS */
    typedef thrust::device_vector<int>::iterator IndexIterator;
    using namespace thrust::placeholders;
    IndexIterator indices_end = thrust::copy_if(thrust::make_counting_iterator(0),
                                                thrust::make_counting_iterator((int)csr.n),
                                                (i & 1) ? csr.oddlvl_d.begin() : csr.evenlvl_d.begin(),
                                                csr.verticesInLevel_d.begin(),
                                                _1 == i);
    //printf("Vertices in level Serial %lu Parallel %lu\n",verticesAtLevel[i].size(),(indices_end-csr.verticesInLevel_d.begin()));
    bfsIterationArraysDevice_serial<<<1,1>>>(thrust::raw_pointer_cast(csr.verticesInLevel_d.data()),
                                      thrust::raw_pointer_cast(csr.oddlvl_d.data()),
                                      thrust::raw_pointer_cast(csr.evenlvl_d.data()),
                                      thrust::raw_pointer_cast(csr.mate_d.data()),
                                      thrust::raw_pointer_cast(csr.offsets_d.data()),
                                      thrust::raw_pointer_cast(csr.cols_d.data()),
                                      thrust::raw_pointer_cast(csr.edgeStatus_d.data()),
                                      thrust::raw_pointer_cast(csr.predecessors_d.data()),
                                      thrust::raw_pointer_cast(csr.predecessors_count_d.data()),
                                      thrust::raw_pointer_cast(csr.bridgeTenacity_d.data()),
                                      (indices_end-csr.verticesInLevel_d.begin()),i);

    /*
    bfsIterationArraysDevice_parallel<<<dimGridBFS,THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(csr.oddlvl_d.data()),
                                      thrust::raw_pointer_cast(csr.evenlvl_d.data()),
                                      thrust::raw_pointer_cast(csr.mate_d.data()),
                                      thrust::raw_pointer_cast(csr.offsets_d.data()),
                                      thrust::raw_pointer_cast(csr.cols_d.data()),
                                      thrust::raw_pointer_cast(csr.edgeStatus_d.data()),
                                      thrust::raw_pointer_cast(csr.predecessors_d.data()),
                                      thrust::raw_pointer_cast(csr.bridgeTenacity_d.data()),
                                      csr.n,i);
                                      */

    typedef thrust::device_vector<char>::iterator   CharIterator;
    typedef thrust::device_vector<int>::iterator IntIterator;

    typedef thrust::tuple<CharIterator, IntIterator> IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

    e_op_t e_op{};
    e_op.depth=i;
    typedef thrust::device_vector<unsigned int>::iterator IndexUINTIterator;
    ZipIterator iter(thrust::make_tuple(csr.edgeStatus_d.begin(), csr.bridgeTenacity_d.begin()));
    IndexUINTIterator indices_bridges_end = thrust::copy_if(thrust::make_counting_iterator((unsigned int)0),
                                                    thrust::make_counting_iterator((unsigned int)(2*csr.m)),
                                                    iter,
                                                    csr.bridgeList_d.begin(),
                                                    e_op);

    /* Method 2 - Graft M-ALT-BFS */
    bfsIterationVectors(verticesAtLevel,bridges,i);

    // check if size of predecessors match between vector and array versions
    /*
    for (int vec = 0; vec < n; ++vec){
      predecessorsSize[vec]=predecessors[vec].size();
    }
    csr.predecessors_count_h=csr.predecessors_count_d;
    assert (csr.predecessors_count_h==predecessorsSize);
  

    for (int vertex = 0; vertex < n; ++vertex){
      std::vector<int> arrayPred;
      unsigned int start = csr.offsets_h[vertex];
      unsigned int end = csr.offsets_h[vertex + 1];
      unsigned int edgeIndex = start;
      for(; edgeIndex < end; edgeIndex++) { // Delete Neighbors of startingVertex
        //printf("src %d dst %d ddfsPredecessorsPtr %d  pred %d \n",u,cols[edgeIndex], ddfsPredecessorsPtr[u], predecessors[edgeIndex]);
        if (csr.predecessors_h[edgeIndex]) {
          arrayPred.push_back((int)csr.cols_h[edgeIndex]);
        }
      }
      assert(predecessors[vertex].size() == arrayPred.size());
    }
  */

    /*
    b_op_t b_op{};
    b_op.rows=thrust::raw_pointer_cast(csr.rows_d.data());
    b_op.cols=thrust::raw_pointer_cast(csr.cols_d.data());
    unsigned int numB = (indices_bridges_end-csr.bridgesInLevel_d.begin());
    thrust::device_vector<thrust::tuple<unsigned int,unsigned int>> bridgesVec_d((indices_bridges_end-csr.bridgesInLevel_d.begin()));
    thrust::host_vector<thrust::tuple<unsigned int,unsigned int>> bridgesVec_h((indices_bridges_end-csr.bridgesInLevel_d.begin()));
    thrust::transform(csr.bridgesInLevel_d.begin(), csr.bridgesInLevel_d.begin()+numB, bridgesVec_d.begin(), b_op);    
    bridgesVec_h=bridgesVec_d;
    thrust::host_vector<thrust::tuple<unsigned int,unsigned int>> cpuBridges(numB);
    for (int index = 0; index < bridges[2 * i + 1].size();++index)
      cpuBridges[index]=thrust::make_tuple<unsigned int, unsigned int>((unsigned int)bridges[2 * i + 1][index].st, (unsigned int)bridges[2 * i + 1][index].nd);
    assert(cpuBridges==bridgesVec_h);
    assert(bridges[2 * i + 1].size()==(indices_bridges_end-csr.bridgesInLevel_d.begin()));
    */
    // This loop should be parallelized using a dynamic worklist.  Bridges should be popped of the stack
    // and ddfs performed using a copy of the graph.  The race condition lies in the bud array.
    // The way to extract parallelism is to check if

    // removed, bud, predecessors, oddlvl, and evenlvl are constant during a ddfs call.
    // color,stack1,stack2,support,childsInDDFSTree are not, and will be private to a ddfs call.
    // int nthreads, tid;
     // #pragma omp parallel for\
        shared(removed,bud,predecessors,oddlvl,evenlvl,i,removedVerticesQueue)\
        firstprivate(globalColorCounter ,color, childsInDDFSTree, ddfsPredecessorsPtr)\
        private(nthreads, tid)
    for (auto b : bridges[2 * i + 1])
    {
      /*
      csr.mate_h=mate;
      csr.mate_d=csr.mate_h;
      csr.removed_h=removed;
      csr.removed_d=csr.removed_h;
      cudaMemcpy(csr.bud.directParent,&bud.directParent[0],bud.directParent.size()*sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(csr.bud.groupRoot,&bud.groupRoot[0],bud.directParent.size()*sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(csr.bud.size,&bud.size[0],bud.directParent.size()*sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(csr.bud.link,&bud.link[0],bud.directParent.size()*sizeof(int),cudaMemcpyHostToDevice);
      csr.removed_d=csr.removed_h;
      */

      /* Start Method 3 - Find Support */
      if (removed[bud[b.st]] || removed[bud[b.nd]])
        continue;
      vector<int> support;

      // Race conditions exist if some threads are contracting bud's
      auto ddfsResult = ddfs(b, support);

      /*even when we found two disjoint paths, we create fake petal, with bud in the end of second path
      the support of this bridge will be these two pathes and some other vertices, which have bases on this paths, so we will remove them and this will not affect corectness
      using this trick, we can simply call augumentPath on these two ends - the first end is just above fake bud, so it will augument exactly the path we need
      the only problem is that some vertices in this support will be uncorrectly classified as inner/outer, so we need to pass initial=true flag to fix this case*/
      pair<pii, pii> curBridge = {b, {bud[b.st], bud[b.nd]}};     

      /*
      csr.support_h=csr.support_d;

      for (int index = 0; index < support.size(); ++index){
         assert(support[index]==csr.support_h[index]);
      }
      */     

      /* End Method 3 - Find Support */

      /* Start Method 4 - Traverse Support */
      for (auto v : support)
      {
        if (v == ddfsResult.second)
          continue; // skip bud
        myBridge[v] = curBridge;

        bud.linkTo(v, ddfsResult.second);
        //bud2.linkTo(v, ddfsResult.second);

        // this part of code is only needed when bottleneck found, but it doesn't mess up anything when called on two paths
        setLvl(v, 2 * i + 1 - minlvl(v));
        for (auto f : graph[v])
          if (evenlvl[v] > oddlvl[v] && f.type == Bridge && tenacity({v, f.to}) < INF && mate[v] != f.to)
            //bridges[tenacity({v, f.to})].push_back({v, f.to});
            insert_sorted(bridges[tenacity({v, f.to})],{v, f.to});
      }


      /* End Method 4 - Traverse Support */

      /* Method 5 - Augment Path */
      if (ddfsResult.first != ddfsResult.second)
      {
        augumentPath(ddfsResult.first, ddfsResult.second, true);
        foundPath = true;
        //printf("cpu removedVerticesQueue.size() %d\n",removedVerticesQueue.size());
        while (!removedVerticesQueue.empty())
        {
          int v = removedVerticesQueue.front();
          //printf("v %d\n",v);
          removedVerticesQueue.pop();
          for (auto e : graph[v]){
            //printf("v %d e.to %d predecessors[%d].size() %d\n",v,e.to,e.to,predecessors[e.to].size());
            if (e.type == Prop && minlvl(e.to) > minlvl(v) && !removed[e.to] && ++removedPredecessorsSize[e.to] == predecessors[e.to].size()){
              //printf("removing %d post augpath\n",e.to);
              removeAndPushToQueue(e.to);
            }
          }
        }
      }

      /* Method 5 - Augment Path */
      /*
      childsInDDFSTreePtr.clear();
      childsInDDFSTreePtr.resize(n,0);
      if (ddfsResult.first != ddfsResult.second)
      {
        //augumentPath2(ddfsResult.first, ddfsResult.second, true);
        augumentPathStack(ddfsResult.first, ddfsResult.second, true);
        foundPath = true;
        while (!removedVerticesQueue2.empty())
        {
          int v = removedVerticesQueue2.front();
          removedVerticesQueue2.pop();
          for (auto e : graph[v])
            if (e.type == Prop && minlvl(e.to) > minlvl(v) && !removed2[e.to] && ++removedPredecessorsSize2[e.to] == predecessors[e.to].size())
              removeAndPushToQueue2(e.to);
        }
      }
      */

      // Resetting this as used in the device kernel.
      //thrust::fill(csr.childsInDDFSTreePtr_d.begin(), csr.childsInDDFSTreePtr_d.end(), 0); // or 999999.f if you prefer
      /*
      csr.mate_h=csr.mate_d;
      csr.removed_h=csr.removed_d;
      assert(csr.mate_h == mate);
      assert(csr.removed_h == removed);
      */
    }
    
    unsigned int numB = (indices_bridges_end-csr.bridgeList_d.begin());
    processBridgesPassStructs<<<1,1>>>(                                        
      (unsigned int *)thrust::raw_pointer_cast(csr.bridgeList_d.data()),
      &numB,
      thrust::raw_pointer_cast(csr.rows_d.data()),
      thrust::raw_pointer_cast(csr.cols_d.data()),
      csr.bud,
      ddfsArgsStruct,
      updateLvlArgsStruct,
      APArgsStruct,
      i
    );
    csr.foundPath_h=csr.foundPath_d;
    if (foundPath!=csr.foundPath_h[0])
      printf("CPU FP %d GPU FP %d\n",foundPath,csr.foundPath_h[0]);
    assert(foundPath==csr.foundPath_h[0]);
  }
  return foundPath;
}

// just for testing purposes
void checkGraph()
{
  for (int i = 0; i < n; i++)
    assert(mate[i] == -1 || mate[mate[i]] == i);
}

void mvMatching(CSRGraph &csr, GreedyMatcher &gm, BFS &b)
{   
    //mate = vector<int>(n,-1);
    do {
        for(auto&a: graph)
            for(auto&e:a)
                e.type = NotScanned;
        
        predecessors = vector<vector<int> > (n);
        ddfsPredecessorsPtr = color = removed = vector<int>(n);
        evenlvl = oddlvl = vector<int>(n,INF);
        childsInDDFSTree = vector<vector<pii> > (n);
        globalColorCounter = 1;
        bud.reset(n);
        myBridge = vector<pair<pii,pii> >(n);        
    }while(bfs());
    //checkGraph();
}

//void mvMatchingGPU(CSRGraph &csr, GreedyMatcher &gm, BFS &b)
void mvMatchingGPU(CSRGraph &csr)
{
  //mate = vector<int>(n, -1);
      
  /*
  std::chrono::time_point<std::chrono::steady_clock> m_StartTime = std::chrono::steady_clock::now();
  int numAugmented = gm.maxMatch(mate);
  std::chrono::time_point<std::chrono::steady_clock> m_EndTime = std::chrono::steady_clock::now();
  double elapsedSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(m_EndTime - m_StartTime).count() / 1000.0;
  std::cout << "Greedy match seconds: " << elapsedSeconds << "; edges augmented: " << numAugmented << std::endl;
  std::chrono::time_point<std::chrono::steady_clock> m2_StartTime = std::chrono::steady_clock::now();
  int numAugmented2 = b.augmentNaivePaths(mate);
  std::chrono::time_point<std::chrono::steady_clock> m2_EndTime = std::chrono::steady_clock::now();
  double elapsedSeconds2 = std::chrono::duration_cast<std::chrono::milliseconds>(m2_EndTime - m2_StartTime).count() / 1000.0;
  std::cout << "Trivial DDFS match seconds: " << elapsedSeconds2 << "; edges augmented: " << numAugmented2 << std::endl;
  */
  int dimGridN = (csr.n + THREADS_PER_BLOCK) / THREADS_PER_BLOCK;
  int dimGridM = (2*csr.m + THREADS_PER_BLOCK) / THREADS_PER_BLOCK;

  do
  {
    printf("Iter %d\n",iter++);
    //csr.reset();
    
    resetN<<<dimGridN,THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(csr.childsInDDFSTreePtr_d.data()),
                                            thrust::raw_pointer_cast(csr.removed_d.data()),
                                            thrust::raw_pointer_cast(csr.color_d.data()),
                                            thrust::raw_pointer_cast(csr.ddfsPredecessorsPtr_d.data()),
                                            thrust::raw_pointer_cast(csr.removedPredecessorsSize_d.data()),
                                            thrust::raw_pointer_cast(csr.predecessors_count_d.data()),
                                            thrust::raw_pointer_cast(csr.evenlvl_d.data()),
                                            thrust::raw_pointer_cast(csr.oddlvl_d.data()),
                                            csr.bud.size,
                                            csr.bud.directParent,
                                            csr.bud.link,
                                            csr.bud.groupRoot,
                                            thrust::raw_pointer_cast(csr.globalColorCounter_d.data()),
                                            csr.n);
    thrust::fill(csr.color_d.begin(), csr.color_d.end(), 0); // or 999999.f if you prefer
    thrust::fill(csr.globalColorCounter_d.begin(), csr.globalColorCounter_d.end(), 1); // or 999999.f if you prefer
    thrust::fill(csr.ddfsPredecessorsPtr_d.begin(), csr.ddfsPredecessorsPtr_d.end(), 0); // or 999999.f if you prefer
    thrust::fill(csr.budAtDDFSEncounter_d.begin(), csr.budAtDDFSEncounter_d.end(), -1); // or 999999.f if you prefer
    thrust::fill(csr.removed_d.begin(), csr.removed_d.end(), false); // or 999999.f if you prefer

    resetM<<<dimGridM,THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(csr.edgeStatus_d.data()),
                                            thrust::raw_pointer_cast(csr.predecessors_d.data()),
                                            thrust::raw_pointer_cast(csr.budAtDDFSEncounter_d.data()),
                                            thrust::raw_pointer_cast(csr.bridgeTenacity_d.data()),
                                            2*csr.m);
  } while (bfsPassStructsDevGPUOnly(csr));
  //} while (bfs(csr));
  //thrust::copy(csr.mate_d.begin(), csr.mate_d.end(), mate.begin());
  //checkGraph();
}


void initializeMatching(CSRGraph &csr, GreedyMatcher &gm, BFS &b)
{
  //mate = vector<int>(n, -1);
      
  std::chrono::time_point<std::chrono::steady_clock> m_StartTime = std::chrono::steady_clock::now();
  int numAugmented = gm.maxMatch(mate);
  std::chrono::time_point<std::chrono::steady_clock> m_EndTime = std::chrono::steady_clock::now();
  double elapsedSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(m_EndTime - m_StartTime).count() / 1000.0;  
  std::cout << "Greedy match seconds: " << elapsedSeconds << "; edges augmented: " << numAugmented << std::endl;
  csr.gm_cnt=numAugmented;
  csr.gm_sec=elapsedSeconds;
  std::chrono::time_point<std::chrono::steady_clock> m2_StartTime = std::chrono::steady_clock::now();
  int numAugmented2 = b.augmentNaivePaths(mate);
  std::chrono::time_point<std::chrono::steady_clock> m2_EndTime = std::chrono::steady_clock::now();
  double elapsedSeconds2 = std::chrono::duration_cast<std::chrono::milliseconds>(m2_EndTime - m2_StartTime).count() / 1000.0;
  std::cout << "Trivial DDFS match seconds: " << elapsedSeconds2 << "; edges augmented: " << numAugmented2 << std::endl;
  csr.na_cnt=numAugmented2;
  csr.na_sec=elapsedSeconds2;
  csr.mate_h=csr.mate_d;
}


void mvMatchingGPU(CSRGraph &csr, GreedyMatcher &gm, BFS &b)
{
      
  int dimGridN = (csr.n + THREADS_PER_BLOCK) / THREADS_PER_BLOCK;
  int dimGridM = (2*csr.m + THREADS_PER_BLOCK) / THREADS_PER_BLOCK;

  do
  {
    printf("Iter %d\n",iter++);
    //csr.reset();
    
    resetN<<<dimGridN,THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(csr.childsInDDFSTreePtr_d.data()),
                                            thrust::raw_pointer_cast(csr.removed_d.data()),
                                            thrust::raw_pointer_cast(csr.color_d.data()),
                                            thrust::raw_pointer_cast(csr.ddfsPredecessorsPtr_d.data()),
                                            thrust::raw_pointer_cast(csr.removedPredecessorsSize_d.data()),
                                            thrust::raw_pointer_cast(csr.predecessors_count_d.data()),
                                            thrust::raw_pointer_cast(csr.evenlvl_d.data()),
                                            thrust::raw_pointer_cast(csr.oddlvl_d.data()),
                                            csr.bud.size,
                                            csr.bud.directParent,
                                            csr.bud.link,
                                            csr.bud.groupRoot,
                                            thrust::raw_pointer_cast(csr.globalColorCounter_d.data()),
                                            csr.n);
                                            /*
    thrust::fill(csr.color_d.begin(), csr.color_d.end(), 0); // or 999999.f if you prefer
    thrust::fill(csr.globalColorCounter_d.begin(), csr.globalColorCounter_d.end(), 1); // or 999999.f if you prefer
    thrust::fill(csr.ddfsPredecessorsPtr_d.begin(), csr.ddfsPredecessorsPtr_d.end(), 0); // or 999999.f if you prefer
    thrust::fill(csr.budAtDDFSEncounter_d.begin(), csr.budAtDDFSEncounter_d.end(), -1); // or 999999.f if you prefer
    thrust::fill(csr.removed_d.begin(), csr.removed_d.end(), false); // or 999999.f if you prefer
*/
    resetM<<<dimGridM,THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(csr.edgeStatus_d.data()),
                                            thrust::raw_pointer_cast(csr.predecessors_d.data()),
                                            thrust::raw_pointer_cast(csr.budAtDDFSEncounter_d.data()),
                                            thrust::raw_pointer_cast(csr.bridgeTenacity_d.data()),
                                            csr.budCC.size,
                                            csr.budCC.directParent,
                                            csr.budCC.link,
                                            csr.budCC.groupRoot,
                                            2*csr.m);
  } while (bfsPassStructsDevGPUOnly(csr));
  //} while (bfs(csr));
  //thrust::copy(csr.mate_d.begin(), csr.mate_d.end(), mate.begin());
  //checkGraph();
}

int32_t main(int argc, char *argv[])
{
  ios::sync_with_stdio(false);
  std::string filename;
  int _depthCutoff=60;
  int _maxNumStacks=0;
  printf("You have entered %d arguments:\n", argc);
  if (argc < 2)
  {
    printf("Provide graph filename!\n");
    exit(1);
  }
  else
  {
    filename = argv[1];
    printf("Reading %s\n", filename.c_str());
  }
  if (argc >= 3){
    _depthCutoff = stoi(argv[2]);
    printf("Reading %d depth cutoff\n", _depthCutoff);
  }
  if (argc >= 4){
    _maxNumStacks = stoi(argv[3]);
    printf("Reading %d maxNumStacks cutoff\n", _maxNumStacks);
  }

  int ret_code, m;
  MM_typecode matcode;
  FILE *f;
  if ((f = fopen(filename.c_str(), "r")) == NULL)
  {
    exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0)
  {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */

  if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
      mm_is_sparse(matcode))
  {
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  /* compute size of sparse matrix .... */
  if ((ret_code = mm_read_mtx_crd_size(f, &n, &n, &m)) != 0)
  {
    printf("ERR CODE %d\n", ret_code);
    exit(1);
  }
  printf("N=%d M=%d\n", n, m);
  graph.resize(n);
  CSRGraph csr(n, m);
  csr.depthCutoff=_depthCutoff;
  GreedyMatcher gm(csr);
  BFS b(csr, gm);
  int readNum = 0;
  for (int i = 0; i < m; i++)
  {
    int a, b;
    fscanf(f, "%u%u", &a, &b);
    // printf("i=%d a=%d b=%d\n",i,a,b);
    a--;
    b--;
    graph[a].push_back(Edge(b, (int)graph[b].size()));
    graph[b].push_back(Edge(a, (int)graph[a].size() - 1));
    csr.rows_h[readNum] = a;
    csr.cols_h[readNum] = b;
    ++readNum;
    csr.rows_h[readNum] = b;
    csr.cols_h[readNum] = a;
    ++readNum;
  }

  csr.createOffsets();

  cudaDeviceSynchronize();

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("\nDevice name: %s\n\n", prop.name);

  int numOfMultiProcessors;
  cudaDeviceGetAttribute(&numOfMultiProcessors,cudaDevAttrMultiProcessorCount,0);
  printf("NumOfMultiProcessors : %d\n",numOfMultiProcessors);
  //numOfMultiProcessors = 1;
  if (!_maxNumStacks) _maxNumStacks=numOfMultiProcessors;
  csr.allocateMatchingDataStructures(csr.n,_maxNumStacks);
  size_t mf, ma;
  cudaMemGetInfo(&mf, &ma);
  long long maxGlobalMemory = ma;
	printf("maxGlobalMemory %lld\n",ma);
	long long consumedGlobalMem = ma-mf;
	printf("consumedGlobalMem %lld\n",consumedGlobalMem);
	long long availableGlobalMem = maxGlobalMemory - consumedGlobalMem;
	printf("availableGlobalMem %lld\n",availableGlobalMem);
  int maxThreadsPerMultiProcessor;
  cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor,cudaDevAttrMaxThreadsPerMultiProcessor,0);
  printf("MaxThreadsPerMultiProcessor : %d\n",maxThreadsPerMultiProcessor);

  int maxThreadsPerBlock;
  cudaDeviceGetAttribute(&maxThreadsPerBlock,cudaDevAttrMaxThreadsPerBlock,0);
  printf("MaxThreadsPerBlock : %d\n",maxThreadsPerBlock);

  int maxSharedMemPerMultiProcessor;
  cudaDeviceGetAttribute(&maxSharedMemPerMultiProcessor,cudaDevAttrMaxSharedMemoryPerMultiprocessor,0);
  printf("MaxSharedMemPerMultiProcessor : %d\n",maxSharedMemPerMultiProcessor);
  #if GLOBALMEMORYDSU
  #else
  csr.maxSM = maxSharedMemPerMultiProcessor-9000;
  csr.maxSM = maxSharedMemPerMultiProcessor-18000;
  #endif
  #if USE_DEVICE
  cudaError_t returnVal = cudaErrorInvalidValue;
  int thousandBytes = -1;
  while(returnVal){
      thousandBytes++;
      csr.maxSM = maxSharedMemPerMultiProcessor-(1000*thousandBytes);
      returnVal = cudaFuncSetAttribute(
      #if TIME
      processBridgesPassStructsTimed,
      #else
      processBridgesPassStructs,
      #endif
      cudaFuncAttributeMaxDynamicSharedMemorySize, 
      csr.maxSM);

      returnVal = cudaFuncSetAttribute(
      #if TIME
      processBridgesPassStructsParallelTimed,
      #else
      processBridgesPassStructs,
      #endif
      cudaFuncAttributeMaxDynamicSharedMemorySize, 
      csr.maxSM);
  }
  #else
  /*
  cudaError_t returnVal = cudaErrorInvalidValue;
  int thousandBytes = -1;
  while(returnVal){
      thousandBytes++;
      csr.maxSM = maxSharedMemPerMultiProcessor-(1000*thousandBytes);
      returnVal = cudaFuncSetAttribute(
      ddfsStructGlobal,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 
      csr.maxSM);
  }
  thousandBytes = -1;
  returnVal = cudaErrorInvalidValue;
  while(returnVal){
      thousandBytes++;
      csr.maxSM = maxSharedMemPerMultiProcessor-(1000*thousandBytes);
      returnVal = cudaFuncSetAttribute(
      updateLvlAndTenacityPassStruct_serial,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 
      csr.maxSM);
  }
    */

  int thousandBytes = -1;
  cudaError_t returnVal = cudaErrorInvalidValue;
  while(returnVal){
      thousandBytes++;
      csr.maxSM = maxSharedMemPerMultiProcessor-(1000*thousandBytes);
      returnVal = cudaFuncSetAttribute(
      augumentPathIterativeSwitchPassStructGlobal,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 
      csr.maxSM);
  }
  #endif
  printf("Removing some to get this running : %d\n",csr.maxSM);

  csr.stackDepth_h[0]=csr.maxSM/16;
  csr.stackDepth_d=csr.stackDepth_h;
  cudaMemcpyToSymbol(MAXSTACKAUGPATH_d, &csr.stackDepth_h[0], sizeof(int));
  #if GLOBALMEMORYDSU
  cudaMemcpyToSymbol(MAXSTACKDSU_d, &csr.maxStackDepth, sizeof(int));
  #else
  #endif

  /*
  CHECK_CUDA_ERROR(cudaFuncSetAttribute(
      augumentPathIterativeSwitchPassStructGlobal,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 
      csr.maxSM)
  );
  */
  #if PRINT
  printf("ROW %s %s %s %s %s %s %s %s\n","globalCounter", "iter", "depth","|avgCC|", "OverHeadElapsedMilliSeconds", "NumBridges","RootPairs","|CC|");
  #endif
  initializeMatching(csr, gm, b);
  mate.resize(n);
  thrust::copy(csr.mate_h.begin(),csr.mate_h.end(),mate.begin());
  std::chrono::time_point<std::chrono::steady_clock> m_StartTime = std::chrono::steady_clock::now();
  mvMatchingGPU(csr, gm, b);
  std::chrono::time_point<std::chrono::steady_clock> m_EndTime = std::chrono::steady_clock::now();
  double elapsedSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(m_EndTime - m_StartTime).count() / 1000.0;
  //int cnt = (n - count(mate.begin(), mate.end(), -1));
  //int cnt = thrust::count_if(csr.mate_d.begin(), csr.mate_d.end(), _1 > -1);
  int defect = thrust::count(csr.mate_d.begin(), csr.mate_d.end(), -1);
  int cnt = csr.n - defect;
  cout << cnt / 2 << endl;
  std::cout << "Seconds: " << elapsedSeconds << std::endl;
  m_StartTime = std::chrono::steady_clock::now();
  mvMatching(csr, gm, b);
  m_EndTime = std::chrono::steady_clock::now();
  double elapsedSecondsCPU = std::chrono::duration_cast<std::chrono::milliseconds>(m_EndTime - m_StartTime).count() / 1000.0;
  //int cnt = (n - count(mate.begin(), mate.end(), -1));
  //int cnt = thrust::count_if(csr.mate_d.begin(), csr.mate_d.end(), _1 > -1);
  int defectCPU = thrust::count(mate.begin(), mate.end(), -1);
  int cntCPU = csr.n - defect;
  cout << cntCPU / 2 << endl;
  std::cout << "Seconds CPU: " << elapsedSecondsCPU << std::endl;

  mate.clear();
  mate.resize(n,-1);
  m_StartTime = std::chrono::steady_clock::now();
  mvMatching(csr, gm, b);
  m_EndTime = std::chrono::steady_clock::now();
  double elapsedSecondsCPUEmpty = std::chrono::duration_cast<std::chrono::milliseconds>(m_EndTime - m_StartTime).count() / 1000.0;
  //int cnt = (n - count(mate.begin(), mate.end(), -1));
  //int cnt = thrust::count_if(csr.mate_d.begin(), csr.mate_d.end(), _1 > -1);
  int defectCPUEmpty = thrust::count(mate.begin(), mate.end(), -1);
  int cntCPUEmpty = csr.n - defectCPUEmpty;
  cout << cntCPUEmpty / 2 << endl;
  std::cout << "Seconds CPU: " << elapsedSecondsCPUEmpty << std::endl;


  //mvMatchingGPU(csr);
  char outputFilename[500];
  strcpy(outputFilename, "Results.csv");
  FILE *output_file;
  if (access(outputFilename, F_OK) == 0)
  {
    // file exists
    output_file = fopen(outputFilename, "a");
  }
  else
  {
    // file doesn't exist
    output_file = fopen(outputFilename, "w");
    fprintf(output_file, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", "Filename", "V","E","NUMSTACK","CUTOFF","GM_CNT","NaiveAug_CNT","MV_CNT","GPU_Match_Size","CPU_Match_Size","CPUEmpty_Match_Size", "GM_seconds","NaiveAug_seconds","MV_seconds","GPU_seconds","CPU_seconds","GPUInitializedTotal_seconds","CPUInitializedTotal_seconds","CPUEmpty_seconds");
  }
  fprintf(output_file,   "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f\n", filename.c_str(), csr.n, csr.m, csr.numStacks,csr.depthCutoff,csr.gm_cnt, csr.na_cnt, (cnt / 2) - csr.gm_cnt - csr.na_cnt,(cnt / 2),(cntCPU / 2), (cntCPUEmpty / 2), csr.gm_sec, csr.na_sec, elapsedSeconds-csr.gm_sec-csr.na_sec,elapsedSeconds,elapsedSecondsCPU,elapsedSeconds+csr.gm_sec+csr.na_sec,elapsedSecondsCPU+csr.gm_sec+csr.na_sec,elapsedSecondsCPUEmpty);
  fclose(output_file);
  /*
  csr.count_h = csr.count_d;
  csr.verticesTraversed_h = csr.verticesTraversed_d;
  csr.K1Time_h = csr.K1Time_d;
  csr.K2Time_h = csr.K2Time_d;
  csr.K3Time_h = csr.K3Time_d;
  #if TIME
  std::string outputFilename2="KernelBreakdown"+filename+".csv";
  FILE *output_file2;
  if (access(outputFilename2.c_str(), F_OK) == 0)
  {
    // file exists
    output_file = fopen(outputFilename2.c_str(), "a");
  }
  else
  {
    // file doesn't exist
    output_file = fopen(outputFilename2.c_str(), "w");
    fprintf(output_file, "%s %s %s %s %s %s %s %s\n", "Depth", "VerticesTraversed", "MeanK1Time", "MeanK2Time", "MeanK3Time", "K1Time", "K2Time", "K3Time");
  }
  for (int level = 1; level < csr.count_h.size(); ++level)
    fprintf(output_file, "%d %f %f %f %f %f %f %f\n", level, 
                                              (double)csr.verticesTraversed_h[level]/(double)csr.count_h[level],
                                              (double)csr.K1Time_h[level]/(double)csr.count_h[level],
                                              (double)csr.K2Time_h[level]/(double)csr.count_h[level],
                                              (double)csr.K3Time_h[level]/(double)csr.count_h[level],
                                              (double)csr.K1Time_h[level],
                                              (double)csr.K2Time_h[level],
                                              (double)csr.K3Time_h[level]);
  fclose(output_file);
  #endif
  */
  return 0;
}
