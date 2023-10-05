#ifndef CONSTANTS_H
#define CONSTANTS_H
#define MAX_THREADS_PER_GRID (2 * *31)
#define THREADS_PER_WARP 32
#define WARP_SZ 32
__device__ inline int lane_id(void) { return threadIdx.x % WARP_SZ; }
#define THREADS_PER_BLOCK 1024
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / THREADS_PER_WARP)
#define I_SIZE ((3 / 2) * THREADS_PER_BLOCK)

const int INF = 1e9;
enum EdgeType
{
  NotScanned,
  Prop,
  Bridge
};
#endif