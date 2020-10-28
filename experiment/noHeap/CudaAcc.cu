#include <OptiX/_Define_7_Device.h>

__global__ void initRandom(curandState* state, unsigned int seed, unsigned int MaxNum)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < MaxNum)
		curand_init(seed, id, 0, state + id);
}
void initRandom(curandState* state, int seed, unsigned int block, unsigned int grid, unsigned int MaxNum)
{
	initRandom << <grid, block >> > (state, seed, MaxNum);
}