//fixed 4 step reduce
void float_atomic_add(__global float *loc, const float f)
{
	private float old = *loc;
	private float sum = old + f;
	while(atomic_cmpxchg((__global int*)loc, *((int*)&old), *((int*)&sum)) != *((int*)&old))
	{
		old = *loc;
		sum = old + f;
	}
}

void cmpxchg(__global float* A, __global float* B, bool dir) 
{
	if ((!dir && *A > *B) || (dir && *A < *B)) 
	{
		float t = *A;
		*A = *B;
		*B = t;
	}
}

void bitonic_merge(int id, __global float* A, int N, bool dir) 
{
	for (int i = N/2; i > 0; i/=2) 
	{
		if ((id % (i*2)) < i)
			cmpxchg(&A[id],&A[id+i],dir);

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

__kernel void sort_bitonic(__global float* A) 
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = 1; i < N/2; i*=2) 
	{
		if (id % (i*4) < i*2)
			bitonic_merge(id, A, i*2, false);
		else
		{
			if ((id + i*2) % (i*4) < i*2)
				bitonic_merge(id, A, i*2, true);
		}

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	bitonic_merge(id,A,N,false);
}

__kernel void reduce_add(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) 
	{
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		float_atomic_add(&B[0],scratch[lid]);
	}
}

__kernel void reduce_min(__global const float* A, __global float* B, __local float* scratch) 
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N) && (scratch[lid] > scratch[lid+1]))
				scratch[lid] = scratch[lid + 1];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) 
	{
		if ((scratch[lid] < B[0]))
			atomic_xchg(&B[0],scratch[lid]);
	}
}

__kernel void reduce_max(__global const float* A, __global float* B, __local float* scratch) 
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N) && (scratch[lid] > scratch[lid+1]))
				scratch[lid] = scratch[lid + 1];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) 
	{
		if ((scratch[lid] > B[0]))
			atomic_xchg(&B[0],scratch[lid]);
	}
}

__kernel void reduce_std(__global const float* A, __global float* B, __local float* scratch, __global float* Mean) 
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
				scratch[lid] += ( (scratch[lid + 1] - Mean[0]) * (scratch[lid + 1] - Mean[0]) ) / 100;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) 
	{
			float_atomic_add(&B[0], scratch[lid]);
	}
}
