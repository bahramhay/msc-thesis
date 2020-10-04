//ACL Kernel
#define IDX(i, j, n) ((i) * (n) + (j))
__kernel 
void SimpleKernel(__global const int *restrict in,  __global int *restrict out, uint n)
{
	//Perform the Math Operation
	size_t gid = get_global_id(0);
	
	for (int v = 0; v < n; v++) 
	{
		out[IDX(gid, v, n)] = in[IDX(gid, v, n)] + 1;
	}
}

/*__kernel 
void MulKernel(__global const float *restrict in, __global const float *restrict in2, __global float * restrict out)
{
	//Perform the Math Operation
	size_t i = get_global_id(0);
	out[i] = in[i] * in2[i];
}*/

