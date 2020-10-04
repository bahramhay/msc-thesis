//ACL Kernel
#define IDX(i, j, n) ((i) * (n) + (j))
__kernel 
void SimpleKernel(__global int * restrict residualFlow, uint n,__global int * restrict height,
__global int * restrict excessFlow,__global int * restrict netFlowOutS,
__global int * restrict netFlowInT,uint s,uint t)
{
	//Perform the Math Operation
	size_t gid = get_global_id(0);
	
	for (int v = 0; v < n; v++) 
	{
		residualFlow[IDX(gid, v, n)] = residualFlow[IDX(gid, v, n)] + 1;
	}
	height[gid] = height[gid] + 1;
	excessFlow[gid] = excessFlow[gid] + 1;
	*netFlowOutS = *netFlowOutS + 10;
	*netFlowInT = *netFlowInT + 10;
}


