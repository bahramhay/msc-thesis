#define IDX(i, j, n) ((i) * (n) + (j))
__kernel void hello_kernel(__global int *residualFlow,
						__global int *height,
						//__global int *result,
						__global int *excessFlow,
						__global int *FOut,
						__global int *FIn,
						__global int *s,
						__global int *t,
						__global int *n)
{
      int gid = get_global_id(0);
    
	while (*FOut != *FIn) {
        
	//for(int x=1;x<10000000;x++ )  { 
        if (gid != *s && gid != *t && excessFlow[gid] > 0) {
		    
            int curExcess = excessFlow[gid];
            int curLowestNeighbor = -1;
            int neighborMinHeight = 10000;
            for (int v = 0; v < *n; v++) {
                if (gid == v) continue;
                if (residualFlow[IDX(gid, v, *n)] > 0) {
                    int tempHeight = height[v];
                    if (tempHeight < neighborMinHeight) {
                        curLowestNeighbor = v;
                        neighborMinHeight = tempHeight;
                    }
                }
            }
			
            if (height[gid] > neighborMinHeight) {
			    
                int delta = min(curExcess, residualFlow[IDX(gid, curLowestNeighbor, *n)]);
				
                atomic_sub(&residualFlow[IDX(gid, curLowestNeighbor, *n)], delta);
                atomic_add(&residualFlow[IDX(curLowestNeighbor, gid, *n)], delta);
                atomic_sub(&excessFlow[gid], delta);
                atomic_add(&excessFlow[curLowestNeighbor], delta);
                if (curLowestNeighbor == *s) {
				    
                    atomic_sub(FOut, delta);
                } else if (curLowestNeighbor == *t) {
				    
                    atomic_add(FIn, delta);
                }
            } else {
                height[gid] = neighborMinHeight + 1;
            }
			
        }//if 
	

    }//while 
}