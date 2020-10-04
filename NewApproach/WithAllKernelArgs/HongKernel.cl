//ACL Kernel
#define IDX(i, j, n) ((i) * (n) + (j))
__kernel 
void SimpleKernel(__global int * restrict residualFlow, uint n,__global int * restrict height,
__global int * restrict excessFlow,__global int * restrict netFlowOutS,
__global int * restrict netFlowInT,uint s,uint t)
{
	int gid = get_global_id(0);
    
	while (*netFlowOutS != *netFlowInT) {
	//for(int x=1;x<40;x++ )  { 
        	if (gid != s && gid != t && excessFlow[gid] > 0) {
		
            		int curExcess = excessFlow[gid];
            		int curLowestNeighbor = -1;
            		int neighborMinHeight = 10000;
            		for (int v = 0; v < n; v++) {
                		if (gid == v) continue;
                		if (residualFlow[IDX(gid, v, n)] > 0) {
                    		int tempHeight = height[v];
                    			if (tempHeight < neighborMinHeight) {
                        		curLowestNeighbor = v;
                        		neighborMinHeight = tempHeight;
                  			}
                		}
            		}
			
         		if (height[gid] > neighborMinHeight) {
			    
				
                		int delta = min(curExcess, residualFlow[IDX(gid, curLowestNeighbor, n)]);
				
                		atomic_sub(&residualFlow[IDX(gid, curLowestNeighbor, n)], delta);
                		atomic_add(&residualFlow[IDX(curLowestNeighbor, gid, n)], delta);
                		atomic_sub(&excessFlow[gid], delta);
                		atomic_add(&excessFlow[curLowestNeighbor], delta);
                		if (curLowestNeighbor == s) {
				    
                    		atomic_sub(netFlowOutS, delta);
                		} 
						else if (curLowestNeighbor == t) {
				   
                    		atomic_add(netFlowInT, delta);
                		}
            		} 

				else 	{
                		height[gid] = neighborMinHeight + 1;
            			}
			
        	}//if 
	}//while    
}


