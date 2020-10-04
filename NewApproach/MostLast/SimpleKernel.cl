//ACL Kernel
#define IDX(i, j, n) ((i) * (n) + (j))
/*__kernel 
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

			else if(curLowestNeighbor != -1)	{
                		height[gid] = neighborMinHeight + 1;
            			}
			
        	}//if 
	}//while
		
}*/

__kernel void PushKernel(__global int * restrict residualFlow,uint n,__global int * restrict height,
__global int * restrict excessFlow,uint s,uint t,__global int * restrict specialExcess,
__global int * restrict netFlowOutS,__global int * restrict netFlowInT)
{
	int gid = get_global_id(0);
	
	if (gid != s && gid != t && excessFlow[gid] > 0)
		{
			int curExcess = excessFlow[gid];
            		int curNeighbor = -1;
            		int curHeight= height[gid];
            		for (int v = 0; v < n; v++) {
                		if (gid == v) continue;
                		if (residualFlow[IDX(gid, v, n)] > 0) {
                    		int neighborHeight = height[v];
                    			if (curHeight == (neighborHeight+1)) {
						curNeighbor = v;
                        			int delta = min(curExcess, residualFlow[IDX(gid, curNeighbor, n)]);
						residualFlow[IDX(gid, curNeighbor, n)]=residualFlow[IDX(gid, curNeighbor, n)]-delta;
						residualFlow[IDX(curNeighbor, gid, n)]=residualFlow[IDX(curNeighbor, gid, n)]+delta;
						excessFlow[gid]=excessFlow[gid]-delta;
						specialExcess[IDX(gid, curNeighbor, n)]=specialExcess[IDX(gid, curNeighbor, n)]+delta;

						if (curNeighbor == s) {
				    
                    					*netFlowOutS=*netFlowOutS-delta;
                				} 
						else if (curNeighbor == t) {
				   
                    					*netFlowInT=*netFlowInT+delta;
                				}
						//break;
						curExcess=curExcess-delta;
						if(curExcess==0){break;}
                  			}
                		}
            		}
		}
}
__kernel void PullKernel(uint n,
__global int * restrict excessFlow,uint s,uint t,__global int * restrict specialExcess)
{
	int gid = get_global_id(0);
	//if (gid != s && gid != t){
		for (int v = 0; v < n; v++){
			if (gid == v) continue;
			if (specialExcess[IDX(v, gid, n)] > 0) {
				excessFlow[gid]=excessFlow[gid]+specialExcess[IDX(v, gid, n)];
				specialExcess[IDX(v, gid, n)]=0;
			}
		}
	//}
	//else if (gid == s || gid == t){
		//for (int v = 0; v < n; v++){
			//specialExcess[IDX(v, gid, n)]=0;
		//}
	//}
}
__kernel void RelabelKernel(uint n,__global int * restrict residualFlow,
__global int * restrict height,uint s,uint t,__global int * restrict excessFlow)
{
	int gid = get_global_id(0);
	if (gid != s && gid != t && excessFlow[gid] > 0){
		int curLowestNeighbor = -1;
        	int neighborMinHeight = 10000;
		for (int v = 0; v < n; v++) {
        		if (gid == v) continue;
        	        if (residualFlow[IDX(gid, v, n)] > 0) {
        	            	int tempHeight = height[v];
        	            	if (tempHeight < neighborMinHeight) {
					//height[gid] = neighborMinHeight + 1;
        	                	curLowestNeighbor = v;
        	                	neighborMinHeight = tempHeight;
        	          	}
			}
		}
		if(curLowestNeighbor != -1){
			height[gid] = neighborMinHeight + 1;
		}
	}
}	
