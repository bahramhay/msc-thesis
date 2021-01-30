//ACL Kernel
#define IDX(i, j, n) ((i) * (n) + (j))
//#include<stdlib.h>

__kernel void PushKernel(__global int * restrict residualFlow, uint column,__constant uint * restrict height,
__global int * restrict excessFlow,__global int * restrict netFlowOutS,
__global int * restrict netFlowInT,uint s,uint t,uint row)
{
	/*int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int globalId=IDX(gidy, gidx, column);
	uint total_nodes=column*row;
	int groupidx = get_group_id(0);
	for (int v = 0; v < total_nodes; v++) 
	{
		residualFlow[IDX(globalId, v, total_nodes)] = residualFlow[IDX(globalId, v, total_nodes)]+groupidx;
	}
	height[globalId] = height[globalId]+1;
	excessFlow[globalId] = excessFlow[globalId]+groupidx;
	*netFlowOutS = *netFlowOutS + 10;
	*netFlowInT = *netFlowInT + 10;*/
	int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int global_id=IDX(gidy, gidx, column);
	
	uint total_nodes=column*row;

	//height block that is 1 size bigger than original block at every edge
	int const hb_width=4;
	int const hb_depth=4;
	//__local uint heights_block[10][10] __attribute__((numbanks(16),bankwidth(64)));
	__local uint  __attribute__((memory,
bank_bits(3,2),
//numbanks(4),
//bankwidth(16),
doublepump
//numreadports(8),
//numwriteports(1)
)) 
heights_block[4][4];
	int num_blockx=column/(hb_width-2);
	int num_blocky=row/(hb_depth-2);
	
	//local ids for block
	int lidx=get_local_id(0);
	int lidy=get_local_id(1);
	int groupidx = get_group_id(0);
	int groupidy = get_group_id(1);
	//int local_hb_id=IDX(lidy+1,lidx+1,hb_width);
	int node_x=lidx+1;
	int node_y=lidy+1;
	int up_x=lidx+1;
	int up_y=lidy;
	int down_x=lidx+1;
	int down_y=lidy+2;
	int right_x=lidx+2;
	int right_y=lidy+1;
	int left_x=lidx;
	int left_y=lidy+1;
	int group_id=IDX(groupidy,groupidx,num_blockx);

	//input height to local memory

	heights_block[node_x][node_y]=height[global_id];
	//top of the block
	if(lidy==0 && gidy > 0){
		heights_block[up_x][up_y]=height[global_id - column];
	}
	//bottom of the block
	if(lidy==(hb_depth-2)-1 && gidy < row-1){
		heights_block[down_x][down_y]=height[global_id + column];
	}
	//right of the block
	if(lidx==(hb_width-2)-1 && gidx < column-1){
		heights_block[right_x][right_y]=height[global_id + 1];
	}
	//left of the block
	if(lidx==0 && gidx > 0){
		heights_block[left_x][left_y]=height[global_id - 1];
	}
	barrier (CLK_LOCAL_MEM_FENCE);
	//atom_add(&in[IDX(gidy, gidx, n)],4);
	
	//push operation
	if (global_id != s && global_id != t){
		int curExcess=excessFlow[global_id];
		int curCapacity=0;

		//push to up
		if(gidy>0){
			curCapacity = residualFlow[IDX(global_id, (global_id - column), total_nodes)];
			if(curCapacity>0 && curExcess>0 && heights_block[node_x][node_y]==heights_block[up_x][up_y]+1){
				int delta = min(curExcess, curCapacity);
		          	atom_sub(&residualFlow[IDX(global_id, (global_id - column), total_nodes)], delta);
                		atom_add(&residualFlow[IDX((global_id - column), global_id, total_nodes)], delta);
                		atom_sub(&excessFlow[global_id], delta);
                		atom_add(&excessFlow[(global_id - column)], delta);

				if ((global_id - column) == s) {atom_sub(netFlowOutS, delta);} 
				else if ((global_id - column) == t) {atom_add(netFlowInT, delta);}				
			}
		}
		
		//push to right
		curExcess=excessFlow[global_id];
		if(gidx<column-1){
			curCapacity = residualFlow[IDX(global_id, (global_id + 1), total_nodes)];
			if(curCapacity>0 && curExcess>0 && heights_block[node_x][node_y]==heights_block[right_x][right_y]+1){
				int delta = min(curExcess, curCapacity);
		          	atom_sub(&residualFlow[IDX(global_id, (global_id + 1), total_nodes)], delta);
                		atom_add(&residualFlow[IDX((global_id + 1), global_id, total_nodes)], delta);
                		atom_sub(&excessFlow[global_id], delta);
                		atom_add(&excessFlow[(global_id + 1)], delta);	

				if ((global_id + 1) == s) {atom_sub(netFlowOutS, delta);} 
				else if ((global_id + 1) == t) {atom_add(netFlowInT, delta);}			
			}
		}

		//push to down
		curExcess=excessFlow[global_id];
		if(gidy<row-1){
			curCapacity = residualFlow[IDX(global_id, (global_id + column), total_nodes)];
			if(curCapacity>0 && curExcess>0 && heights_block[node_x][node_y]==heights_block[down_x][down_y]+1){
				int delta = min(curExcess, curCapacity);
		          	atom_sub(&residualFlow[IDX(global_id, (global_id + column), total_nodes)], delta);
                		atom_add(&residualFlow[IDX((global_id + column), global_id, total_nodes)], delta);
                		atom_sub(&excessFlow[global_id], delta);
                		atom_add(&excessFlow[(global_id + column)], delta);	

				if ((global_id + column) == s) {atom_sub(netFlowOutS, delta);} 
				else if ((global_id + column) == t) {atom_add(netFlowInT, delta);}				
			}
		}

		//push to left
		curExcess=excessFlow[global_id];
		if(gidx>0){
			curCapacity = residualFlow[IDX(global_id, (global_id - 1), total_nodes)];
			if(curCapacity>0 && curExcess>0 && heights_block[node_x][node_y]==heights_block[left_x][left_y]+1){
				int delta = min(curExcess, curCapacity);
		          	atom_sub(&residualFlow[IDX(global_id, (global_id - 1), total_nodes)], delta);
                		atom_add(&residualFlow[IDX((global_id - 1), global_id, total_nodes)], delta);
                		atom_sub(&excessFlow[global_id], delta);
                		atom_add(&excessFlow[(global_id - 1)], delta);	

				if ((global_id - 1) == s) {atom_sub(netFlowOutS, delta);} 
				else if ((global_id - 1) == t) {atom_add(netFlowInT, delta);}				
			}
		}
		
	}
}

/*__kernel void RelabelKernel(__global int * restrict residualFlow, uint column,__global uint * restrict height,
__global int * restrict excessFlow,__global int * restrict netFlowOutS,
__global int * restrict netFlowInT,uint s,uint t,uint row)
{
	int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int global_id=IDX(gidy, gidx, column);
	
	uint total_nodes=column*row;

	//hight block that is 1 size bigger than original block at every edge
	int const hb_width=10;
	int const hb_depth=10;
	__local uint heights_block[hb_width*hb_width];
	int num_blockx=column/(hb_width-2);
	int num_blocky=row/(hb_depth-2);
	
	//local ids for block
	int lidx=get_local_id(0);
	int lidy=get_local_id(1);
	int groupidx = get_group_id(0);
	int groupidy = get_group_id(1);
	int local_hb_id=IDX(lidy+1,lidx+1,hb_width);
	int group_id=IDX(groupidy,groupidx,num_blockx);

	//input height to local memory

	heights_block[local_hb_id]=height[global_id];
	//top of the block
	if(lidy==0 && gidy > 0){
		heights_block[local_hb_id - hb_width]=height[global_id - column];
	}
	//bottom of the block
	if(lidy==(hb_depth-2)-1 && gidy < row-1){
		heights_block[local_hb_id + hb_width]=height[global_id + column];
	}
	//right of the block
	if(lidx==(hb_width-2)-1 && gidx < column-1){
		heights_block[local_hb_id + 1]=height[global_id + 1];
	}
	//left of the block
	if(lidx==0 && gidx > 0){
		heights_block[local_hb_id - 1]=height[global_id - 1];
	}
	barrier (CLK_LOCAL_MEM_FENCE);
	
	//relable operation
	int curExcess=excessFlow[global_id];
	int upres;
	int downres;
	int rightres;
	int leftres;
	int curLowestNeighbor = -1;
	uint neighborMinHeight = 4294967295;
	uint neighbourh;
	if(global_id != s && global_id != t){
	if(curExcess>0){
		//begin from up
		if(gidy>0){
			upres=residualFlow[IDX(global_id, (global_id - column), total_nodes)];
			//check weight
			if(upres>0){
				neighbourh = heights_block[local_hb_id - hb_width];
				if(heights_block[local_hb_id]==neighbourh+1) {goto END;} 
				else{//do compare
		    	            	if (neighbourh < neighborMinHeight) {
	      	               		curLowestNeighbor = (local_hb_id - hb_width);
        	                	neighborMinHeight = neighbourh;
        	          		}			
				}			
			}
		}
		//then down
		if(gidy<row-1){
			downres=residualFlow[IDX(global_id, (global_id + column), total_nodes)];
			//check weight
			if(downres>0){
				neighbourh = heights_block[local_hb_id + hb_width];
				if(heights_block[local_hb_id]==neighbourh+1){goto END;} 
				else{//do compare
		    	            	if (neighbourh < neighborMinHeight) {
	      	               		curLowestNeighbor = (local_hb_id + hb_width);
        	                	neighborMinHeight = neighbourh;
        	          		}			
				}			
			}
		}
		//then right
		if(gidx<column-1){
			rightres=residualFlow[IDX(global_id, (global_id + 1), total_nodes)];
			//check weight
			if(rightres>0){
				neighbourh = heights_block[local_hb_id + 1];
				if(heights_block[local_hb_id]==neighbourh+1){goto END;} 
				else{//do compare
		    	            	if (neighbourh < neighborMinHeight) {
	      	               		curLowestNeighbor = (local_hb_id + 1);
        	                	neighborMinHeight = neighbourh;
        	          		}			
				}			
			}
		}
		//then left
		if(gidx>0){
			leftres=residualFlow[IDX(global_id, (global_id - 1), total_nodes)];
			//check weight
			if(leftres>0){
				neighbourh = heights_block[local_hb_id - 1];
				if(heights_block[local_hb_id]==neighbourh+1){goto END;} 
				else{//do compare
		    	            	if (neighbourh < neighborMinHeight) {
	      	               		curLowestNeighbor = (local_hb_id - 1);
        	                	neighborMinHeight = neighbourh;
        	          		}			
				}			
			}
		}
		//relable
		if(curLowestNeighbor != -1){
			height[global_id] = neighborMinHeight + 1;
		}
	END:;}
	}
}
*/

