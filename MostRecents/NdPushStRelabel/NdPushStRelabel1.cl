//ACL Kernel
#define IDX(i, j, n) ((i) * (n) + (j))
//#include<stdlib.h>

__kernel void PushKernel( uint column,__constant int * restrict height,
__global int * restrict excessFlow,__global int * restrict netFlowOutS,
__global int * restrict netFlowInT,uint s,uint t,uint row,
__global int * restrict residualFlow_up,__global int * restrict residualFlow_down,
__global int * restrict residualFlow_right,__global int * restrict residualFlow_left)
{
const uint num_column=16;
const uint num_row=10;
	int lidx=get_local_id(0);
	int lidy=get_local_id(1);
	int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int groupidx = get_group_id(0);
	int groupidy = get_group_id(1);
	int global_id=IDX(gidy, gidx, column);

	uint total_nodes=column*row;

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

	__local int heights_block[4][4] __attribute__((memory,bank_bits(3,2)
,doublepump
,numreadports(8)
,numwriteports(2)
))
;
	if(lidx==0 && lidy==0){
		#pragma unroll
		for (int i = 0; i < 2; i++) {
		
		heights_block[1+i][1]=height[global_id+i*column];
		heights_block[1+i][2]=height[global_id+i*column+1];
		
		}
	
	//rast
	if(groupidx < (num_column/2)-1){
		#pragma unroll
		for (int i = 0; i < 2; i++) {
			heights_block[1+i][3]=height[global_id+i*column+2];
		}
	}

	//bala
	if(groupidy > 0){
		#pragma unroll
		for (int i = 0; i < 2; i++) {
			heights_block[0][1+i]=height[global_id-column+i];
		}
	}

	//payin
	if(groupidy < (num_row/2)-1){
		#pragma unroll
		for (int i = 0; i < 2; i++) {
			heights_block[3][1+i]=height[global_id+2*column+i];
		}
	}

	//chap
	if(groupidx > 0){
		#pragma unroll
		for (int i = 0; i < 2; i++) {
			heights_block[1+i][0]=height[global_id+i*column-1];
		}
	}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	//for(int i = 0; i < 4; i++){
	//for(int j = 0; j < 4; j++){
	//height[IDX(i, j, column)]=heights_block[i][j];}}
	//if(lidx==0 && lidy==0)	
//	{
	//height[global_id]=heights_block[node_y][node_x];
	//}	
	//push operation
	if (global_id != s && global_id != t){
		int curExcess=excessFlow[global_id];
		int curCapacity=0;
int dif1;
		//push to up
		if(gidy>0){
			curCapacity = residualFlow_up[global_id];
			if(curCapacity>0 && curExcess>0 && heights_block[node_y][node_x]==heights_block[up_y][up_x]+1){
//printf("oomadam \n");
				int delta = min(curExcess, curCapacity);
		          	atom_sub(&residualFlow_up[global_id], delta);
                		atom_add(&residualFlow_down[(global_id - column)], delta);
                		atom_sub(&excessFlow[global_id], delta);
                		atom_add(&excessFlow[(global_id - column)], delta);

				if ((global_id - column) == s) {atom_sub(netFlowOutS, delta);} 
				else if ((global_id - column) == t) {atom_add(netFlowInT, delta);}				
			}
		}
		
		//push to right
		curExcess=excessFlow[global_id];
		if(gidx<column-1){
			curCapacity = residualFlow_right[global_id];
			if(curCapacity>0 && curExcess>0 && heights_block[node_y][node_x]==heights_block[right_y][right_x]+1){
//printf("oomadam \n");
				int delta = min(curExcess, curCapacity);

		          	atom_sub(&residualFlow_right[global_id], delta);

                		atom_add(&residualFlow_left[(global_id + 1)], delta);
                		atom_sub(&excessFlow[global_id], delta);
                		atom_add(&excessFlow[(global_id + 1)], delta);	

				if ((global_id + 1) == s) {atom_sub(netFlowOutS, delta);} 
				else if ((global_id + 1) == t) {atom_add(netFlowInT, delta);}			
			}
		}

		//push to down
		curExcess=excessFlow[global_id];
		if(gidy<row-1){
			curCapacity = residualFlow_down[global_id];
			if(curCapacity>0 && curExcess>0 && heights_block[node_y][node_x]==heights_block[down_y][down_x]+1){
//printf("oomadam \n");
				int delta = min(curExcess, curCapacity);
		          	atom_sub(&residualFlow_down[global_id], delta);
                		atom_add(&residualFlow_up[(global_id + column)], delta);
                		atom_sub(&excessFlow[global_id], delta);
                		atom_add(&excessFlow[(global_id + column)], delta);	

				if ((global_id + column) == s) {atom_sub(netFlowOutS, delta);} 
				else if ((global_id + column) == t) {atom_add(netFlowInT, delta);}				
			}
		}

		//push to left
		curExcess=excessFlow[global_id];
		if(gidx>0){
			curCapacity = residualFlow_left[global_id];
//if(curExcess>0&& heights_block[node_y][node_x]==heights_block[left_y][left_x]+1){printf("curExcess:%d  curCapacity:%d \n",curExcess,curCapacity);}
			if(curCapacity>0 && curExcess>0 && heights_block[node_y][node_x]==heights_block[left_y][left_x]+1){
//printf("oomadam \n");
				int delta = min(curExcess, curCapacity);
		          	atom_sub(&residualFlow_left[global_id], delta);

                		atom_add(&residualFlow_right[(global_id - 1)], delta);

//printf("residualFlow_right[global_id - 1]:%d  \n",residualFlow_right[global_id - 1]);
//if( residualFlow_right[global_id - 1]>0){printf("global_id:%d  \n",global_id);}
                		atom_sub(&excessFlow[global_id], delta);
                		atom_add(&excessFlow[(global_id - 1)], delta);	

				if ((global_id - 1) == s) {atom_sub(netFlowOutS, delta);} 
				else if ((global_id - 1) == t) {atom_add(netFlowInT, delta);}				
			}
		}
	
	}

}


__kernel void RelableKernel( uint column,__global int * restrict height,
__global int * restrict excessFlow,__global int * restrict netFlowOutS,
__global int * restrict netFlowInT,uint s,uint t,uint row,
__global int * restrict residualFlow_up,__global int * restrict residualFlow_down,
__global int * restrict residualFlow_right,__global int * restrict residualFlow_left
)
{
	__local int heights_cache[16*10];
	for(int i=0; i<row*column; i++){
		heights_cache[i]=height[i];
	}
	/*__local int excessFlow_cache[24];
	for(int i=0; i<row*column; i++){
		excessFlow_cache[i]=excessFlow[i];
	}*/
	//////////////////////////////////////////
	for(int i=0; i<row*column; i++){
		int curLowestNeighbor = -1;
		int neighborMinHeight = 2147483647;

		int neighbourh;
		if(i != s && i != t && excessFlow[i]>0){
/*if(i==17){printf("excess:%d  \n",excessFlow[i]);
printf("h:%d  \n",heights_cache[i]);
printf("res_r:%d  \n",residualFlow_right[i]);
printf("res_l:%d  \n",residualFlow_left[i]);
printf("res_u:%d  \n",residualFlow_up[i]);
printf("res_d:%d  \n",residualFlow_down[i]);
}*/
			if(residualFlow_right[i]>0){

				neighbourh = heights_cache[i+1];
				
				//do compare

		    	        if (neighbourh < neighborMinHeight) {
	      	               	curLowestNeighbor = i+1;
        	                neighborMinHeight = neighbourh;
        	          	}			
							
			}
			if(residualFlow_left[i]>0){
				neighbourh = heights_cache[i-1];
				
				//do compare
		    	        if (neighbourh < neighborMinHeight) {
	      	               	curLowestNeighbor = i-1;
        	                neighborMinHeight = neighbourh;
        	          	}			
							
			}
			if(residualFlow_up[i]>0){
				neighbourh = heights_cache[i-column];
				
				//do compare
		    	        if (neighbourh < neighborMinHeight) {
	      	               	curLowestNeighbor = i-column;
        	                neighborMinHeight = neighbourh;
        	          	}			
							
			}
			if(residualFlow_down[i]>0){
				neighbourh = heights_cache[i+column];
				
				//do compare
		    	        if (neighbourh < neighborMinHeight) {
	      	               	curLowestNeighbor = i+column;
        	                neighborMinHeight = neighbourh;
        	          	}			
							
			}
		}
		//relable
		if(curLowestNeighbor != -1){
			height[i] = neighborMinHeight + 1;
		}
	}
	/*for(int i=0; i<row*column; i++){
		height[i]=heights_cache[i];
	}*/
}
