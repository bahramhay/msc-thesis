//ACL Kernel
#define IDX(i, j, n) ((i) * (n) + (j))
//#include<stdlib.h>

//__attribute__((num_simd_work_items(2)))__attribute__((reqd_work_group_size(1,2,1)))
__kernel void PushKernel(__global int * restrict residualFlow, uint column,__global uint * restrict height,
__global int * restrict excessFlow,__global int * restrict netFlowOutS,
__global int * restrict netFlowInT,uint s,uint t,uint row)
{
	int lidx=get_local_id(0);
	int lidy=get_local_id(1);
	int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int groupidx = get_group_id(0);
	int groupidy = get_group_id(1);
	
	uint total_nodes=column*row;
	int nodes_in_wi_x=2;
	int nodes_in_wi_y=1;
	int groupids_x=column/nodes_in_wi_x;
	int groupids_y=row/nodes_in_wi_y;
	
	int global_id=IDX(gidy, gidx, groupids_x);
	int first_node_x=nodes_in_wi_x*lidx+1;
	int first_node_y=nodes_in_wi_y*lidy+1;
	
	int first_up_x=first_node_x;
	int first_up_y=first_node_y-1;
	int first_right_x=first_node_x+1;
	int first_right_y=first_node_y;
	int first_down_x=first_node_x;
	int first_down_y=first_node_y+1;
	int first_left_x=first_node_x-1;
	int first_left_y=first_node_y;

	__local int heights_block[4][4] 
__attribute__((
memory
,bank_bits(3,2)
//,numbanks(1)
,bankwidth(16)
,doublepump
,numreadports(4)
,numwriteports(1)
))
;

	if(lidx==0 && lidy==0){
		#pragma unroll
		for (int i = 0; i < 2; i++) {
		heights_block[1][1+i]=height[global_id*nodes_in_wi_x+i];
		heights_block[2][1+i]=height[(global_id*nodes_in_wi_x)+column+i];
		}

		//up		
		if(groupidy>0){
			#pragma unroll
			for (int i = 0; i < 2; i++) {
			heights_block[0][1+i]=height[(global_id*nodes_in_wi_x)-column+i];
			}
		}
		else{
			#pragma unroll
			for (int i = 0; i < 2; i++) {
			heights_block[0][1+i]=0;
			}
		}
		heights_block[0][0]=0;
		heights_block[0][3]=0;
		
		//down
		if(groupidy<3){
			#pragma unroll
			for (int i = 0; i < 2; i++) {
			heights_block[3][1+i]=height[(global_id*nodes_in_wi_x)+2*column+i];
			}
		}
		else{
			#pragma unroll
			for (int i = 0; i < 2; i++) {
			heights_block[3][1+i]=0;
			}
		}
		heights_block[3][0]=0;
		heights_block[3][3]=0;
		//left
		if(groupidx>0){
			#pragma unroll
			for (int i = 0; i < 2; i++) {
				heights_block[1+i][0]=height[(global_id*nodes_in_wi_x)+i*column-1];
			}			
		}
		else{
			#pragma unroll
			for (int i = 0; i < 2; i++) {
			heights_block[1+i][0]=0;
			}
		}
		//right
		if(groupidx<2){
			#pragma unroll
			for (int i = 0; i < 2; i++) {
				heights_block[1+i][3]=height[(global_id*nodes_in_wi_x)+i*column+2];
			}			
		}
		else{
			#pragma unroll
			for (int i = 0; i < 2; i++) {
			heights_block[1+i][3]=0;
			}
		}
	
	}
	barrier(CLK_LOCAL_MEM_FENCE);	
	
	//height[global_id*nodes_in_wi_x]=heights_block[first_node_y][first_node_x];
	//height[global_id*nodes_in_wi_x+1]=heights_block[first_node_y][first_node_x+1];
///////////////////////////////////////////////////////////////////////////////////////////////////
	#pragma unroll
	for(int i = 0; i < nodes_in_wi_x; i++){	
		int me, up, right, down, left;
		
		left=heights_block[first_node_y][first_node_x-1+i];
		me=heights_block[first_node_y][first_node_x+i];
		right=heights_block[first_node_y][first_node_x+1+i];
		
		up=heights_block[first_node_y-1][first_node_x+i];
		down=heights_block[first_node_y+1][first_node_x+i];

		height[global_id*nodes_in_wi_x+1+i]=left+me+right+up+down;
		/*//push operation
		if ((global_id*nodes_in_wi_x)+i != s && (global_id*nodes_in_wi_x)+i != t){
			int curExcess=excessFlow[(global_id*nodes_in_wi_x)+i];
			int curCapacity=0;

			//push to up
			if(gidy>0){
				curCapacity = residualFlow[IDX((global_id*nodes_in_wi_x)+i, ((global_id*nodes_in_wi_x)+i - column), total_nodes)];
				if(curCapacity>0 && curExcess>0 && me==up+1){
					int delta = min(curExcess, curCapacity);
				  	atom_sub(&residualFlow[IDX((global_id*nodes_in_wi_x)+i, ((global_id*nodes_in_wi_x)+i - column), total_nodes)], delta);
		        		atom_add(&residualFlow[IDX(((global_id*nodes_in_wi_x)+i - column), (global_id*nodes_in_wi_x)+i, total_nodes)], delta);
		        		atom_sub(&excessFlow[(global_id*nodes_in_wi_x)+i], delta);
		        		atom_add(&excessFlow[((global_id*nodes_in_wi_x)+i - column)], delta);

					if (((global_id*nodes_in_wi_x)+i - column) == s) {atom_sub(netFlowOutS, delta);} 
					else if (((global_id*nodes_in_wi_x)+i - column) == t) {atom_add(netFlowInT, delta);}				
				}
			}
			
			//push to right
			curExcess=excessFlow[(global_id*nodes_in_wi_x)+i];
			if(gidx<column-1){
				curCapacity = residualFlow[IDX((global_id*nodes_in_wi_x)+i, ((global_id*nodes_in_wi_x)+i + 1), total_nodes)];
				if(curCapacity>0 && curExcess>0 && me==right+1){
					int delta = min(curExcess, curCapacity);
				  	atom_sub(&residualFlow[IDX((global_id*nodes_in_wi_x)+i, ((global_id*nodes_in_wi_x)+i + 1), total_nodes)], delta);
		        		atom_add(&residualFlow[IDX(((global_id*nodes_in_wi_x)+i + 1), (global_id*nodes_in_wi_x)+i, total_nodes)], delta);
		        		atom_sub(&excessFlow[(global_id*nodes_in_wi_x)+i], delta);
		        		atom_add(&excessFlow[((global_id*nodes_in_wi_x)+i + 1)], delta);	

					if (((global_id*nodes_in_wi_x)+i + 1) == s) {atom_sub(netFlowOutS, delta);} 
					else if (((global_id*nodes_in_wi_x)+i + 1) == t) {atom_add(netFlowInT, delta);}			
				}
			}

			//push to down
			curExcess=excessFlow[((global_id*nodes_in_wi_x)+i*nodes_in_wi_x)+i];
			if(gidy<row-1){
				curCapacity = residualFlow[IDX((global_id*nodes_in_wi_x)+i, ((global_id*nodes_in_wi_x)+i + column), total_nodes)];
				if(curCapacity>0 && curExcess>0 && me==down+1){
					int delta = min(curExcess, curCapacity);
				  	atom_sub(&residualFlow[IDX((global_id*nodes_in_wi_x)+i, ((global_id*nodes_in_wi_x)+i + column), total_nodes)], delta);
		        		atom_add(&residualFlow[IDX(((global_id*nodes_in_wi_x)+i + column), (global_id*nodes_in_wi_x)+i, total_nodes)], delta);
		        		atom_sub(&excessFlow[(global_id*nodes_in_wi_x)+i], delta);
		        		atom_add(&excessFlow[((global_id*nodes_in_wi_x)+i + column)], delta);	

					if (((global_id*nodes_in_wi_x)+i + column) == s) {atom_sub(netFlowOutS, delta);} 
					else if (((global_id*nodes_in_wi_x)+i + column) == t) {atom_add(netFlowInT, delta);}				
				}
			}

			//push to left
			curExcess=excessFlow[((global_id*nodes_in_wi_x)+i*nodes_in_wi_x)+i];
			if(gidx>0){
				curCapacity = residualFlow[IDX((global_id*nodes_in_wi_x)+i, ((global_id*nodes_in_wi_x)+i - 1), total_nodes)];
				if(curCapacity>0 && curExcess>0 && me==left+1){
					int delta = min(curExcess, curCapacity);
				  	atom_sub(&residualFlow[IDX((global_id*nodes_in_wi_x)+i, ((global_id*nodes_in_wi_x)+i - 1), total_nodes)], delta);
		        		atom_add(&residualFlow[IDX(((global_id*nodes_in_wi_x)+i - 1), (global_id*nodes_in_wi_x)+i, total_nodes)], delta);
		        		atom_sub(&excessFlow[(global_id*nodes_in_wi_x)+i], delta);
		        		atom_add(&excessFlow[((global_id*nodes_in_wi_x)+i - 1)], delta);	

					if (((global_id*nodes_in_wi_x)+i - 1) == s) {atom_sub(netFlowOutS, delta);} 
					else if (((global_id*nodes_in_wi_x)+i - 1) == t) {atom_add(netFlowInT, delta);}				
				}
			}
		}*/
///////////////////////////////////////////////////////////////////////////////////////////////////	
	}
	
}


