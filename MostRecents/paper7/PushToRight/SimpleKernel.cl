//ACL Kernel
#define IDX(i, j, n) ((i) * (n) + (j))
//#include<stdlib.h>

//__attribute__((num_simd_work_items(2)))
__attribute__((reqd_work_group_size(1,2,1)))
__kernel void PushKernel(__global int * restrict residualFlow_right,__global int * restrict residualFlow_left,
__global int * restrict temp,
 uint column,__global uint * restrict height,
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
	//int first_node_x=nodes_in_wi_x*lidx+1;
	//int first_node_y=nodes_in_wi_y*lidy+1;
	int first_node=global_id*nodes_in_wi_x;

	__local int heights_block[2][3] 
//__attribute__((
//memory
//,bank_bits(2)
//,numbanks(1)
//,bankwidth(16)
//,doublepump
//,numreadports(3)
//,numwriteports(2)
//))
;

__local int excess_block[2][2] 
__attribute__((
memory
,bank_bits(0)
,numbanks(2)
,bankwidth(4)
,doublepump
,numreadports(3)
,numwriteports(3)
))
;

__local int residualFlow_right_block[2][2] 
/*__attribute__((
memory
,bank_bits(0)
,numbanks(2)
,bankwidth(4)
,doublepump
,numreadports(3)
,numwriteports(3)
))*/
;

__local int residualFlow_left_block[2][2] 
/*__attribute__((
memory
,bank_bits(0)
,numbanks(2)
,bankwidth(4)
,doublepump
,numreadports(3)
,numwriteports(3)
))*/
;
	#pragma unroll
	for (int i = 0; i < 2; i++) {
		heights_block[lidy][i]=height[first_node+i];
	}
	if(groupidx<2){
		heights_block[lidy][2]=height[first_node+2];
	}
	else{
		heights_block[lidy][2]=0;
	}
	
	#pragma unroll
	for (int i = 0; i < 2; i++) {
		excess_block[lidy][i]=excessFlow[first_node+i];
	}

	#pragma unroll
	for (int i = 0; i < 2; i++) {
		residualFlow_right_block[lidy][i]=residualFlow_right[first_node+i];
	}

	#pragma unroll
	for (int i = 0; i < 2; i++) {
		residualFlow_left_block[lidy][i]=residualFlow_left[first_node+i+1];
	}

	barrier(CLK_LOCAL_MEM_FENCE);	

	//push operation
	#pragma unroll
	for(int i = 0; i < 2; i++){	
		int curCapacity=0;
		curCapacity = residualFlow_right_block[lidy][i];
		int curExcess = excess_block[lidy][i];
		int neighborExcess = excess_block[lidy][i+1];
		if ((first_node)+i != s && (first_node)+i != t && curCapacity>0 && curExcess>0 && heights_block[lidy][i]==heights_block[lidy][i+1]+1){
			

			//if(curCapacity>0 && curExcess>0 && heights_block[lidy][i]==heights_block[lidy][i+1]+1){
				int delta = min(curExcess, curCapacity);
				//atom_sub(&residualFlow_right[(global_id*nodes_in_wi_x)+i], delta);
				residualFlow_right_block[lidy][i]=residualFlow_right_block[lidy][i]-delta;		
				excess_block[lidy][i]=curExcess-delta;
				//atom_add(&residualFlow_left[(global_id*nodes_in_wi_x)+i+1], delta);
				//residualFlow_left[(global_id*nodes_in_wi_x)+i+1]=residualFlow_left[(global_id*nodes_in_wi_x)+i+1]+delta;
				residualFlow_left_block[lidy][i]=residualFlow_left_block[lidy][i]+delta;
		        	if (i != nodes_in_wi_x-1) {
					excess_block[lidy][i+1]=neighborExcess+delta;
					
				}
				else{
					//temp[gidy + 4(gidx)]=residualFlow_left[(global_id*nodes_in_wi_x)+i+1] + delta;
					temp[gidy + 4*(gidx)]=neighborExcess + delta;
				}	

				if (((first_node)+i + 1) == s) {*netFlowOutS=*netFlowOutS-delta;} 
				else if (((first_node)+i + 1) == t) {*netFlowOutS=*netFlowOutS+delta;}				
			//}	
		
		}
	//excessFlow[global_id*nodes_in_wi_x+i]=excess_block[lidy][i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int i = 0; i < 2; i++){excessFlow[first_node+i]=excess_block[lidy][i];}
	for(int i = 0; i < 2; i++){residualFlow_right[(first_node)+i]=residualFlow_right_block[lidy][i];}
	for(int i = 0; i < 2; i++){residualFlow_left[(first_node)+i+1]=residualFlow_left_block[lidy][i];}
}
