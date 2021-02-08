//ACL Kernel
#define IDX(i, j, n) ((i) * (n) + (j))
//#include<stdlib.h>

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

	int res = 0;

	__local int heights_block[4][4] __attribute__((memory,bank_bits(3,2)
,doublepump
,bankwidth(16)
,numreadports(5)
,numwriteports(1)
))
;
	if(lidx==0 && lidy==0){
		//#pragma unroll
		//for (int i = 0; i < 2; i++) {
		
		//heights_block[1+i][1]=height[global_id+i*column];
		//heights_block[1+i][2]=height[global_id+i*column+1];
		//heights_block[1][0]=0;
		heights_block[1][1]=height[global_id];
		heights_block[1][2]=height[global_id];
		//heights_block[1][3]=0;

		//heights_block[2][0]=0;
		heights_block[2][1]=height[global_id+1*column];
		heights_block[2][2]=height[global_id+1*column];
		//heights_block[2][3]=0;
		//}
	
	//rast
	if(groupidx < 1){
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
		heights_block[0][0]=0;
		heights_block[0][3]=0;
	}

	//payin
	if(groupidy < 1){
		#pragma unroll
		for (int i = 0; i < 2; i++) {
			heights_block[3][i]=height[global_id+2*column+i];
		}
		heights_block[3][0]=0;
		heights_block[3][3]=0;
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
///////////////////////////////////////////////////////////////////////////
int me, up, right, down, left;
if(lidx==0 && lidy==0){
	left=heights_block[1][0];
	me=heights_block[1][1];
	right=heights_block[1][2];
	up=heights_block[0][1];
	
	down=heights_block[2][1];
	
}
if(lidx==0 && lidy==1){
	left=heights_block[2][0];
	me=heights_block[2][1];
	right=heights_block[2][2];
	up=heights_block[1][1];

	down=heights_block[3][1];
	
}
if(lidx==1 && lidy==0){
	left=heights_block[1][1];	
	me=heights_block[1][2];
	right=heights_block[1][3];
	up=heights_block[0][2];

	down=heights_block[2][2];
	
}
if(lidx==1 && lidy==1){
	left=heights_block[2][1];
	me=heights_block[2][2];
	right=heights_block[2][3];
	up=heights_block[1][2];

	down=heights_block[3][2];

}
	/*int me=heights_block[node_y][node_x];
	int up=heights_block[up_y][up_x];
	//int right=heights_block[right_y][right_x];
int right=heights_block[node_y][node_x+1];
	int down=heights_block[down_y][down_x];
	//int left=heights_block[left_y][left_x];
int left=heights_block[node_y][node_x-1];*/
///////////////////////////////////////////////////////////////////////////
height[global_id]=left+me+right+up+down;
	/*//push operation
	if (global_id != s && global_id != t){
		int curExcess=excessFlow[global_id];
		int curCapacity=0;

		//push to up
		if(gidy>0){
			curCapacity = residualFlow[IDX(global_id, (global_id - column), total_nodes)];
			if(curCapacity>0 && curExcess>0 && me==up+1){
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
			if(curCapacity>0 && curExcess>0 && me==right+1){
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
			if(curCapacity>0 && curExcess>0 && me==down+1){
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
			if(curCapacity>0 && curExcess>0 && me==left+1){
				int delta = min(curExcess, curCapacity);
		          	atom_sub(&residualFlow[IDX(global_id, (global_id - 1), total_nodes)], delta);
                		atom_add(&residualFlow[IDX((global_id - 1), global_id, total_nodes)], delta);
                		atom_sub(&excessFlow[global_id], delta);
                		atom_add(&excessFlow[(global_id - 1)], delta);	

				if ((global_id - 1) == s) {atom_sub(netFlowOutS, delta);} 
				else if ((global_id - 1) == t) {atom_add(netFlowInT, delta);}				
			}
		}
	}*/
}

/*__kernel void RelabelKernel(__global int * restrict residualFlow, uint column,__global uint * restrict height,
__global int * restrict excessFlow,__global int * restrict netFlowOutS,
__global int * restrict netFlowInT,uint s,uint t,uint row)
{
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

	//hight block that is 1 size bigger than original block at every edge
	__local int heights_block[4][4] 
__attribute__((memory,bank_bits(3,2)
,doublepump
,bankwidth(16)
,numreadports(4)
,numwriteports(2)
))
;


	//input height to local memory

	if(lidx==0 && lidy==0){
		#pragma unroll
		for (int i = 0; i < 2; i++) {
		
		heights_block[1+i][1]=height[global_id+i*column];
		heights_block[1+i][2]=height[global_id+i*column+1];
		
		}
	
	//rast
	if(groupidx < 1){
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
	if(groupidy < 1){
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
	barrier (CLK_LOCAL_MEM_FENCE);
	
	//relable operation
	int curExcess=excessFlow[global_id];
	int upres;
	int downres;
	int rightres;
	int leftres;
	bool curLowestNeighbor = false;
	uint neighborMinHeight = 4294967295;
	uint neighbourh;*/
/////////////////////////////////////////////////////////
/*int node_h=heights_block[node_y][node_x];
int up_h=heights_block[up_y][up_x];
int down_h=heights_block[down_y][down_x];
int right_h=heights_block[right_y][right_x];
int left_h=heights_block[left_y][left_x];*/
/*int heigh_node[5];
#pragma unroll
for(int i = 0; i < 3; i++){
heigh_node[i]=heights_block[left_y][left_x+i];
}
heigh_node[3]=heights_block[up_y][up_x];
heigh_node[4]=heights_block[down_y][down_x];
/////////////////////////////////////////////////////////
	if(global_id != s && global_id != t){
	if(curExcess>0){
		//begin from up
		if(gidy>0){
			upres=residualFlow[IDX(global_id, (global_id - column), total_nodes)];
			//check weight
			if(upres>0){
				neighbourh = heigh_node[3];
				if(heigh_node[1]==neighbourh+1) {goto END;} 
				else{//do compare
		    	            	if (neighbourh < neighborMinHeight) {
	      	               		curLowestNeighbor = true;
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
				neighbourh = heigh_node[4];
				if(heigh_node[1]==neighbourh+1){goto END;} 
				else{//do compare
		    	            	if (neighbourh < neighborMinHeight) {
	      	               		curLowestNeighbor = true;
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
				neighbourh = heigh_node[2];
				if(heigh_node[1]==neighbourh+1){goto END;} 
				else{//do compare
		    	            	if (neighbourh < neighborMinHeight) {
	      	               		curLowestNeighbor = true;
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
				neighbourh = heigh_node[0];
				if(heigh_node[1]==neighbourh+1){goto END;} 
				else{//do compare
		    	            	if (neighbourh < neighborMinHeight) {
	      	               		curLowestNeighbor = true;
        	                	neighborMinHeight = neighbourh;
        	          		}			
				}			
			}
		}
		//relable
		if(curLowestNeighbor == true){
			height[global_id] = neighborMinHeight + 1;
		}
	END:;}
	}
}*/


