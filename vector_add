// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

 // ACL kernel for adding two input vectors
#define IDX(i, j, n) ((i) * (n) + (j))
__kernel void vector_add(__global int * residualFlow, 
                         //__global const int *b, 
                         __global int * height,
                         __global int * excessFlow,
                         __global int * netFlowOutS,
			 __global int * netFlowInT,
			 __global int * s,
			 __global int * t,
                         __global int * n
                         )
{
    int gid = get_global_id(0);
    //int x=*n-1;
	 //while (*netFlowOutS != *netFlowInT) {
	 for(int x=0;x<7;x++ )  { //if(netFlowOutS==netFlowInT){breake;}
         barrier(CLK_GLOBAL_MEM_FENCE);

         printf("gid: %d, excess:%d  \n",gid, excessFlow[gid]);
         if (gid != *s && gid != *t && excessFlow[gid] > 0) {
		//if (gid ==1) {printf("excess:%d  \n",excessFlow[gid]);}
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
            }//if (gid ==1) {printf("entekhab:%d  \n",curLowestNeighbor);}

			
            if (height[gid] > neighborMinHeight) {
		
                int delta = min(curExcess, residualFlow[IDX(gid, curLowestNeighbor, *n)]);
				//printf("%d  \n",delta);
                    //printf("before gid: %d, excess:%d  \n",gid, excessFlow[gid]); 
                atomic_sub(&residualFlow[IDX(gid, curLowestNeighbor, *n)], delta);
                atomic_add(&residualFlow[IDX(curLowestNeighbor, gid, *n)], delta);
                atomic_sub(&excessFlow[gid], delta);
                atomic_add(&excessFlow[curLowestNeighbor], delta);
                    //printf("after gid: %d, excess:%d  \n",gid, excessFlow[gid]);
                    //if (gid ==2) {printf("hala:%d  \n",excessFlow[1]);
                    //printf("to who:%d  \n",curLowestNeighbor);}
                if (curLowestNeighbor == *s) {
				  
                    atomic_sub(netFlowOutS, delta);

                } else if (curLowestNeighbor == *t) {
				
                    atomic_add(netFlowInT, delta);
                }
            } else {
                height[gid] = neighborMinHeight + 1;
                   }
			//printf("%d  \n",height[gid]);
            }//if 
	
    }//while 
}


