#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "CL/opencl.h"
//#include "AOCLUtils/aocl_utils.h"
//#include "CL/cl.hpp"
#include "graphs.h"

#include "CL/cl.hpp"
//#include "utility.h"
#define IDX(i, j, n) ((i) * (n) + (j))

//using namespace aocl_utils;
using namespace std;

//static const cl_uint vectorSize = 8;
static const cl_uint workSize = 2;
static const cl_uint NodesNo = 4;

void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name
                 << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
/////////////////////////////////////////////////////////////////////////////////////
void InputGraph(Graph *g) {
	ifstream in;
	in.open("/home/bahram/Desktop/newFromScratch/a.txt");

	/*int i;
	int j;*/
	//Graph *g = (Graph *)malloc(sizeof(Graph));

	g->n = NodesNo;
	g->capacities = (int *)calloc((g->n * g->n), sizeof(int));
	while (!in.eof())
	{
		for (int i = 0; i < NodesNo; i++) {
			for (int j = 0; j < NodesNo; j++) {
				in >> g->capacities[IDX(i, j, g->n)];
			}
		}
	}
	in.close();
	/*for (int i = 0; i < NodesNo; i++) {
		for (int j = 0; j < NodesNo; j++) {
			cout << g->capacities[IDX(i, j, g->n)] << " ";
		}
	}*/
}
//////////////////////////////////////////////////////////////////
int main()
{
  cl_int err;
  //platform
  vector<cl::Platform> plist;
  err=cl::Platform::get(&plist);
  if (err != CL_SUCCESS) {
        cout << "error platform" << endl;
    }
  //devices
  vector<cl::Device> dlist;
  err=plist[0].getDevices(CL_DEVICE_TYPE_ALL,&dlist);
  //context
  cl::Context myContext(dlist, NULL, NULL, NULL, &err);
  
  
  //Read in binaries from file
	std::ifstream aocx_stream("compiled/SimpleKernel.aocx", std::ios::in|std::ios::binary);
	checkErr(aocx_stream.is_open() ? CL_SUCCESS:-1, "SimpleKernel.aocx");
	std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));
	cl::Program::Binaries mybinaries (1, std::make_pair(prog.c_str(), prog.length()));
  //create then build program
  cl::Program myProgram(myContext,dlist,mybinaries);
  err= myProgram.build(dlist);
  //create kernels from program 
  cl::Kernel myKernel (myProgram,"SimpleKernel",&err);
  
  //create a command queue to the device
  cl::CommandQueue myQueue(myContext,dlist[0]);
  ///////////////////////////////////////////////////////////////Input graph
  Graph *g = (Graph *)malloc(sizeof(Graph));
	InputGraph(g);
  int s = 0;
	int t = g->n - 1;
	int *tempExcessFlows = (int *)calloc(g->n, sizeof(int));
	int *residualFlow = (int *)malloc((g->n * g->n) * sizeof(int));
	memcpy(residualFlow, g->capacities, (g->n * g->n) * sizeof(int));

	// initialize preflow
	int flowOutS = 0;
	int flowInT = 0;
	int *tempHeights = (int *)calloc(g->n, sizeof(int));
	tempHeights[s] = g->n;

	for (int v = 0; v < g->n; v++) {
		int cap = g->capacities[IDX(s, v, g->n)];
		if (cap > 0 && (s != v)) {
			residualFlow[IDX(s, v, g->n)] = 0;
			residualFlow[IDX(v, s, g->n)] += cap;
			flowOutS += cap;
			tempExcessFlows[v] = cap;
			if (v == t) {
				flowInT += cap;
			}
		}
	}

  ///////////////////////////////////////////////////////////////////////////
  //allocate and transfer buffers on/to the device
	cl_int Z[NodesNo*NodesNo];

  cl::Buffer Buffer_In(myContext, CL_MEM_READ_WRITE, sizeof(cl_int)*(NodesNo*NodesNo));
  
  cl::Buffer Buffer_Out(myContext, CL_MEM_READ_ONLY, sizeof(cl_int)*(NodesNo*NodesNo));
  

  err= myQueue.enqueueWriteBuffer(Buffer_In,CL_FALSE,0,sizeof(cl_int)*(NodesNo*NodesNo),residualFlow);
  checkErr(err, "WriteBuffer");
  // setup kernel argument list
  err = myKernel.setArg(0, Buffer_In);
	checkErr(err, "Arg 0");
	err = myKernel.setArg(1, Buffer_Out);
	checkErr(err, "Arg 1");
  err = myKernel.setArg(2, NodesNo);
	checkErr(err, "Arg 2");

  //launch the kernel
  err=myQueue.enqueueNDRangeKernel(myKernel, cl::NullRange, cl::NDRange(NodesNo), cl::NDRange(workSize));
  
  //err=myQueue.enqueueTask(myKernel);
  checkErr(err, "Kernel Execution");

  //transfer result buffer back
  err= myQueue.enqueueReadBuffer(Buffer_Out,CL_TRUE,0,sizeof(cl_int)*(NodesNo*NodesNo),Z);
  err=myQueue.finish();
  checkErr(err, "Read Buffer");

  
  //print results
  for(unsigned j = 0; j < (NodesNo*NodesNo); ++j) {
      
      std::cout<<Z[j]<<"  ";

    }
    std::cout<<"\n";

  /*for(unsigned j = 0; j < (vectorSize); ++j) {
      
      std::cout<<Z2[j]<<"  ";

    }
    std::cout<<"\n";*/

  clReleaseKernel(myKernel());
  
  return 0;
}


