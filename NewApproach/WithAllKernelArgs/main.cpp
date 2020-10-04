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
	int *excessFlows = (int *)calloc(g->n, sizeof(int));
	int *residualFlow = (int *)malloc((g->n * g->n) * sizeof(int));
	memcpy(residualFlow, g->capacities, (g->n * g->n) * sizeof(int));

	// initialize preflow
	int flowOutS = 0;
	int flowInT = 0;
	int *heights = (int *)calloc(g->n, sizeof(int));
	heights[s] = g->n;

	for (int v = 0; v < g->n; v++) {
		int cap = g->capacities[IDX(s, v, g->n)];
		if (cap > 0 && (s != v)) {
			residualFlow[IDX(s, v, g->n)] = 0;
			residualFlow[IDX(v, s, g->n)] += cap;
			flowOutS += cap;
			excessFlows[v] = cap;
			if (v == t) {
				flowInT += cap;
			}
		}
	}
  //cl_mem netFlowOutS_buf;
  //netFlowOutS_buf = (cl_mem )calloc(1, sizeof(int));
  //int* OutS=&flowOutS;

  ///////////////////////////////////////////////////////////////////////////
  //allocate and transfer buffers on/to the device
	

  cl::Buffer residualFlowBuffer(myContext, CL_MEM_READ_WRITE, sizeof(cl_int)*(NodesNo*NodesNo));
  cl::Buffer heightsBuffer(myContext, CL_MEM_READ_WRITE, sizeof(cl_int)*(NodesNo));
  cl::Buffer excessFlowsBuffer(myContext, CL_MEM_READ_WRITE, sizeof(cl_int)*(NodesNo));
  cl::Buffer netFlowOutSBuffer(myContext, CL_MEM_READ_WRITE, sizeof(cl_int));
  cl::Buffer flowInTBuffer(myContext, CL_MEM_READ_WRITE, sizeof(cl_int));
 
  err= myQueue.enqueueWriteBuffer(residualFlowBuffer,CL_TRUE,0,sizeof(cl_int)*(NodesNo*NodesNo),residualFlow);
  checkErr(err, "WriteBuffer");
  err= myQueue.enqueueWriteBuffer(heightsBuffer,CL_TRUE,0,sizeof(cl_int)*(NodesNo),heights);
  checkErr(err, "WriteBuffer");
  err= myQueue.enqueueWriteBuffer(excessFlowsBuffer,CL_TRUE,0,sizeof(cl_int)*(NodesNo),excessFlows);
  checkErr(err, "WriteBuffer");
  err= myQueue.enqueueWriteBuffer(netFlowOutSBuffer,CL_TRUE,0,sizeof(cl_int),&flowOutS);
  checkErr(err, "WriteBuffer");
  err= myQueue.enqueueWriteBuffer(flowInTBuffer,CL_TRUE,0,sizeof(cl_int),&flowInT);
  checkErr(err, "WriteBuffer");
  // setup kernel argument list
  err = myKernel.setArg(0, residualFlowBuffer);
	checkErr(err, "Arg 0");
	
  err = myKernel.setArg(1, NodesNo);
	checkErr(err, "Arg 1");
  err = myKernel.setArg(2, heightsBuffer);
	checkErr(err, "Arg 2");
  err = myKernel.setArg(3, excessFlowsBuffer);
	checkErr(err, "Arg 3");
  err = myKernel.setArg(4, netFlowOutSBuffer);
	checkErr(err, "Arg 4");
  err = myKernel.setArg(5, flowInTBuffer);
	checkErr(err, "Arg 5");
  err = myKernel.setArg(6, s);
	checkErr(err, "Arg 6");
   err = myKernel.setArg(7, t);
	checkErr(err, "Arg 7");

  //launch the kernel
  err=myQueue.enqueueNDRangeKernel(myKernel, cl::NullRange, cl::NDRange(NodesNo), cl::NDRange(workSize));
  
  //err=myQueue.enqueueTask(myKernel);
  checkErr(err, "Kernel Execution");

  //transfer result buffer back
  err= myQueue.enqueueReadBuffer(residualFlowBuffer,CL_TRUE,0,sizeof(cl_int)*(NodesNo*NodesNo),residualFlow);
  err=myQueue.finish();
  checkErr(err, "Read Buffer");

  err= myQueue.enqueueReadBuffer(heightsBuffer,CL_TRUE,0,sizeof(cl_int)*(NodesNo),heights);
  err=myQueue.finish();
  checkErr(err, "Read Buffer");

  err= myQueue.enqueueReadBuffer(excessFlowsBuffer,CL_TRUE,0,sizeof(cl_int)*(NodesNo),excessFlows);
  err=myQueue.finish();
  checkErr(err, "Read Buffer");

  err= myQueue.enqueueReadBuffer(netFlowOutSBuffer,CL_TRUE,0,sizeof(cl_int),&flowOutS);
  err=myQueue.finish();
  checkErr(err, "Read Buffer");

  err= myQueue.enqueueReadBuffer(flowInTBuffer,CL_TRUE,0,sizeof(cl_int),&flowInT);
  err=myQueue.finish();
  checkErr(err, "Read Buffer");

  //print results
    for(unsigned j = 0; j < (NodesNo*NodesNo); ++j) {
      
      std::cout<<residualFlow[j]<<"  ";

    }
    std::cout<<"\n";

    for(unsigned j = 0; j < (NodesNo); ++j) {
      
      std::cout<<heights[j]<<"  ";

    }
    std::cout<<"\n";

    for(unsigned j = 0; j < (NodesNo); ++j) {
      
      std::cout<<excessFlows[j]<<"  ";

    }
    std::cout<<"\n";

    std::cout<<flowOutS<<"  ";
    std::cout<<"\n";

    std::cout<<flowInT<<"  ";
    std::cout<<"\n";

  clReleaseKernel(myKernel());
  
  return 0;
}


