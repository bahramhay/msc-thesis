#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iomanip>
//#include "CL/opencl.h"
//#include "AOCLUtils/aocl_utils.h"
//#include "CL/cl.hpp"
#include "graphs.h"
#include "CL/timer.hpp"
#include "CL/cl.hpp"
//#include "utility.h"
#define IDX(i, j, n) ((i) * (n) + (j))

//using namespace aocl_utils;
using namespace std;

//static const cl_uint vectorSize = 8;
//static const cl_uint workSize = 2;

static const cl_uint Nodescolumn = 24;
static const cl_uint NodesRow = 50;
static const cl_uint TotalNodes = NodesRow*Nodescolumn;
uint s = 0;
uint t = (TotalNodes) - 1;

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
	//in.open("/home/bahram/Desktop/newFromScratch/a.txt");
  in.open("a.txt");
	/*int i;
	int j;*/
	//Graph *g = (Graph *)malloc(sizeof(Graph));

	g->N = Nodescolumn;
  g->M = NodesRow;
	g->capacities = (int *)calloc((TotalNodes*TotalNodes), sizeof(int));
	while (!in.eof())
	{
		for (int i = 0; i < TotalNodes; i++) {
			for (int j = 0; j < TotalNodes; j++) {
				in >> g->capacities[IDX(i, j, TotalNodes)];
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
	//std::ifstream aocx_stream("compiled/SimpleKernel.aocx", std::ios::in|std::ios::binary);
  std::ifstream aocx_stream("SimpleKernel.aocx", std::ios::in|std::ios::binary);
	checkErr(aocx_stream.is_open() ? CL_SUCCESS:-1, "SimpleKernel.aocx");
	std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));
	cl::Program::Binaries mybinaries (1, std::make_pair(prog.c_str(), prog.length()));
  //create then build program
  cl::Program myProgram(myContext,dlist,mybinaries);
  err= myProgram.build(dlist);
  //create kernels from program 
  cl::Kernel myKernel (myProgram,"PushKernel",&err);
  cl::Kernel myKernel1 (myProgram,"RelabelKernel",&err);
  //create a command queue to the device
  cl::CommandQueue myQueue(myContext,dlist[0]);
  ///////////////////////////////////////////////////////////////Input graph
  Graph *g = (Graph *)malloc(sizeof(Graph));
	InputGraph(g);
  //int s = 0;
	//int t = g->n - 1;
	int *excessFlows = (int *)calloc(TotalNodes, sizeof(int));
	int *residualFlow = (int *)malloc((TotalNodes*TotalNodes) * sizeof(int));
	memcpy(residualFlow, g->capacities, (TotalNodes*TotalNodes) * sizeof(int));
  
	// initialize preflow
	int flowOutS = 0;
	int flowInT = 0;
	uint *heights = (uint *)calloc(TotalNodes, sizeof(uint));
	heights[s] = TotalNodes;

	for (int v = 0; v < TotalNodes; v++) {
		int cap = g->capacities[IDX(s, v, TotalNodes)];
		if (cap > 0 && (s != v)) {
			residualFlow[IDX(s, v, TotalNodes)] = 0;
			residualFlow[IDX(v, s, TotalNodes)] += cap;
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
  //////////for test
  /*ifstream in;
	in.open("/home/bahram/Desktop/newFromScratch/ForTest/height.txt");
  
	while (!in.eof())
	{
		for (int i = 0; i < NodesRow; i++) {
			for (int j = 0; j < Nodescolumn; j++) {
				in >> heights[IDX(i, j, Nodescolumn)];
			}
		}
	}
	in.close();
  /////////////////////////
/*std::cout<<"heights"<<"\n";
    for(unsigned i = 0; i < (NodesRow); ++i) {
      for(unsigned j = 0; j < (Nodescolumn); ++j){  
        std::cout<<setw(5)<<heights[IDX(i, j, Nodescolumn)];
      }
      std::cout<<"\n";
    }
    std::cout<<"\n";*/
  //////////////////////////

  /*in.open("/home/bahram/Desktop/newFromScratch/ForTest/excess.txt");
  
	while (!in.eof())
	{
		for (int i = 0; i < NodesRow; i++) {
			for (int j = 0; j < Nodescolumn; j++) {
				in >> excessFlows[IDX(i, j, Nodescolumn)];
			}
		}
	}
	in.close();

  in.open("/home/bahram/Desktop/newFromScratch/ForTest/residual.txt");
  
	while (!in.eof())
	{
		for (int i = 0; i < TotalNodes; i++) {
			for (int j = 0; j < TotalNodes; j++) {
				in >> residualFlow[IDX(i, j, TotalNodes)];
			}
		}
	}
	in.close();
  //////////end of test
  //allocate and transfer buffers on/to the device
	//cl_mem netFlowOutSBuffer = (cl_mem)flowOutS;
*/
  cl::Buffer residualFlowBuffer(myContext, CL_MEM_READ_WRITE, sizeof(cl_int)*(TotalNodes*TotalNodes));
  cl::Buffer heightsBuffer(myContext, CL_MEM_READ_WRITE, sizeof(cl_uint)*(TotalNodes));
  cl::Buffer excessFlowsBuffer(myContext, CL_MEM_READ_WRITE, sizeof(cl_int)*(TotalNodes));
  cl::Buffer netFlowOutSBuffer(myContext, CL_MEM_READ_WRITE, sizeof(cl_int));
  cl::Buffer flowInTBuffer(myContext, CL_MEM_READ_WRITE, sizeof(cl_int));
 
  err= myQueue.enqueueWriteBuffer(residualFlowBuffer,CL_TRUE,0,sizeof(cl_int)*(TotalNodes*TotalNodes),residualFlow);
  checkErr(err, "WriteBuffer");
  err= myQueue.enqueueWriteBuffer(heightsBuffer,CL_TRUE,0,sizeof(cl_uint)*(TotalNodes),heights);
  checkErr(err, "WriteBuffer");
  err= myQueue.enqueueWriteBuffer(excessFlowsBuffer,CL_TRUE,0,sizeof(cl_int)*(TotalNodes),excessFlows);
  checkErr(err, "WriteBuffer");
  err= myQueue.enqueueWriteBuffer(netFlowOutSBuffer,CL_TRUE,0,sizeof(cl_int),&flowOutS);
  checkErr(err, "WriteBuffer");
  err= myQueue.enqueueWriteBuffer(flowInTBuffer,CL_TRUE,0,sizeof(cl_int),&flowInT);
  checkErr(err, "WriteBuffer");
  // setup kernel argument list
  err = myKernel.setArg(0, residualFlowBuffer);
	checkErr(err, "Arg 0");
	err = myKernel.setArg(1, Nodescolumn);
	checkErr(err, "Arg 1");
  err = myKernel.setArg(2, heightsBuffer);
	checkErr(err, "Arg 2");
  err = myKernel.setArg(3, excessFlowsBuffer);
	checkErr(err, "Arg 3");
  err = myKernel.setArg(4, netFlowOutSBuffer);
  //err = myKernel.setArg(4, &flowOutS);
	checkErr(err, "Arg 4");
  err = myKernel.setArg(5, flowInTBuffer);
	checkErr(err, "Arg 5");
  err = myKernel.setArg(6, s);
	checkErr(err, "Arg 6");
  err = myKernel.setArg(7, t);
	checkErr(err, "Arg 7");
  err = myKernel.setArg(8, NodesRow);
  //err=clSetKernelArg(myKernel, 0, sizeof(cl_mem), &NodesRow);
	checkErr(err, "Arg 8");

  err = myKernel1.setArg(0, residualFlowBuffer);
	checkErr(err, "Arg 0");
	err = myKernel1.setArg(1, Nodescolumn);
	checkErr(err, "Arg 1");
  err = myKernel1.setArg(2, heightsBuffer);
	checkErr(err, "Arg 2");
  err = myKernel1.setArg(3, excessFlowsBuffer);
	checkErr(err, "Arg 3");
  err = myKernel1.setArg(4, netFlowOutSBuffer);
	checkErr(err, "Arg 4");
  err = myKernel1.setArg(5, flowInTBuffer);
	checkErr(err, "Arg 5");
  err = myKernel1.setArg(6, s);
	checkErr(err, "Arg 6");
  err = myKernel1.setArg(7, t);
	checkErr(err, "Arg 7");
  err = myKernel1.setArg(8, NodesRow);
	checkErr(err, "Arg 8");

  //launch the kernel
  const double start_time = getCurrentTimestamp();
  int beforeOutS=-1;
  int beforeInT=-1;
  int count=0;
  while(flowOutS!=flowInT){
  //for(int i=0 ;i<20000 ;i++){
    err=myQueue.enqueueNDRangeKernel(myKernel, cl::NullRange, cl::NDRange(Nodescolumn,NodesRow), cl::NDRange(2,2));
    err=myQueue.finish(); 
    checkErr(err, "enqueue"); 
    err=myQueue.enqueueNDRangeKernel(myKernel1, cl::NullRange, cl::NDRange(Nodescolumn,NodesRow), cl::NDRange(2,2));
    err=myQueue.finish();
    checkErr(err, "enqueue");

    err= myQueue.enqueueReadBuffer(netFlowOutSBuffer,CL_TRUE,0,sizeof(cl_int),&flowOutS);
    err=myQueue.finish();
    checkErr(err, "Read Buffer");

    err= myQueue.enqueueReadBuffer(flowInTBuffer,CL_TRUE,0,sizeof(cl_int),&flowInT);
    err=myQueue.finish();
    checkErr(err, "Read Buffer");

    count++;
    
    if(beforeOutS!=flowOutS || beforeInT!=flowInT){
    std::cout<<flowOutS<<"  "<<flowInT<<"  "<< count;
    std::cout<<"\n";
    }
    //if(count%100==0){std::cout<<"counter reached: "<<count<<std::endl;}
    beforeOutS=flowOutS;
    beforeInT=flowInT;
    //if(flowOutS == flowInT){break;}
  }
  std::cout<<"count: "<<count<<std::endl;
  ////////////////////////////////////////////////////////////////
    /*err=myQueue.enqueueNDRangeKernel(myKernel, cl::NullRange, cl::NDRange(Nodescolumn,NodesRow), cl::NDRange(1,1));
    err=myQueue.finish(); 
    checkErr(err, "enqueue");

    err= myQueue.enqueueReadBuffer(netFlowOutSBuffer,CL_TRUE,0,sizeof(cl_int),&flowOutS);
    err=myQueue.finish();
    checkErr(err, "Read Buffer");

    err= myQueue.enqueueReadBuffer(flowInTBuffer,CL_TRUE,0,sizeof(cl_int),&flowInT);
    err=myQueue.finish();
    checkErr(err, "Read Buffer");

    std::cout<<flowOutS<<"  "<<flowInT<<"  ";
    std::cout<<"\n";*/
  /////////////////////////////////////////////////////////////////
  const double end_time = getCurrentTimestamp();
  //err=myQueue.enqueueTask(myKernel);
  //checkErr(err, "Kernel Execution");

  //transfer result buffer back
  err= myQueue.enqueueReadBuffer(residualFlowBuffer,CL_TRUE,0,sizeof(cl_int)*(TotalNodes*TotalNodes),residualFlow);
  err=myQueue.finish();
  checkErr(err, "Read Buffer");

  err= myQueue.enqueueReadBuffer(heightsBuffer,CL_TRUE,0,sizeof(cl_uint)*(TotalNodes),heights);
  err=myQueue.finish();
  checkErr(err, "Read Buffer");

  err= myQueue.enqueueReadBuffer(excessFlowsBuffer,CL_TRUE,0,sizeof(cl_int)*(TotalNodes),excessFlows);
  err=myQueue.finish();
  checkErr(err, "Read Buffer");

  /*err= myQueue.enqueueReadBuffer(netFlowOutSBuffer,CL_TRUE,0,sizeof(cl_int),&flowOutS);
  err=myQueue.finish();
  checkErr(err, "Read Buffer");

  err= myQueue.enqueueReadBuffer(flowInTBuffer,CL_TRUE,0,sizeof(cl_int),&flowInT);
  err=myQueue.finish();
  checkErr(err, "Read Buffer");*/

  /*std::ofstream of{ "/home/bahram/Desktop/newFromScratch/ForTest/800+/excess20000.txt" };
	for (int i = 0; i < NodesRow; i++) {
		for (int j = 0; j < Nodescolumn; j++) {
			of << setw(4)<< excessFlows[IDX(i, j, Nodescolumn)];
		}
		of<<std::endl;
	}
	of.close();

  std::ofstream of2{ "/home/bahram/Desktop/newFromScratch/ForTest/800+/height20000.txt" };
	for (int i = 0; i < NodesRow; i++) {
		for (int j = 0; j < Nodescolumn; j++) {
			of2 << setw(4)<< heights[IDX(i, j, Nodescolumn)];
		}
		of2<<std::endl;
	}
	of2.close();

  //print results
  /*std::cout<<"Residuals"<<"\n";
  std::cout<< setw(2)<<"";
  for(unsigned j = 0; j < (TotalNodes); ++j){
        std::cout<< setw(5)<<j;}
  std::cout<<"\n"<<"\n";
    for(unsigned i = 0; i < (TotalNodes); ++i) {
      std::cout<<setw(2)<<i;
      for(unsigned j = 0; j < (TotalNodes); ++j){
        std::cout<< setw(5)<<residualFlow[IDX(i, j, TotalNodes)];
        
      }  
      std::cout<<"\n";

    }
    std::cout<<"\n";

  std::cout<<"heights"<<"\n";
    for(unsigned i = 0; i < (NodesRow); ++i) {
      for(unsigned j = 0; j < (Nodescolumn); ++j){  
        std::cout<<setw(5)<<heights[IDX(i, j, Nodescolumn)];
      }
      std::cout<<"\n";
    }
    std::cout<<"\n";

  std::cout<<"excess"<<"\n";
    for(unsigned i = 0; i < (NodesRow); ++i) {
      for(unsigned j = 0; j < (Nodescolumn); ++j){
        std::cout<<setw(5)<<excessFlows[IDX(i, j, Nodescolumn)];
      }
      std::cout<<"\n";
    }
    std::cout<<"\n";*/
    

    std::cout<<flowOutS<<"  ";
    std::cout<<"\n";

    std::cout<<flowInT<<"  ";
    std::cout<<"\n";

    std::cout<<"Time:"<<(end_time - start_time) * 1e3<<" ms"<<"  "<<endl;

  clReleaseKernel(myKernel());
  clReleaseKernel(myKernel1());

  return 0;
}
