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

//using namespace aocl_utils;
using namespace std;

static const cl_uint vectorSize = 8;
static const cl_uint workSize = 2;

void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name
                 << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

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
  cl::Kernel myKernel2 (myProgram,"MulKernel",&err);
  //create a command queue to the device
  cl::CommandQueue myQueue(myContext,dlist[0]);
  //allocate and transfer buffers on/to the device
  cl_float X[vectorSize];  
	cl_float Y[vectorSize]; 
	cl_float Z[vectorSize];
  cl_float Z2[vectorSize];
	
  for(unsigned j = 0; j < (vectorSize); ++j) {
      X[j] = (cl_float)j;
      Y[j] = (cl_float)(j * 2);
    }

  cl::Buffer Buffer_In(myContext, CL_MEM_READ_WRITE, sizeof(cl_float)*(vectorSize));
  cl::Buffer Buffer_In2(myContext, CL_MEM_READ_WRITE, sizeof(cl_float)*(vectorSize));
  cl::Buffer Buffer_Out(myContext, CL_MEM_READ_ONLY, sizeof(cl_float)*(vectorSize));
  cl::Buffer Buffer_Out2(myContext, CL_MEM_READ_ONLY, sizeof(cl_float)*(vectorSize));

  err= myQueue.enqueueWriteBuffer(Buffer_In,CL_FALSE,0,sizeof(cl_float)*(vectorSize),X);
  checkErr(err, "WriteBuffer");
  err= myQueue.enqueueWriteBuffer(Buffer_In2,CL_FALSE,0,sizeof(cl_float)*(vectorSize),Y);
  checkErr(err, "WriteBuffer");
  // setup kernel argument list
  err = myKernel.setArg(0, Buffer_In);
	checkErr(err, "Arg 0");
	err = myKernel.setArg(1, Buffer_In2);
	checkErr(err, "Arg 1");
	err = myKernel.setArg(2, Buffer_Out);
	checkErr(err, "Arg 2");

  err = myKernel2.setArg(0, Buffer_In);
	checkErr(err, "Arg 0");
	err = myKernel2.setArg(1, Buffer_In2);
	checkErr(err, "Arg 1");
	err = myKernel2.setArg(2, Buffer_Out2);
	checkErr(err, "Arg 2");
	//err = myKernel.setArg(3, (vectorSize));
	//checkErr(err, "Arg 3");
  
  //launch the kernel
  err=myQueue.enqueueNDRangeKernel(myKernel, cl::NullRange, cl::NDRange(vectorSize), cl::NDRange(workSize));
  err=myQueue.enqueueNDRangeKernel(myKernel2, cl::NullRange, cl::NDRange(vectorSize), cl::NDRange(workSize));
  //err=myQueue.enqueueTask(myKernel);
  checkErr(err, "Kernel Execution");

  //transfer result buffer back
  err= myQueue.enqueueReadBuffer(Buffer_Out,CL_TRUE,0,sizeof(cl_float)*(vectorSize),Z);
  err=myQueue.finish();
  checkErr(err, "Read Buffer");

  err= myQueue.enqueueReadBuffer(Buffer_Out2,CL_TRUE,0,sizeof(cl_float)*(vectorSize),Z2);
  err=myQueue.finish();
  checkErr(err, "Read Buffer");

  //print results
  for(unsigned j = 0; j < (vectorSize); ++j) {
      
      std::cout<<Z[j]<<"  ";

    }
    std::cout<<"\n";

  for(unsigned j = 0; j < (vectorSize); ++j) {
      
      std::cout<<Z2[j]<<"  ";

    }
    std::cout<<"\n";

  clReleaseKernel(myKernel());
  clReleaseKernel(myKernel2());
  return 0;
}


