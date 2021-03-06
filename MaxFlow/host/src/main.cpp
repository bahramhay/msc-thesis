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

///////////////////////////////////////////////////////////////////////////////////
// This host program executes a vector addition kernel to perform:
//  C = A + B
// where A, B and C are vectors with N elements.
//
// This host program supports partitioning the problem across multiple OpenCL
// devices if available. If there are M available devices, the problem is
// divided so that each device operates on N/M points. The host program
// assumes that all devices are of the same type (that is, the same binary can
// be used), but the code can be generalized to support different device typesnum_devices
// easily.
//
// Verification is performed against the same computation on the host CPU.
///////////////////////////////////////////////////////////////////////////////////
#define IDX(i, j, n) ((i) * (n) + (j))
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "graphs.h"

using namespace aocl_utils;
using namespace std;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernel; // num_devices elements
#if USE_SVM_API == 0
scoped_array<cl_mem> residualFlow_buf; // num_devices elements
scoped_array<cl_mem> height_buf;
scoped_array<cl_mem> excessFlow_buf;
cl_mem netFlowOutS_buf;
cl_mem netFlowInT_buf;
cl_mem s_buf;
cl_mem t_buf;
cl_mem n_buf;
//scoped_array<cl_mem> input_b_buf; // num_devices elements
//scoped_array<cl_mem> output_buf; // num_devices elements
#endif /* USE_SVM_API == 0 */

// Problem data.
//unsigned N = 15; // problem size
int NodesNo = 4;
int OutS=0;
int InT=0;
int S=0;
int T=0;
//int PowerTwo = NodesNo * NodesNo;
void InputGraph(Graph *g) {
	ifstream in;
	in.open("/home/bahram/intelFPGA/18.1/hld/board/terasic/de10_nano/test/vector_add/host/src/a.txt");

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
#if USE_SVM_API == 0
scoped_array<scoped_aligned_ptr<int> > residualFlow,height,excessFlow; // num_devices elements
//int n;
//scoped_array<scoped_aligned_ptr<int> > output; // num_devices elements
//int* n=&NodesNo;
#else
scoped_array<scoped_SVM_aligned_ptr<int> > input_a, input_b; // num_devices elements
scoped_array<scoped_SVM_aligned_ptr<int> > output; // num_devices elements
#endif /* USE_SVM_API == 0 */
//scoped_array<scoped_array<int> > ref_output; // num_devices elements
scoped_array<unsigned> n_per_device; // num_devices elements
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
// Function prototypes
//float rand_float();
bool init_opencl();
void init_problem();
void run();
void cleanup();

// Entry point.
int main(int argc, char **argv) {
  Options options(argc, argv);

  // Optional argument to specify the problem size.
  if(options.has("n")) {
    NodesNo = options.get<unsigned>("n");
  }

  // Initialize OpenCL.
  if(!init_opencl()) {
    return -1;
  }

  // Initialize the problem data.
  // Requires the number of devices to be known.
  init_problem();

  // Run the kernel.
  run();

  // Free the resources allocated
  cleanup();

  return 0;
}

/////// HELPER FUNCTIONS ///////

// Randomly generate a floating-point number between -10 and 10.
/*float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}*/

// Initializes the OpenCL objects.
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device[i]).c_str());
  }

  // Create the context.
  context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("vector_add", device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create per-device objects.
  queue.reset(num_devices);
  kernel.reset(num_devices);
  n_per_device.reset(num_devices);
#if USE_SVM_API == 0
  residualFlow_buf.reset(num_devices);
  height_buf.reset(num_devices);
  excessFlow_buf.reset(num_devices);
  netFlowOutS_buf = (cl_mem )calloc(1, sizeof(int));
  netFlowInT_buf = (cl_mem )calloc(1, sizeof(int));
  s_buf = (cl_mem )calloc(1, sizeof(int));
  t_buf = (cl_mem )calloc(1, sizeof(int));
  n_buf = (cl_mem )calloc(1, sizeof(int));
  //input_b_buf.reset(num_devices);
  //output_buf.reset(num_devices);
#endif /* USE_SVM_API == 0 */

  for(unsigned i = 0; i < num_devices; ++i) {
    // Command queue.
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Kernel.
    const char *kernel_name = "vector_add";
    kernel[i] = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Determine the number of elements processed by this device.
    n_per_device[i] = NodesNo / num_devices; // number of elements handled by this device

    // Spread out the remainder of the elements over the first
    // N % num_devices.
    if(i < (NodesNo % num_devices)) {
      n_per_device[i]++;
    }

#if USE_SVM_API == 0
    // Input buffers.
    residualFlow_buf[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, 
        n_per_device[i]*n_per_device[i] * sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    height_buf[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, 
        n_per_device[i] * sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    excessFlow_buf[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, 
        n_per_device[i] * sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for input C");

    netFlowOutS_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
         sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for input C");

    netFlowInT_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
         sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for input C");

    s_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
         sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for input C");

    t_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
         sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for input C");

    n_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for input D");

    /*input_b_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        n_per_device[i]*n_per_device[i] * sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for input B");*/

    // Output buffer.
    /*output_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
        n_per_device[i]* sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for output");*/
#else
    cl_device_svm_capabilities caps = 0;

    status = clGetDeviceInfo(
      device[i],
      CL_DEVICE_SVM_CAPABILITIES,
      sizeof(cl_device_svm_capabilities),
      &caps,
      0
    );
    checkError(status, "Failed to get device info");

    if (!(caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
      printf("The host was compiled with USE_SVM_API, however the device currently being targeted does not support SVM.\n");
      // Free the resources allocated
      cleanup();
      return false;
    }
#endif /* USE_SVM_API == 0 */
  }

  return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }

  residualFlow.reset(num_devices);
  height.reset(num_devices);
  excessFlow.reset(num_devices);
  //input_b.reset(num_devices);
  //output.reset(num_devices);
  //ref_output.reset(num_devices);

  Graph *g = (Graph *)malloc(sizeof(Graph));
	InputGraph(g);
  int s = 0;
	int t = g->n - 1;
	int *tempExcessFlows = (int *)calloc(g->n, sizeof(int));
	int *finalFlow = (int *)malloc((g->n * g->n) * sizeof(int));
	memcpy(finalFlow, g->capacities, (g->n * g->n) * sizeof(int));

	// initialize preflow
	int flowOutS = 0;
	int flowInT = 0;
	int *tempHeights = (int *)calloc(g->n, sizeof(int));
	tempHeights[s] = g->n;

	for (int v = 0; v < g->n; v++) {
		int cap = g->capacities[IDX(s, v, g->n)];
		if (cap > 0 && (s != v)) {
			finalFlow[IDX(s, v, g->n)] = 0;
			finalFlow[IDX(v, s, g->n)] += cap;
			flowOutS += cap;
			tempExcessFlows[v] = cap;
			if (v == t) {
				flowInT += cap;
			}
		}
	}
  OutS=flowOutS;
  InT=flowInT;
  S=s;
  T=t;
  // Generate input vectors A and B and the reference output consisting
  // of a total of N elements.
  // We create separate arrays for each device so that each device has an
  // aligned buffer.
  
  for(unsigned i = 0; i < num_devices; ++i) {
#if USE_SVM_API == 0
    residualFlow[i].reset(n_per_device[i]*n_per_device[i]);
    height[i].reset(n_per_device[i]);
    excessFlow[i].reset(n_per_device[i]);
    //input_b[i].reset(n_per_device[i]*n_per_device[i]);
    //output[i].reset(n_per_device[i]);
    //ref_output[i].reset(n_per_device[i]*n_per_device[i]);

    for(unsigned j = 0; j < n_per_device[i]*n_per_device[i]; ++j) {
      residualFlow[i][j] = finalFlow[j];
      //height[i][j] = tempHeights[j];
      //input_b[i][j] = 1;
      //ref_output[i][j] = input_a[i][j] + 10;
    }
    for(unsigned j = 0; j < n_per_device[i] ; ++j) {
      height[i][j] = tempHeights[j];
    }
    for(unsigned j = 0; j < n_per_device[i] ; ++j) {
      excessFlow[i][j] = tempExcessFlows[j];
    }
#else
    input_a[i].reset(context, n_per_device[i]);
    input_b[i].reset(context, n_per_device[i]);
    output[i].reset(context, n_per_device[i]);
    ref_output[i].reset(n_per_device[i]);

    cl_int status;

    status = clEnqueueSVMMap(queue[i], CL_TRUE, CL_MAP_WRITE,
        (void *)input_a[i], n_per_device[i] * sizeof(int), 0, NULL, NULL);
    checkError(status, "Failed to map input A");
    status = clEnqueueSVMMap(queue[i], CL_TRUE, CL_MAP_WRITE,
        (void *)input_b[i], n_per_device[i] * sizeof(int), 0, NULL, NULL);
    checkError(status, "Failed to map input B");

    for(unsigned j = 0; j < n_per_device[i]; ++j) {
      input_a[i][j] = (int)i;
      input_b[i][j] = (int)(i * 2);
      ref_output[i][j] = input_a[i][j] + input_b[i][j];
    }

    status = clEnqueueSVMUnmap(queue[i], (void *)input_a[i], 0, NULL, NULL);
    checkError(status, "Failed to unmap input A");
    status = clEnqueueSVMUnmap(queue[i], (void *)input_b[i], 0, NULL, NULL);
    checkError(status, "Failed to unmap input B");
#endif /* USE_SVM_API == 0 */
  }
}

void run() {
  cl_int status;

  const double start_time = getCurrentTimestamp();

  // Launch the problem for each device.
  scoped_array<cl_event> kernel_event(num_devices);
  scoped_array<cl_event> finish_event(num_devices);

  for(unsigned i = 0; i < num_devices; ++i) {

#if USE_SVM_API == 0
    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[8];
    status = clEnqueueWriteBuffer(queue[i], residualFlow_buf[i], CL_FALSE,
        0, n_per_device[i]*n_per_device[i] * sizeof(int), residualFlow[i], 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue[i], height_buf[i], CL_FALSE,
        0, n_per_device[i] * sizeof(int), height[i], 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");
    
    status = clEnqueueWriteBuffer(queue[i], excessFlow_buf[i], CL_FALSE,
        0, n_per_device[i] * sizeof(int), excessFlow[i], 0, NULL, &write_event[2]);
    checkError(status, "Failed to transfer input C");

    status = clEnqueueWriteBuffer(queue[i], netFlowOutS_buf, CL_FALSE,
        0, sizeof(int), &OutS, 0, NULL, &write_event[3]);
    checkError(status, "Failed to transfer input D");

    status = clEnqueueWriteBuffer(queue[i], netFlowInT_buf, CL_FALSE,
        0, sizeof(int), &InT, 0, NULL, &write_event[4]);
    checkError(status, "Failed to transfer input D");

    status = clEnqueueWriteBuffer(queue[i], s_buf, CL_FALSE,
        0, sizeof(int), &S, 0, NULL, &write_event[5]);
    checkError(status, "Failed to transfer input D");

    status = clEnqueueWriteBuffer(queue[i], t_buf, CL_FALSE,
        0, sizeof(int), &T, 0, NULL, &write_event[6]);
    checkError(status, "Failed to transfer input D");

    status = clEnqueueWriteBuffer(queue[i], n_buf, CL_FALSE,
        0, sizeof(int), &NodesNo, 0, NULL, &write_event[7]);
    checkError(status, "Failed to transfer input D");

    /*status = clEnqueueWriteBuffer(queue[i], input_b_buf[i], CL_FALSE,
        0, n_per_device[i]*n_per_device[i] * sizeof(int), input_b[i], 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");*/
#endif /* USE_SVM_API == 0 */

    // Set kernel arguments.
    unsigned argi = 0;

#if USE_SVM_API == 0
    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &residualFlow_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    /*status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_b_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);*/

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &height_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &excessFlow_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &netFlowOutS_buf);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &netFlowInT_buf);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &s_buf);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &t_buf);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &n_buf);
    checkError(status, "Failed to set argument %d", argi - 1);

    /*status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &output_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);*/
#else
    status = clSetKernelArgSVMPointer(kernel[i], argi++, (void*)input_a[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArgSVMPointer(kernel[i], argi++, (void*)input_b[i]);
    checkError(status, "Failed to set argument %d", argi - 1)output_buf;

    status = clSetKernelArgSVMPointer(kernel[i], argi++, (void*)output[i]);
    checkError(status, "Failed to set argument %d", argi - 1);
#endif /* USE_SVM_API == 0 */

    // Enqueue kernel.
    // Use a global work size corresponding to the number of elements to add
    // for this device.
    //
    // We don't specify a local work size and let the runtime choose
    // (it'll choose to use one work-group with the same size as the global
    // work-size).
    //
    // Events are used to ensure that the kernel is not launched until
    // the writes to the input buffers have completed.
    const size_t global_work_size = n_per_device[i];
    printf("Launching for device %d (%zd elements)\n", i, global_work_size);

#if USE_SVM_API == 0
    status = clEnqueueNDRangeKernel(queue[i], kernel[i], 1, NULL,
        &global_work_size, NULL, 8, write_event, &kernel_event[i]);
#else
    status = clEnqueueNDRangeKernel(queue[i], kernel[i], 1, NULL,
        &global_work_size, NULL, 0, NULL, &kernel_event[i]);
#endif /* USE_SVM_API == 0 */
    checkError(status, "Failed to launch kernel");

#if USE_SVM_API == 0
    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue[i], residualFlow_buf[i], CL_FALSE,
        0, n_per_device[i]*n_per_device[i]* sizeof(int), residualFlow[i], 1, &kernel_event[i], &finish_event[i]);

    status = clEnqueueReadBuffer(queue[i], height_buf[i], CL_FALSE,
        0, n_per_device[i]* sizeof(int), height[i], 1, &kernel_event[i], &finish_event[i]);

    status = clEnqueueReadBuffer(queue[i], excessFlow_buf[i], CL_FALSE,
        0, n_per_device[i]* sizeof(int), excessFlow[i], 1, &kernel_event[i], &finish_event[i]);

    status = clEnqueueReadBuffer(queue[i], netFlowOutS_buf, CL_FALSE,
        0, sizeof(int), &OutS, 1, &kernel_event[i], &finish_event[i]);

    status = clEnqueueReadBuffer(queue[i], netFlowInT_buf, CL_FALSE,
        0, sizeof(int), &InT, 1, &kernel_event[i], &finish_event[i]);
        //-------------------------------------------------------------------------------
        //std::cout << output[i] << " ";
        //printf("chap %d \n",output[i] );
        //-------------------------------------------------------------------------------

    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
    clReleaseEvent(write_event[2]);
    clReleaseEvent(write_event[3]);
    clReleaseEvent(write_event[4]);
    clReleaseEvent(write_event[5]);
    clReleaseEvent(write_event[6]);
    clReleaseEvent(write_event[7]);
#else
    status = clEnqueueSVMMap(queue[i], CL_TRUE, CL_MAP_READ,
        (void *)output[i], n_per_device[i] * sizeof(int), 0, NULL, NULL);
        //------------------------------------
        //printf("chap %d \n",output[i] );
        //------------------------------------
    checkError(status, "Failed to map output");
	clFinish(queue[i]);
#endif /* USE_SVM_API == 0 */
  }

  // Wait for all devices to finish.
  clWaitForEvents(num_devices, finish_event);

  const double end_time = getCurrentTimestamp();

  // Wall-clock time taken.
  printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

  // Get kernel times using the OpenCL event profiling API.
  for(unsigned i = 0; i < num_devices; ++i) {
    cl_ulong time_ns = getStartEndTime(kernel_event[i]);
    printf("Kernel time (device %d): %0.3f ms\n", i, double(time_ns) * 1e-6);
  }

  /*for (int i = 0; i < 100; i++)
    {
        std::cout << output[0][i] << " ";
        //printf("chap %d \n",output[i] );
    }*/

  // Release all events.
  for(unsigned i = 0; i < num_devices; ++i) {
    clReleaseEvent(kernel_event[i]);
    clReleaseEvent(finish_event[i]);
  }

  // Verify results.
  bool pass = true;
  std::cout <<"residuals ";
  for(unsigned i = 0; i < num_devices && pass; ++i) {
    for(unsigned j = 0; j < n_per_device[i]*n_per_device[i] && pass; ++j) {
      
      std::cout << residualFlow[i][j] << " ";
    }
  }
  std::cout <<endl;
  std::cout <<"heights ";
  for(unsigned i = 0; i < num_devices && pass; ++i) {
    for(unsigned j = 0; j < n_per_device[i] && pass; ++j) {
      
      std::cout << height[i][j] << " ";
    }
  }
  std::cout <<endl;
  std::cout <<"excesses ";
  for(unsigned i = 0; i < num_devices && pass; ++i) {
    for(unsigned j = 0; j < n_per_device[i] && pass; ++j) {
      
      std::cout << excessFlow[i][j] << " ";
    }
  }
  std::cout <<endl;
  std::cout <<"out "<< OutS << " ";
  std::cout <<"in "<< InT << " ";
  //std::cout <<*n_buf;
  /*for(unsigned i = 0; i < num_devices && pass; ++i) {
    for(unsigned j = 0; j < n_per_device[i]*n_per_device[i] && pass; ++j) {
      std::cout << input_a[i][j] << " ";
    }
  }*/

#if USE_SVM_API == 1
  for (unsigned i = 0; i < num_devices; ++i) {
    status = clEnqueueSVMUnmap(queue[i], (void *)output[i], 0, NULL, NULL);
    checkError(status, "Failed to unmap output");
  }
#endif /* USE_SVM_API == 1 */
  printf("\nVerification: %s\n", pass ? "PASS" : "FAIL");
}

// Free the resources allocated during initialization
void cleanup() {
  for(unsigned i = 0; i < num_devices; ++i) {
    if(kernel && kernel[i]) {
      clReleaseKernel(kernel[i]);
    }
    if(queue && queue[i]) {
      clReleaseCommandQueue(queue[i]);
    }
#if USE_SVM_API == 0
    if(residualFlow_buf && residualFlow_buf[i]) {
      clReleaseMemObject(residualFlow_buf[i]);
    }
    /*if(input_b_buf && input_b_buf[i]) {
      clReleaseMemObject(input_b_buf[i]);
    }*/
    if(height_buf && height_buf[i]) {
      clReleaseMemObject(height_buf[i]);
    }
    if(excessFlow_buf && excessFlow_buf[i]) {
      clReleaseMemObject(excessFlow_buf[i]);
    }
    if(netFlowOutS_buf) {
      clReleaseMemObject(netFlowOutS_buf);
    }
    if(netFlowInT_buf) {
      clReleaseMemObject(netFlowInT_buf);
    }
    if(s_buf) {
      clReleaseMemObject(s_buf);
    }
    if(t_buf) {
      clReleaseMemObject(t_buf);
    }
    if(n_buf) {
      clReleaseMemObject(n_buf);
    }
    /*if(output_buf && output_buf[i]) {
      clReleaseMemObject(output_buf[i]);
    }*/
    
#else
    if(input_a[i].get())
      input_a[i].reset();
    if(input_b[i].get())
      input_b[i].reset();
    if(output[i].get())
      output[i].reset();
#endif /* USE_SVM_API == 0 */
  }

  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}


