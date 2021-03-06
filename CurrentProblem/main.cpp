//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#define IDX(i, j, n) ((i) * (n) + (j))
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
//#include <math.h>
#include <cstring>
#include "graphs.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
using namespace std;

const int NodesNo = 6;
const int PowerTwo = NodesNo * NodesNo;
void InputGraph(Graph *g) {
	ifstream in;
	in.open("a.txt");

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
//class FlowGraph {
//public:
//private:
//};
//Flow myfunction(Graph *g,s) {
//	int *tempHeights = (int *)calloc(g->n, sizeof(int));
//	// initialize preflow
//	int flowOutS = 0;
//	int flowInT = 0;
//	tempHeights[s] = g->n;
//}

///
//  Constants
//
const int ARRAY_SIZE = 1000;

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;

	// First, select an OpenCL platform to run on.  For this example, we
	// simply choose the first available platform.  Normally, you would
	// query for all available platforms and select the most appropriate one.
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return NULL;
	}

	// Next, create an OpenCL context on the platform.  Attempt to
	// create a GPU-based context, and if that fails, try to create
	// a CPU-based context.
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
		NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
			NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
			return NULL;
		}
	}

	return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;

	// First get the size of the devices buffer
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
		return NULL;
	}

	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available.";
		return NULL;
	}

	// Allocate memory for the devices buffer
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if (errNum != CL_SUCCESS)
	{
		delete[] devices;
		std::cerr << "Failed to get device IDs";
		return NULL;
	}

	// In this example, we just choose the first available device.  In a
	// real program, you would likely use all available devices or choose
	// the highest performance device based on OpenCL device queries
	commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
	if (commandQueue == NULL)
	{
		delete[] devices;
		std::cerr << "Failed to create commandQueue for device 0";
		return NULL;
	}

	*device = devices[0];
	delete[] devices;
	return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
	cl_int errNum;
	cl_program program;

	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1,
		(const char**)&srcStr,
		NULL, NULL);
	if (program == NULL)
	{
		std::cerr << "Failed to create CL program from source." << std::endl;
		return NULL;
	}

	//errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	errNum = clBuildProgram(program, 0, NULL, "-g -s /home/bahram/Desktop/CurrentProblem/KernelOne.cl", NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog), buildLog, NULL);

		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		return NULL;
	}

	return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[8],
	int *residualFlow, int *height, int *excessFlow, int *FOut, int *FIn, int s, int t, int n)
{
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * PowerTwo, residualFlow, NULL);
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * NodesNo, height, NULL);
	//memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
	//sizeof(float) * ARRAY_SIZE, NULL, NULL);
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * NodesNo, excessFlow, NULL);
	memObjects[3] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(int) , FOut, NULL);
	memObjects[4] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(int) , FIn, NULL);
	memObjects[5] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int), &s, NULL);
	memObjects[6] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int), &t, NULL);
	memObjects[7] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int), &n, NULL);

	if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL || memObjects[3] == NULL || memObjects[4] == NULL || memObjects[5] == NULL || memObjects[6] == NULL || memObjects[7] == NULL)
	{
		std::cerr << "Error creating memory objects." << std::endl;
		return false;
	}

	return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
	cl_program program, cl_kernel kernel, cl_mem memObjects[8])
{
	for (int i = 0; i < 8; i++)
	{
		if (memObjects[i] != 0)
			clReleaseMemObject(memObjects[i]);
	}
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

	if (kernel != 0)
		clReleaseKernel(kernel);

	if (program != 0)
		clReleaseProgram(program);

	if (context != 0)
		clReleaseContext(context);

}

///
//	main() for HelloWorld example
//
int main(int argc, char** argv)
{
	cl_context context = 0;
	cl_command_queue commandQueue = 0;
	cl_program program = 0;
	cl_device_id device = 0;
	cl_kernel kernel = 0;
	cl_mem memObjects[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	cl_int errNum;

	// Create an OpenCL context on first available platform
	context = CreateContext();
	if (context == NULL)
	{
		std::cerr << "Failed to create OpenCL context." << std::endl;
		return 1;
	}

	// Create a command-queue on the first device available
	// on the created context
	commandQueue = CreateCommandQueue(context, &device);
	if (commandQueue == NULL)
	{
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Create OpenCL program from HelloWorld.cl kernel source
	program = CreateProgram(context, device, "KernelOne.cl");
	if (program == NULL)
	{
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Create OpenCL kernel
	kernel = clCreateKernel(program, "hello_kernel", NULL);
	if (kernel == NULL)
	{
		std::cerr << "Failed to create kernel" << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Create memory objects that will be used as arguments to
	// kernel.  First create host memory arrays that will be
	// used to store the arguments to the kernel
	int result[PowerTwo];

	//---------------------------------------------------------------------------------------------------
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

	int *residualFlow = (int *)calloc(g->n * g->n, sizeof(int));
	memcpy(residualFlow, finalFlow, (g->n * g->n) * sizeof(int));

	int *height = (int *)calloc(g->n, sizeof(int));
	memcpy(height, tempHeights, g->n * sizeof(int));

	int *excessFlow = (int *)calloc(g->n, sizeof(int));
	memcpy(excessFlow, tempExcessFlows, g->n * sizeof(int));

	int *FOut = (int *)calloc(1, sizeof(int));
	memcpy(FOut, &flowOutS, sizeof(int));

	int *FIn = (int *)calloc(1, sizeof(int));
	memcpy(FIn, &flowInT, sizeof(int));

	    /*for (int i = 0; i < NodesNo; i++)
		{
			std::cout << netFlowInT[i] << " ";
		}*/
		//cout << *netFlowInT << " ";
        free(tempExcessFlows);
	    free(tempHeights);
	    free(finalFlow);
	//myfunction(g);
	//---------------------------------------------------------------------------------------------------

	if (!CreateMemObjects(context, memObjects, residualFlow, height, excessFlow, FOut, FIn, s, t, g->n))
	{
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Set the kernel arguments (result, a, b)
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
	//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjects[3]);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjects[4]);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &memObjects[5]);
	errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &memObjects[6]);
	errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &memObjects[7]);
	//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error setting kernel arguments." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	size_t globalWorkSize[1] = { NodesNo };
	size_t localWorkSize[1] = { 1 };

	// Queue the kernel up for execution across the array
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
		globalWorkSize, localWorkSize,
		0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error queuing kernel for execution." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Read the output buffer back to the Host
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[0], CL_TRUE,
		0, PowerTwo * sizeof(int), residualFlow,
		0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error reading result buffer." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	errNum = clEnqueueReadBuffer(commandQueue, memObjects[3], CL_TRUE,0,sizeof(int), FOut,0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error reading result buffer." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	errNum = clEnqueueReadBuffer(commandQueue, memObjects[4], CL_TRUE,0,sizeof(int), FIn,0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error reading result buffer." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Output the result buffer
	for (int i = 0; i < PowerTwo; i++)
	{
		std::cout << residualFlow[i] << " ";
	}
	std::cout << std::endl;
	std::cout << *FOut << std::endl;
	//std::cout << std::endl;
	std::cout << *FIn << std::endl;
	std::cout << "Executed program succesfully." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);

	std::cin.get();
	
	free(residualFlow);
	free(height);
	free(excessFlow);

	return 0;
}