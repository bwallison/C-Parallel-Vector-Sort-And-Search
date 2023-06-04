#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <math.h>


#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#include <cctype>
#endif

#include "Utils.h"

std::string last_token(std::string str)
{
	while (!str.empty() && std::isspace(str.back())) str.pop_back(); // remove trailing white space
		const auto pos = str.find_last_of(" \t\n"); // locate the last white space		
	return pos == std::string::npos ? str : str.substr(pos + 1); // if not found, return the entire string else return the tail after the space
}

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		cl::Context context = GetContext(platform_id, device_id); //Select computing devices

		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl; //display the selected device

		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE); //create a queue to which we will push commands for the device

		cl::Program::Sources sources; //Load & build the device code

		AddSources(sources, "kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef float mytype; //define mytype as a float
		std::vector<mytype> temperatureOriginal; //initialise vector to hold temperature values

		std::string line;
		ifstream myfile("temp_lincolnshire.txt"); //ifstream reads in only the temperature values as floats
		int i = 0;
		if (myfile.is_open())
		{
			while (getline(myfile, line))
				temperatureOriginal.push_back(stof(last_token(line)));
			myfile.close();
		}

		float originalSize = temperatureOriginal.size(); //store the original size of vector

		/* ###################### AVERAGE KERNEL ###################### */

		size_t local_size_add = 32; //initialise local size for addition

		size_t padding_size_add = temperatureOriginal.size() % local_size_add; //get size of padding for vector

		std::vector<mytype> temperatureAdd = temperatureOriginal; //copy vec contents

		if (padding_size_add) //if the input vector is not a multiple of the local_size_min, insert additional neutral elements (0 for addition) so that the total will not be affected
		{
			std::vector<float> A_ext(local_size_add - padding_size_add, 0); //create an extra vector with neutral values
			temperatureAdd.insert(temperatureAdd.end(), A_ext.begin(), A_ext.end()); //append that extra vector to our input
		}

		size_t input_elements_add = temperatureAdd.size();//number of input elements
		size_t input_size_add = temperatureAdd.size() * sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements_add / local_size_add;

		//host - output vector
		std::vector<mytype> output_add(input_elements_add); //create output vector of size of the temperature
		size_t output_size_add = output_add.size() * sizeof(mytype);//size in bytes
		
		//device - create buffers 
		cl::Buffer buffer_A_add(context, CL_MEM_READ_ONLY, output_size_add); //buffer a for temperatures
		cl::Buffer buffer_B_add(context, CL_MEM_READ_WRITE, output_size_add); //buffer b for output

		//initialise vectors on device memory
		queue.enqueueWriteBuffer(buffer_A_add, CL_TRUE, 0, input_size_add, &temperatureAdd[0]); //write temperature to buffer_A_add
		queue.enqueueFillBuffer(buffer_B_add, 0, 0, output_size_add); //zero B buffer on device memory

		//Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_add = cl::Kernel(program, "reduce_add"); //initiliase kernel
		kernel_add.setArg(0, buffer_A_add); //temperature argument
		kernel_add.setArg(1, buffer_B_add); //output argument
		kernel_add.setArg(2, cl::Local(local_size_add * sizeof(mytype))); //size of buffer

		cl::Event prof_event_add; //profiling of kernel
		cl::Event prof_event_add_read; //profiling of kernel
		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(input_elements_add), cl::NDRange(local_size_add), NULL, &prof_event_add);
		queue.enqueueReadBuffer(buffer_B_add, CL_TRUE, 0, output_size_add, &output_add[0], NULL, &prof_event_add_read); //read the output buffer
		float meanTemp = output_add[0] / temperatureAdd.size(); //calculate average from sum

		/* ###################### MINIMUM KERNEL ###################### */

		std::vector<mytype> temperatureMin = temperatureOriginal; //copy vec contents

		size_t local_size_min = 256; //initialise local size for minimum

		size_t padding_size_min = temperatureOriginal.size() % local_size_min;  //get size of padding for vector

		if (padding_size_min) //if the input vector is not a multiple of the local_size_add, insert additional neutral elements (0 for addition) so that the total will not be affected
		{
			std::vector<float> A_ext(local_size_min - padding_size_min, 0); //create an extra vector with neutral values
			temperatureMin.insert(temperatureMin.end(), A_ext.begin(), A_ext.end()); //append that extra vector to our input
		}

		size_t input_elements_min = temperatureMin.size();//number of input elements
		size_t input_size_min = temperatureMin.size() * sizeof(mytype);//size in bytes
		size_t nr_groups_min = input_elements_min / local_size_min;

		//host - output
		std::vector<mytype> output_min(input_elements_min); //ouput vector
		size_t output_size_min = output_min.size() * sizeof(mytype);//size in bytes

		//device - create buffers 
		cl::Buffer buffer_A_min(context, CL_MEM_READ_ONLY, output_size_min); //buffer which has temperature values
		cl::Buffer buffer_B_min(context, CL_MEM_READ_WRITE, output_size_min); //buffer for output

		//initialise vectors on device memory
		queue.enqueueWriteBuffer(buffer_A_min, CL_TRUE, 0, input_size_min, &temperatureMin[0]); //read in temperature to buffer on device memory
		queue.enqueueFillBuffer(buffer_B_min, 0, 0, output_size_min);//zero B buffer on device memory

		//Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_min = cl::Kernel(program, "reduce_min"); //initiliase min kernel
		kernel_min.setArg(0, buffer_A_min); //first argument is vec of temperatures
		kernel_min.setArg(1, buffer_B_min); //second argument is vec of output
		kernel_min.setArg(2, cl::Local(local_size_min * sizeof(mytype))); //size of buffer

		cl::Event prof_event_min; //profiling of kernel
		cl::Event prof_event_min_read; //profiling of kernel
		queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(input_elements_min), cl::NDRange(local_size_min), NULL, &prof_event_min);
		queue.enqueueReadBuffer(buffer_B_min, CL_TRUE, 0, output_size_min, &output_min[0], NULL, &prof_event_min_read); //read output vector from buffer
		float minTemp = output_min[0]; //get min temp

		/* ###################### MAXIMUM KERNEL ###################### */

		std::vector<mytype> temperatureMax = temperatureOriginal; //make a copy of temperature vector

		size_t local_size_max = 64;  //initialise local size for maximum

		size_t padding_size_max = temperatureOriginal.size() % local_size_max; //get size of padding for vector

		if (padding_size_max) //if the input vector is not a multiple of the local_size_max, insert additional neutral elements (0 for addition) so that the total will not be affected
		{
			std::vector<float> A_ext(local_size_max - padding_size_max, 0); //create an extra vector with neutral values
			temperatureMax.insert(temperatureMax.end(), A_ext.begin(), A_ext.end()); //append that extra vector to our input
		}

		size_t input_elements_max = temperatureMax.size();//number of input elements
		size_t input_size_max = temperatureMax.size() * sizeof(mytype);//size in bytes
		size_t nr_groups_max = input_elements_max / local_size_max;

		//host - output
		std::vector<mytype> output_max(input_elements_max); //initiliase vec for maximum output
		size_t output_size_max = output_max.size() * sizeof(mytype); //size in bytes

		//device - buffers
		cl::Buffer buffer_A_max(context, CL_MEM_READ_ONLY, output_size_max); //buffer which has temperature values
		cl::Buffer buffer_B_max(context, CL_MEM_READ_WRITE, output_size_max);  //buffer for output

		//copy vectors initialise other vectors on device memory
		queue.enqueueWriteBuffer(buffer_A_max, CL_TRUE, 0, input_size_min, &temperatureMax[0]); //write temperatures to the buffer
		queue.enqueueFillBuffer(buffer_B_max, 0, 0, output_size_max); //zero B buffer on device memory

		//Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_max = cl::Kernel(program, "reduce_max");
		kernel_max.setArg(0, buffer_A_max); //pass the temperatures as first argument
		kernel_max.setArg(1, buffer_B_max); //pass the output argument
		kernel_max.setArg(2, cl::Local(local_size_max * sizeof(mytype))); //size of buffer

		cl::Event prof_event_max; //profiling for reduce_max event
		cl::Event prof_event_max_read; //profiling for reduce_max event
		queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(input_elements_max), cl::NDRange(local_size_max), NULL, &prof_event_max);
		queue.enqueueReadBuffer(buffer_B_max, CL_TRUE, 0, output_size_max, &output_max[0], NULL, &prof_event_max_read); //read the output buffer to output_max
		float maxTemp = output_max[0]; //get the max temp from first element of the output

		/* ###################### STANDARD DEVIATION KERNEL ###################### */

		std::vector<mytype> temperatureStd = temperatureOriginal; //copy vec contents

		size_t local_size_std = 32; //set local size for standard deviation

		size_t padding_size_std = temperatureOriginal.size() % local_size_std; //get size of padding for vector

		if (padding_size_std) //if the input vector is not a multiple of the local_size_std, insert additional neutral elements (0 for addition) so that the total will not be affected
		{
			std::vector<float> A_ext(local_size_std - padding_size_std, 0); //create an extra vector with neutral values
			temperatureStd.insert(temperatureStd.end(), A_ext.begin(), A_ext.end()); //append that extra vector to our input
		}

		std::vector<mytype> mean = { meanTemp }; //vector containing mean value for calculating standard deviation

		size_t input_elements_std = temperatureStd.size();//number of input elements
		size_t input_size_std = temperatureStd.size() * sizeof(mytype);//size in bytes
		size_t nr_groups_std = input_elements_std / local_size_std;
		size_t input_size_std_mean = mean.size() * sizeof(mytype);//size in bytes

		//host - output
		std::vector<mytype> output_std(input_elements_std); //output  vector for standard deviation
		size_t output_size_std = output_std.size() * sizeof(mytype);//size in bytes


		//device - buffers
		cl::Buffer buffer_A_std(context, CL_MEM_READ_ONLY, output_size_std); //buffer for temperatures
		cl::Buffer buffer_B_std(context, CL_MEM_READ_WRITE, output_size_std); //buffer for output
		cl::Buffer buffer_C_std(context, CL_MEM_READ_ONLY, input_size_std_mean); //buffer for mean

		//Part 5 - device operations

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A_std, CL_TRUE, 0, input_size_std, &temperatureStd[0]); //read in temperatures to local memory
		queue.enqueueFillBuffer(buffer_B_std, 0, 0, output_size_std);//zero B buffer on device memory
		queue.enqueueWriteBuffer(buffer_C_std, CL_TRUE, 0, input_size_std_mean, &mean[0]); //read in mean to local memory

		//5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_std = cl::Kernel(program, "reduce_std"); //kernel for standard deviation
		kernel_std.setArg(0, buffer_A_std); //temperature buffer
		kernel_std.setArg(1, buffer_B_std); //output buffer
		kernel_std.setArg(2, cl::Local(local_size_std * sizeof(mytype))); //size of buffer
		kernel_std.setArg(3, buffer_C_std); //fourth argument - the mean vector

		cl::Event prof_event_std; //profiling of kernel
		cl::Event prof_event_std_read; //profiling of kernel
		queue.enqueueNDRangeKernel(kernel_std, cl::NullRange, cl::NDRange(input_elements_std), cl::NDRange(local_size_std), NULL, &prof_event_std);
		queue.enqueueReadBuffer(buffer_B_std, CL_TRUE, 0, output_size_std, &output_std[0], NULL, &prof_event_std_read); //read output from kernel

		float paddingValue = ((local_size_std - padding_size_std) * (((0 - mean[0])*(0 - mean[0]))) / 100); //calculate extra value for padding values
		output_std[0] = output_std[0] - paddingValue; //take extra value to get true standard deviation

		float stdev = sqrt(output_std[0]*100 / originalSize); //calculate last part of standard deviation

		/*
		//Attempt at Bitonic sort, using a test vector, works for sizes which do not exceed the maximum amount of local memeory

		std::vector<float> testVec(2000);

		for (int i = 0; i < 2000; i++)
			testVec[i] = 2000.00 - float(i);

		int counter = 0;
		while (1) {
			counter++;
			if (testVec.size() < pow(2,counter) )
				break;
		}

		int New_size = pow(2, counter);
		size_t padding_size1 = New_size - testVec.size();

		if (padding_size1) {
			std::vector<float> A_ext(padding_size1, 9000);
			testVec.insert(testVec.end(), A_ext.begin(), A_ext.end());
		}
		
		size_t local_size1 = 1024;
		
		size_t input_elements1 = testVec.size();//number of input elements
		size_t input_size1 = testVec.size() * sizeof(mytype);//size in bytes
		size_t nr_groups1 = input_elements1 / local_size1;

		size_t output_size1 = testVec.size() * sizeof(mytype);//size in bytes

		cl::Buffer buffer_A1(context, CL_MEM_READ_ONLY, input_size1);

		queue.enqueueWriteBuffer(buffer_A1, CL_TRUE, 0, input_size1, &testVec[0]);

		cl::Kernel kernel_2 = cl::Kernel(program, "sort_bitonic");
		kernel_2.setArg(0, buffer_A1);

		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements1), cl::NDRange(local_size1));
		queue.enqueueReadBuffer(buffer_A1, CL_TRUE, 0, output_size1, &testVec[0]);

		int paddingAfter = testVec.size();

		testVec.erase(std::remove(testVec.begin(), testVec.end(), 9000), testVec.end());

		for(int i = 0; i < testVec.size(); i++)
			std::cout << "\t" << testVec[i];
		*/

		//output run times and figures

		std::cout << "\nAddition execution time: " << prof_event_add.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_add.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns], " << GetFullProfilingInfo(prof_event_add, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Minimum execution time: " << prof_event_min.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_min.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns], " << GetFullProfilingInfo(prof_event_min, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Maximum execution time: " << prof_event_max.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_max.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns], " << GetFullProfilingInfo(prof_event_max, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Standard Deviation execution time: " << prof_event_std.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_std.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns], " << GetFullProfilingInfo(prof_event_std, ProfilingResolution::PROF_US) << std::endl;

		std::cout << "\nAddition read time: " << prof_event_add_read.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_add_read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns], " << GetFullProfilingInfo(prof_event_add_read, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Minimum read time: " << prof_event_min_read.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_min_read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns], " << GetFullProfilingInfo(prof_event_min_read, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Maximum read time: " << prof_event_max_read.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_max_read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns], " << GetFullProfilingInfo(prof_event_max_read, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Standard Deviation read time: " << prof_event_std_read.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_std_read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns], " << GetFullProfilingInfo(prof_event_std_read, ProfilingResolution::PROF_US) << std::endl;

		std::cout << "\nMean Temperature:\t" << meanTemp << std::endl;
		std::cout << "Minimum Temperature:\t" << minTemp << std::endl;
		std::cout << "Maximum Temperature:\t" << maxTemp << std::endl;
		std::cout << "Standard Deviation:\t" << stdev << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
