#include "./aocl_module.h"
#include <fstream>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <iostream>
#include <cstring>
#include <typeinfo>

namespace TVM {
namespace runtime {

namespace {


void PrintIndent(std::ofstream& stream, int indent) {
  for (int i = 0; i < indent; i++)
    stream << ' ';
}


inline size_t GetTypeSize(TVMType t) {
  size_t byte = (t.bits + 7) / 8;
  if (byte > 2){
    if (byte <= 4) byte = 4;
    else if (byte <= 8) byte = 8;
    else byte = 16;
  }
  return byte;
}


inline size_t GetDataSize(TVMArray* arr) {
  size_t size = 1;
  for (tvm_index_t i = 0; i < arr->ndim; ++i) {
    size *= arr->shape[i];
  }
  size_t byte = (arr->dtype.bits + 7) / 8;
  if (byte > 2){
    if (byte <= 4) byte = 4;
    else if (byte <= 8) byte = 8;
    else byte = 16;
  }
  size *= (byte * 8 * arr->dtype.lanes + 7) / 8;
  return size;
}



inline TVMType Type2TVMType(Type t) {
  TVMType tt;
  if (t.is_int())        tt.code = kDLInt;
  else if (t.is_uint())  tt.code = kDLUInt;
  else if (t.is_float()) tt.code = kDLFloat;
  else                   LOG(FATAL) << "Unacceptable type: " << t;
  tt.bits = static_cast<uint8_t>(t.bits());
  tt.fracs = static_cast<uint8_t>(t.fracs());
  return tt;
}

inline std::string Type2Str(TVMType t) {
  std::string str = "";
  if (t.code == kDLInt) {
    str += "int";
  } else if (t.code == kDLUInt) {
    str += "unsigned int";
  } else if (t.code == kDLFloat) {
    str += "float";
  } else {
    LOG(FATAL) << "Unknown type";
  }
  return str;
}

inline std::string Type2Byte(TVMType t) {
  std::string str = "";
  if (t.code == kDLFloat) {
    str += "float";
  } else if (t.code == kDLInt || t.code == kDLUInt) {
    if (t.code == kDLUInt) str += "unsigned";
    str += "int";
    if      (t.bits <= 8)  str += "8";
    else if (t.bits <= 16) str += "16";
    else if (t.bits <= 32) str += "32";
    else                   str += "64";
  }
  return str;
}

void CollectArgInfo(TVMArgs& args, 
                    LoweredFunc func,
                    std::vector<size_t>& arg_sizes,
                    std::vector<TVMType>& arg_types) {
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      arg_sizes.push_back(GetDataSize(arr));
      arg_types.push_back(arr->dtype);
    } else {
      const Variable* var = func->api_args[i].as<Variable>();
      TVMType t = Type2TVMType(var->type);
      arg_sizes.push_back(GetTypeSize(t));
      arg_types.push_back(t);
    }
  }
}

void GenSharedMem(TVMArgs& args,
                  std::vector<int>& shmids,
                  std::vector<size_t>& arg_sizes) {
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      // generate shared memory key and id
      // TODO: maybe get the current path??
      key_t key = ftok("/", i+1);
      int shmid = shmget(key, arg_sizes[i], 0666|IPC_CREAT);
      shmids.push_back(shmid);
      // copy mem from TVM args to the shared memory
      void* mem = shmat(shmid, nullptr, 0);
      memcpy(mem, arr->data, arg_sizes[i]);
    } else {
      shmids.push_back(0);
    }
  }
}

void FreeSharedMem(TVMArgs& args, 
                   const std::vector<int>& shmids,
                   std::vector<size_t>& arg_sizes) {
  for (size_t i = 0; i < shmids.size(); i++) {
      TVMArray* arr = args[i];
      int shmid = shmids[i];
      void* mem = shmat(shmid, nullptr, 0);
      memcpy(arr->data, mem, arg_sizes[i]);
      shmdt(mem);
      shmctl(shmid, IPC_RMID, nullptr);
  }
}

// copy values from the shared mem to local mem
void PrintCopy(TVMArray* arr, 
               std::ofstream& stream, 
               int indent, size_t nth_arr) {
  for (int i = 0; i < arr->ndim; i++) {
    PrintIndent(stream, indent);
    stream << "for (size_t i" << i << " = 0; ";
    stream << "i" << i << " < " << arr->shape[i] << "; ";
    stream << "i" << i << "++) {\n";
    indent += 2;
    if (i == arr->ndim-1) {
      PrintIndent(stream, indent);
      stream << "source_" << nth_arr;
      stream << "[i" << arr->ndim-1;
      int mul = 1;
      for (int j = arr->ndim-2;j >= 0;j--) {
        mul *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul;
      }
      stream << "] = ";
      stream << "arg_" << nth_arr;
      stream << "[i" << arr->ndim - 1;

      int mul2 = 1;
      for (int j = arr->ndim-2;j >= 0;j--) {
        mul2 *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul2;
      }
      stream << "]";
      if (arr->dtype.fracs > 0)
        stream << " >> " << static_cast<int>(arr->dtype.fracs);
      stream << ";\n";
    }
  }
  for (int i = 0; i < arr->ndim; i++) {
    indent -= 2;
    PrintIndent(stream, indent);
    stream << "}\n";
  }
}

// copy values from local mem back to shared mem
void PrintCopyBack(TVMArray* arr, 
                   std::ofstream& stream, 
                   int indent, size_t nth_arr) {
  for (int i = 0; i < arr->ndim; i++) {
    PrintIndent(stream, indent);
    stream << "for (size_t i" << i << " = 0; ";
    stream << "i" << i << " < " << arr->shape[i] << "; ";
    stream << "i" << i << "++) {\n";
    indent += 2;
    if (i == arr->ndim-1) {
      PrintIndent(stream, indent);
      stream << "arg_" << nth_arr;
      stream << "[i" << arr->ndim-1;
      int mul = 1;
      for (int j = arr->ndim-2; j >= 0; j--) {
        mul *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul;
      }
      stream << "] = ";
      stream << "source_" << nth_arr;
      stream << "[i" << arr->ndim - 1;
      int mul2 = 1;
      for (int j = arr->ndim-2;j >=0;j--) {
        mul2 *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul2;
      }
      stream << "]";
      if (arr->dtype.fracs > 0)
        stream << " << " << static_cast<int>(arr->dtype.fracs);
      stream << ";\n";
    }
  }
  for (int i = 0; i < arr->ndim; i++) {
    indent -= 2;
    PrintIndent(stream, indent);
    stream << "}\n";
  }
}




void GenHostCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 LoweredFunc func,
                 std::string test_file) {
  int indent = 0;
  std::ofstream stream;
  stream.open("main.cpp");
  indent += 2;

  stream << "#include <assert.h>\n";
  stream << "#include <stdio.h>\n";
  stream << "#include <stdlib.h>\n";
  stream << "#include <math.h>\n";
  stream << "#include <cstring>\n";
  stream << "#include \"CL/opencl.h\"\n";
  stream << "#include \"AOCLUtils/aocl_utils.h\"\n";
  stream << "\n\n";
  stream << "using namespace aocl_utils;\n";
  
  stream << "\n\n";
  stream << "// OpenCL runtime configuration\n";
  stream << "cl_platform_id platform = NULL;\n";
  stream << "unsigned num_devices = 0\n";
  stream << "scoped_array<cl_device_id> device;\n";
  stream << "cl_context context = NULL\n";
  stream << "scoped_array<cl_command_queue> queue;\n";
  stream << "cl_program program = NULL;\n";
  stream << "scoped_array<cl_kernel> kernel;\n";
  stream << "\n\n";
  

  stream << "// Control whether the emulator should be used.\n";
  stream << "bool use_emulator = false;\n";

  stream << "// Function prototypes\n";
  stream << "bool init_opencl()\n";
  stream << "void init_problem()\n";
  stream << "void run()\n";
  stream << "void cleanup()\n";

  stream << "\n\n";

  stream << "int main(int argc, char **argv) { \n";

  indent += 2;
  PrintIndent(stream, indent);
  stream << "Options options(argc, argv);\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// Optional argument to specify whether the emulator should be used.\n";
  PrintIndent(stream, indent);
  stream <<"use_emulator = options.get<bool>(\"emulator\");\n";  
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// Initialize OpenCL.\n";
  PrintIndent(stream, indent);
  stream << "if(!init_opencl()) {return -1};\n";
  stream << "\n\n";

  PrintIndent(stream, indent);
  stream << "// Requires the number of devices to be known.\n";
  PrintIndent(stream, indent);
  stream << "init_problem();\n";
  stream << "\n\n";


  PrintIndent(stream, indent);
  stream << "// Run the kernel.\n";
  PrintIndent(stream, indent);
  stream << "run()\n";
  stream << "\n\n";

  PrintIndent(stream, indent);
  stream << "// Free the resources allocated.\n";
  PrintIndent(stream, indent);
  stream << "cleanall()\n";

  PrintIndent(stream, indent);
  stream << "return 0;\n";
  stream << "}\n";
  stream << "\n\n";



  stream << "// Initializes the OpenCL objects.\n";
  stream << "bool init_opencl() {\n";
  PrintIndent(stream, indent);
  stream << "cl_int status;\n";
  PrintIndent(stream, indent);
  stream << "printf(\"Initializeing OpenCL\\n\");\n";
  PrintIndent(stream, indent);
  stream << "if(!setCwdToExeDir()) { return false;}\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// Get the OpenCL platform.\n";
  PrintIndent(stream, indent);
  stream << "platform = findPlatform(\"Intel(R) FPGA SDK for OpenCL(TM)\");\n";
  PrintIndent(stream, indent);
  stream << "// Query the available OpenCL device.\n";
  PrintIndent(stream, indent);
  stream << "device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));\n";
  PrintIndent(stream, indent);
  stream << "printf(\"Platform: %s\\n\", getPlatformName(platform).c_str());\n";
  PrintIndent(stream, indent);
  stream << "printf(\"Using %d device(s)\\n\", num_devices);\n";

  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// Create the context\n";
  PrintIndent(stream, indent);
  stream << "context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);\n";
  PrintIndent(stream, indent);
  stream << "checkError(status, \"Failed to create context\");\n";
  stream << "\n\n";

  PrintIndent(stream, indent);
  stream << "std::string binary_file = getBoardBinaryFile(\"default_function\", device[0]);\n";
  PrintIndent(stream, indent);
  stream << "printf(\"Using AOCX: %s\\n\", binary_file.c_str());\n";
  PrintIndent(stream, indent);
  stream << "program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);\n";

  PrintIndent(stream, indent);
  stream << "// Build the program that was just created.\n";
  PrintIndent(stream, indent);
  stream << "status = clBuildProgram(program, 0, NULL, "", NULL, NULL);\n";
  PrintIndent(stream, indent);
  stream << "checkError(status, \"Failed to build program\");\n";
  stream << "\n\n";

  PrintIndent(stream, indent);
  stream << "// Create per-device objects\n";
  PrintIndent(stream, indent);
  stream << "queue.reset(num_devices);\n";
  PrintIndent(stream, indent);
  stream << "kernel.reset(num_devices);\n";
  PrintIndent(stream, indent);
  stream << "n_per_device.reset(num_devices);\n";
  PrintIndent(stream, indent);
  // this part is for the buffer processing.
  // input_a_buf.reset(num_devices);
  // input_b_buf.reset(num_devices);
  // output_buf.reset(num_devices);

  PrintIndent(stream, indent);
  stream << "// Command Queue\n";
  PrintIndent(stream, indent);
  stream << "queue[0] = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);\n";
  PrintIndent(stream, indent);
  stream << "checkError(status, \"Failed to create command queue\");\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// Kernel\n";
  PrintIndent(stream, indent);
  stream << "const char *kernel_name = \"default_function\";\n";
  PrintIndent(stream, indent);
  stream << "kernel[0] = clCreateKernel(program, kernel_name, &status);\n";
  PrintIndent(stream, indent);
  stream << "checkError(status, \"Failed to create kernel\")\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// Determine the number of elements processed by the device\n";
  PrintIndent(stream, indent);
  stream << "n_per_device[0] = N;\n";

  // this part is for input buffers 
  // input_a_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, n_per_device[0] * sizeof(float), NULL, &status);


  // this part is for output buffers
  // output_buf[0] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_per_device[0] * sizeof(float), NULL, &status);

  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "return 0;\n";
  stream << "}\n";
  stream << "\n\n";

  stream << "void run() {\n";
  PrintIndent(stream, indent);
  stream << "cl_int status;\n";
  PrintIndent(stream, indent);
  stream << "// Launch the problem for the device\n";
  PrintIndent(stream, indent);
  stream << "scoped_array<cl_event> kernel_event(num_devices);\n"; 
  PrintIndent(stream, indent);
  stream << "scoped_array<cl_event> finish_event(num_devices);\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// for the host-to-device transfer\n";
  PrintIndent(stream, indent);
  stream << "cl_event write_event[2];\n";
  PrintIndent(stream, indent);
  // stream << "status = clEnqueueWriteBuffer(queue[0], input_a_buf)"
  // status = clEnqueueWriteBuffer(queue[0], input_a_buf[0], CL_FALSE, 0, n_per_device[0] * sizeof(float), 
  // 	input_a[i], 0, NULL, &write_event[0]);


  stream << "\n\n";
  PrintIndent(stream , indent);
  stream << "const size_t global_work_size = n_per_device[0];\n";
  PrintIndent(stream, indent);
  stream << "printf(\"Launching for device %d (%d elements)\\n\", i,  global_work_size);\n";
  PrintIndent(stream, indent);
  stream << "status = clEnqueueNDRangeKernel(queue[0], kernel[0], 1, NULL, &global_work_size, NULL, 2, write_event, &kernel_event[0]);\n";
  PrintIndent(stream, indent);
  stream << "checkError(status, \"Failed to Launch kernel\");\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// Read the result, this is the final operation;\n";
  stream << "\n\n";

  PrintIndent(stream, indent);
  stream << "// Release local events.\n";
  PrintIndent(stream ,indent);
  //clReleaseEvent(write_event[0]);
  stream << "\n\n";

  PrintIndent(stream, indent);
  stream << "// Release all events.\n";
  PrintIndent(stream, indent);
  stream << "clReleaseEvent(kernel_event[0];\n";
  PrintIndent(stream, indent);
  stream << "clReleaseEvent(finish_event[0];\n";
  stream << "}\n";


  stream << "\n\n";
  stream << "void cleanup() {\n";
  PrintIndent(stream, indent);
  stream << "if(kernel && kernel[0]) {clReleaseKernel(kernel[0]);}\n";
  PrintIndent(stream, indent);
  stream << "if(queue && queue[0]) {clReleaseCommandQueue(queue[0]);}\n";
  PrintIndent(stream, indent);

  // for input_a_buf, out_put_buf

  stream << "if(stream) {clReleaseProgram(program);}\n";
  PrintIndent(stream, indent);
  stream << "if(program) {clReleaseContext(context);}\n";
  stream << "}\n";


















  // Source Memories
  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "std::vector<" << Type2Str(arg_types[i]);
    stream << "> ";
    stream << "source_" << i << "(";
    TVMArray* arr = args[i];
    for (int j = 0;j < arr->ndim;j++) {
      if (j == arr->ndim-1) {
        stream << arr->shape[j] << ")";
      } else {
        // stream << " * " << arr->shape[j] << ")";
        stream << arr->shape[j] << " * ";
      }
    }
    stream << ";\n";
  }
  stream << "\n";

  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "size_t vector_size_bytes_" << i;
    stream << " = sizeof(" << Type2Str(arg_types[i]);
    stream << ")";
    TVMArray* arr = args[i];
    for (int j = 0;j < arr->ndim;j++) {
      stream << " * " << arr->shape[j];
    }
    stream << ";\n";
  }
  stream << "\n";

  for (int i = 0;i < args.size();i++ ) {
      // read from the shared memory
      PrintIndent(stream, indent);
      stream << Type2Str(arg_types[i]) << "* ";
      stream << "arg_" << i << " = ";
      stream << "(" << Type2Str(arg_types[i]) << "*)";
      stream << "shmat(" << shmids[i] << ", nullptr, 0);\n";
      TVMArray* arr = args[i];
      // copy from shared mem  
      PrintCopy(arr, stream, indent, i);
  }




  // Getting First Platform
  PrintIndent(stream, indent);
  stream << "std::vector<cl::Platform> platforms;\n";
  PrintIndent(stream, indent);
  stream << "cl::Platform::get(&platforms);\n";
  PrintIndent(stream, indent);
  stream << "cl::Platform platform = platforms[0];\n";
  stream << "\n";



  // Getting ACCELERATOR Devices and selecting 1st such device
  PrintIndent(stream, indent);
  stream << "std::vector<cl::Device> devices;\n";
  PrintIndent(stream, indent);
  stream << "platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);\n";
  PrintIndent(stream, indent);
  stream << "cl::Device device = devices[0];\n";
  stream << "\n";

  // Creating Context and Command Queue for selected Device
  PrintIndent(stream, indent);
  stream << "cl::Context context(device);\n";
  PrintIndent(stream, indent);
  stream << "cl::CommandQueue q(context, device);\n";
  stream << "\n";


  // Loading XCL Bin into char buffer
  PrintIndent(stream, indent);
  stream << "std::ifstream bin_file(xclbinFilename, std::ifstream::binary);\n";
  PrintIndent(stream, indent);
  stream << "bin_file.seekg (0, bin_file.end);\n";
  PrintIndent(stream, indent);
  stream << "unsigned nb = bin_file.tellg();\n";
  PrintIndent(stream, indent);
  stream << "bin_file.seekg (0, bin_file.beg);\n";
  PrintIndent(stream, indent);
  stream << "char *buf = new char [nb];\n";
  PrintIndent(stream, indent);
  stream << "bin_file.read(buf, nb);\n";
  stream << "\n";


  // Creating Program from Binary File
  PrintIndent(stream, indent);
  stream << "cl::Program::Binaries bins;\n";
  PrintIndent(stream, indent);
  stream << "bins.push_back({buf,nb});\n";
  PrintIndent(stream, indent);
  stream << "devices.resize(1);\n";
  PrintIndent(stream, indent);
  stream << "cl::Program program(context, devices, bins);\n";
  stream << "\n";


  // Creating Kernel and Functor of Kernel
  PrintIndent(stream, indent);
  stream << "int err1;\n";
  PrintIndent(stream, indent);
  stream << "cl::Kernel kernel(program, \"default_function\", &err1);\n";
  PrintIndent(stream, indent);
  stream << "auto default_function = cl::KernelFunctor<";
  for (int i = 0;i < args.size();i++) {
    if (i == args.size() - 1) {
      stream << "cl::Buffer&>(kernel);\n";
    } else {
      stream << "cl::Buffer&, ";
    }
  }
  // stream << "auto default_function = cl::KernelFunctor<cl::Buffer&, cl::Buffer&, cl::Buffer&>(kernel);\n";
  stream << "\n";


  // Creating Buffers inside Device
  // cl::Buffer buffer_a(context, CL_MEM_READ_ONLY,  vector_size_bytes);
  // cl::Buffer buffer_b(context, CL_MEM_WRITE_ONLY, vector_size_bytes);
  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "cl::Buffer buffer_" << i;
    stream << "(context, CL_MEM_READ_WRITE, vector_size_bytes_" << i << ");\n";
  }
  stream << "\n";

  // Copying input data to Device buffer from host memory
  // q.enqueueWriteBuffer(buffer_a, CL_TRUE, 0, vector_size_bytes, source_a.data());
  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "q.enqueueWriteBuffer(buffer_" << i;
    stream << ", CL_TRUE, 0, vector_size_bytes_" << i;
    stream << ", source_" << i << ".data());\n"; 
  }
  stream << "\n";

  // Running Kernel
  PrintIndent(stream, indent);
  stream << func->name << "(";
  stream << "cl::EnqueueArgs(q, cl::NDRange(1,1,1), cl::NDRange(1,1,1)),";
  for (int i = 0; i < args.size(); i++) {
    stream << "buffer_" << i;
    if (i != args.size()-1) 
      stream << ", ";
  }
  stream << ");\n";

  PrintIndent(stream, indent);
  stream << "q.finish();\n";
  stream << "\n";


  // Copying Device result data to Host memory
  // q.enqueueReadBuffer(buffer_c, CL_TRUE, 0, vector_size_bytes, result_krnl.data());
  for (int i = 0;i < args.size(); i++) {
    PrintIndent(stream, indent);
    stream << "q.enqueueReadBuffer(buffer_" << i;
    stream << ", CL_TRUE, 0, vector_size_bytes_" << i;
    stream << ", source_" << i << ".data());\n";
  }
  stream << "\n";

  // copy to shared mem
  for (int i = 0;i < args.size();i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      PrintCopyBack(arr, stream, indent, i);
      PrintIndent(stream, indent);
      stream << "shmdt(";
      stream << "arg_" << i << ");\n";
    }
  }

  stream << "}\n";
  stream.close();
}
} // namespace



class AOCLModuleNode final : public ModuleNode {
 public:
  AOCLModuleNode(LoweredFunc func, std::string test_file) 
    : func_(func), test_file_(test_file) {}

  const char* type_key() const {
    return "aocl_sw_emu";

  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    return PackedFunc([this](TVMArgs args, TVMRetValue* rv){

        if (args.size() != (int)func_->args.size())
          LOG(FATAL) << "The function should take in " << func_->args.size() 
                     << " inputs but get " << args.size();
        std::vector<size_t> arg_sizes;
        std::vector<TVMType> arg_types;
        std::vector<int> shmids;
        CollectArgInfo(args, func_, arg_sizes, arg_types);
        GenSharedMem(args, shmids, arg_sizes);
        LOG(CLEAN) << "Creating a Host file for AOCL Runtime ...";

        GenHostCode(args, shmids, arg_types, func_, test_file_);

        LOG(CLEAN) << "Creating a Common folder for AOCL Runtime ...";
        // GenCommonFile();
        // system(""); // from common to current folder


        LOG(CLEAN) << "Creating a Makfile for compling the AOCL OpenCL Code ...";
        // system(""); // from common to current folder 


        LOG(CLEAN) << "Compiling the generated AOCL OpenCL Kernel Code ...";
        // system("aoc -march=emulator device/default_function.cl -o bin/default_function.aocx");
        LOG(CLEAN) << "Compiling the Host Code ...";
        system("make");
        LOG(CLEAN) << "Running AOCL OpenCL Software Simulation ...";
        // system("CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 bin/host ")
        LOG(CLEAN) << "Finished AOCL OpenCL Software Simulation ...";
        FreeSharedMem(args, shmids, arg_sizes);
      });
  }

 private:
  LoweredFunc func_;
  std::string test_file_;
};

Module CreateAOCLModule(
    LoweredFunc func,
    std::string code) {

  std::shared_ptr<AOCLModuleNode> n =
    std::make_shared<AOCLModuleNode>(func, code);

  return Module(n);
}

} // namespace runtime
} // namespace TVM