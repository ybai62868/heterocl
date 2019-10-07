/*!
 *  Copyright (c) 2018 by Contributors
 * \file build_vhls.cc
 * \brief Build HLS C modules from source.
 */
#include "./aws_module.h"
#include <fstream>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <iostream>

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
    if (t.fracs > 0) str += "ap_fixed<";
    else             str += "ap_int<";
    str += std::to_string(static_cast<int>(t.bits));
    if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.bits - t.fracs)) + ">";
    else             str += ">";
  } else if (t.code == kDLUInt) {
    if (t.fracs > 0) str += "ap_ufixed<";
    else             str += "ap_uint<";
    str += std::to_string(static_cast<int>(t.bits));
    if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.bits - t.fracs)) + ">";
    else             str += ">";
  } else if (t.code == kDLFloat) {
    str += "float";
  } else {
    LOG(FATAL) << "Unknown type";
  }
  return str;
}

inline std::string Type2ExtStr(TVMType t) {
  std::string str = "";
  if (t.code == kDLInt) {
    if (t.fracs > 0) str += "ap_fixed<";
    else             str += "ap_int<";
    str += std::to_string(static_cast<int>(t.bits + t.fracs));
    if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.bits)) + ">";
    else             str += ">";
  } else if (t.code == kDLUInt) {
    if (t.fracs > 0) str += "ap_ufixed<";
    else             str += "ap_uint<";
    str += std::to_string(static_cast<int>(t.bits + t.fracs));
    if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.bits)) + ">";
    else             str += ">";
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
    if (t.code == kDLUInt) str += "u";
    str += "int";
    if      (t.bits <= 8)  str += "8";
    else if (t.bits <= 16) str += "16";
    else if (t.bits <= 32) str += "32";
    else                   str += "64";
    str += "_t";
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
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      int shmid = shmids[i];
      void* mem = shmat(shmid, nullptr, 0);
      memcpy(arr->data, mem, arg_sizes[i]);
      shmdt(mem);
      shmctl(shmid, IPC_RMID, nullptr);
    }
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
      // stream << "arg_top_" << nth_arr;
      // for (int j = 0; j < arr->ndim; j++) {
      //   stream << "[i" << j << "]"; 
      // }

      stream << "arg_top" << nth_arr;
      stream << "[i" << arr->ndim-1;
      int mul2 = 1;
      for (int j = arr->ndim-2; j >= 0; j--) {
        mul2 *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul2;
      }
      stream << "]";


      stream << " = (";
      // stream << Type2ExtStr(arr->dtype);
      stream << Type2Byte(arr->dtype);

      stream << ")(arg_" << nth_arr;
      stream << "[i" << arr->ndim-1;
      int mul = 1;
      for (int j = arr->ndim-2; j >= 0; j--) {
        mul *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul;
      }
      stream << "])";
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
      stream << "] = (";
      // stream << Type2ExtStr(arr->dtype);
      stream << Type2Byte(arr->dtype);
      stream << ")(arg_top_" << nth_arr;
      for (int j = 0; j < arr->ndim; j++) {
        stream << "[i" << j << "]"; 
      }
      stream << ")";
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
  stream.open("digit_recognition.cpp");
  stream << "#include <sys/ipc.h>\n";
  stream << "#include <sys/shm.h>\n";
  stream << "\n\n";
  stream << "// standard C/C++ headers\n";
  stream << "#include <cstdio>\n";
  stream << "#include <cstdlib>\n";
  stream << "#include <getopt.h>\n";
  stream << "#include <string>\n";
  stream << "#include <time.h>\n";
  stream << "#include <sys/time.h>\n";
  stream << "\n\n";
  stream << "// opencl harness headers\n";
  stream << "#include \"CLWorld.h\"\n";
  stream << "#include \"CLKernel.h\"\n";
  stream << "#include \"CLMemObj.h\"\n";
  stream << "// harness namespace\n";
  stream << "using namespace rosetta\n";
  stream << "\n\n";
  stream << "//other headers\n";
  stream << "#include \"utils.h\"\n";
  stream << "#include \"typedefs.h\"\n";
  stream << "#include \"check_data.h\"\n";
  stream << "\n\n";
  stream << "// data\n";
  stream << "#include \"training_data.h\"\n";
  stream << "#include \"testing_data.h\"\n";
  stream << "int main(int argc, char ** argv) {\n";
  indent += 2;

  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      // read from the shared memory
      PrintIndent(stream, indent);
      stream << Type2Byte(arg_types[i]) << "* "; 
      stream << "arg_" << i << " = ";
      stream << "(" << Type2Byte(arg_types[i]) << "*)";
      stream << "shmat(" << shmids[i] << ", nullptr, 0);\n";
      PrintIndent(stream, indent);
      stream << Type2Byte(arg_types[i]) << " ";
      // stream << Type2Str(arg_types[i]) << " ";
      stream << "arg_top_" << i;
      TVMArray* arr = args[i];
      for (int j = 0; j < arr->ndim; j++)
        stream << "[" << arr->shape[j] << "]";
      stream << ";\n";
      // copy from shared mem
      PrintCopy(arr, stream, indent, i);
    } else {
      // directly assign the value to the variable
      PrintIndent(stream, indent);
      stream << Type2Byte(arg_types[i]) << " ";
      stream << "arg_" << i << " = ";
      stream << "(" << Type2Byte(arg_types[i]) << ")";
      if (args[i].type_code() == kDLInt || 
          args[i].type_code() == kDLUInt) {
        stream << int64_t(args[i]);
      }
      stream << ";\n";
      PrintIndent(stream, indent);
      stream << Type2Byte(arg_types[i]) << " ";
      // stream << Type2Str(arg_types[i]) << " ";
      stream << "arg_top_" << i;
      stream << " = (";
      // stream << Type2ExtStr(arg_types[i]);
      stream << Type2Byte(arg_types[i]);

      stream << ")(arg_" << i << ")";
      if (arg_types[i].fracs > 0)
        stream << " >> " << static_cast<int>(arg_types[i].fracs);
      stream << ";\n";
    }
    stream << "\n\n";
  }




  PrintIndent(stream,indent);
  stream << "printf(\"Digit Recognition Application\\n\");\n";

  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// parse command line arguments for opencl version\n";
  PrintIndent(stream, indent);
  stream << "std::string kernelFile("");\n";
  PrintIndent(stream, indent);
  stream << "parse_sdaccel_command_line_args(argc, argv, kernelFile);\n";
  stream << "\n\n";





  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// create OpenCL world\n";
  PrintIndent(stream, indent);
  stream << "CLWorld digit_rec_world = CLword(TARGET_DEVICE, CL_DEVICE_TYPE_ACCELERATOR);\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// add the bitstream file\n";
  PrintIndent(stream, indent);
  stream << "digit_rec_world.addProgram(kernelFile);\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// create kernels\n";
  PrintIndent(stream, indent);
  stream << "CLKernel DigitRec(digit_rec_world.getContext(), digit_rec_world.getProgram(), \"DigitRec\", digit_rec_world.getDevice());\n";

  stream << "\n\n";



  PrintIndent(stream, indent);
  // TODO
  stream << "// create space for the result\n";
  PrintIndent(stream, indent);
  stream << "bit4_t* results = new bit4_t[NUM_TEST];\n";
  PrintIndent(stream, indent);
  stream << "// create mem objects\n";
  for (int i = 0;i < 1;i++) {
    PrintIndent(stream, indent);
    stream << "CLMemObj training_mem";
    stream << " ( (void *)training_data, sizeof(digit), NUM_TRAINING * 10, CL_MEM_READ_WRITE;\n";
  }
  for (int i = 0;i < 1;i++) {
    PrintIndent(stream, indent);
    stream << "CLMemObj testing_data" << i;
    stream << " ( (void *)testing_data, sizeof(digit), NUM_TEST, CL_MEM_READ_ONLY;\n";
  }
  for (int i = 0;i < 1;i++) {
    PrintIndent(stream, indent);
    stream << "CLMemObj result_mem" << i;
    stream << " ( (void *)results, sizeof(bit4_t), NUM_TEST, CL_MEM_WRITE_ONLY;\n";
  }

  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "CLMemObj source_" << i;
    stream << "((void*)arg_top" << i;
    stream << ", sizeof(" << Type2Byte(arg_types[i]) << "), ";
  //   TVMArray* arr = args[i];
  //   for (int j = 0;j < arr->ndim;j++) {
  //     if (j==0) {
  //       stream << arr->shape[j] << " ";
  //     } else {
  //       stream << "*" << arr->shape[j];
  //     }
  //   }
  //   stream << ", CL_MEM_READ_WRITE);\n";
  // }
  }



  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// add them to the world\n";
  // TODO
  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "digit_rec_world.addMemObj(arg_top_" << i;
    stream << ");\n";
  }


  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << " // set work size\n";
  PrintIndent(stream, indent);
  stream << "int global_size[3] = {1, 1, 1};\n";
  PrintIndent(stream, indent);
  stream << "int local_size[3] = {1, 1, 1};\n";
  PrintIndent(stream, indent);
  stream << "DigitRec.set_global(global_size);\n";
  PrintIndent(stream, indent);
  stream << "Digit.set_local(local_size);\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// add them to the world\n";
  PrintIndent(stream, indent);
  stream << "digit_rec_world.addKernel(DigitRec);\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// set kernel arguments\n";
  // TODO
  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "digit_rec_world.setMemKernelArg(0, 1, " << i;
    stream << ");\n";
  }


  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// run\n";
  PrintIndent(stream, indent);
  stream << "digit_rec_world.runKernels();\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// read the data back\n";
  // TODO



  // copy to shared mem
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      PrintCopyBack(arr, stream, indent, i);
      PrintIndent(stream, indent);
      stream << "shmdt(";
      stream << "arg_" << i << ");\n";
    }
  }



  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// cleanup\n";
  PrintIndent(stream, indent);
  stream << "digit_rec_world.releaseWorld();\n";
  PrintIndent(stream, indent);
  stream << "delete []results;\n";

  stream << "}\n";
  stream.close();







 



  // stream << test_file;
  // stream << "int main(void) { \n";
  // indent += 2;
  // for (int i = 0; i < args.size(); i++) {
  //   if (args[i].type_code() == kArrayHandle) {
  //     // read from the shared memory
  //     PrintIndent(stream, indent);
  //     stream << Type2Byte(arg_types[i]) << "* "; 
  //     stream << "arg_" << i << " = ";
  //     stream << "(" << Type2Byte(arg_types[i]) << "*)";
  //     stream << "shmat(" << shmids[i] << ", nullptr, 0);\n";
  //     PrintIndent(stream, indent);
  //     stream << Type2Str(arg_types[i]) << " ";
  //     stream << "arg_top_" << i;
  //     TVMArray* arr = args[i];
  //     for (int j = 0; j < arr->ndim; j++)
  //       stream << "[" << arr->shape[j] << "]";
  //     stream << ";\n";
  //     // copy from shared mem
  //     PrintCopy(arr, stream, indent, i);
  //   } else {
  //     // directly assign the value to the variable
  //     PrintIndent(stream, indent);
  //     stream << Type2Byte(arg_types[i]) << " ";
  //     stream << "arg_" << i << " = ";
  //     stream << "(" << Type2Byte(arg_types[i]) << ")";
  //     if (args[i].type_code() == kDLInt || 
  //         args[i].type_code() == kDLUInt) {
  //       stream << int64_t(args[i]);
  //     }
  //     stream << ";\n";
  //     PrintIndent(stream, indent);
  //     stream << Type2Str(arg_types[i]) << " ";
  //     stream << "arg_top_" << i;
  //     stream << " = (";
  //     stream << Type2ExtStr(arg_types[i]);
  //     stream << ")(arg_" << i << ")";
  //     if (arg_types[i].fracs > 0)
  //       stream << " >> " << static_cast<int>(arg_types[i].fracs);
  //     stream << ";\n";
  //   }
  // }
  // // call the function
  // PrintIndent(stream, indent);
  // stream << func->name << "(";
  // for (int i = 0; i < args.size(); i++) {
  //   stream << "arg_top_" << i;
  //   if (i != args.size()-1) 
  //     stream << ", ";
  // }
  // stream << ");\n";
  // // copy to shared mem
  // for (int i = 0; i < args.size(); i++) {
  //   if (args[i].type_code() == kArrayHandle) {
  //     TVMArray* arr = args[i];
  //     PrintCopyBack(arr, stream, indent, i);
  //     PrintIndent(stream, indent);
  //     stream << "shmdt(";
  //     stream << "arg_" << i << ");\n";
  //   }
  // }
  // stream << "}\n";
  // stream.close();
}
} // namespace

class AWSHLSModuleNode final : public ModuleNode {
 public:
  AWSHLSModuleNode(LoweredFunc func, std::string test_file) 
    : func_(func), test_file_(test_file) {}

  const char* type_key() const {
    return "aws_hls_csim";
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
        GenHostCode(args, shmids, arg_types, func_, test_file_);
        // TODO: find a better way to do the following
        LOG(CLEAN) << "Compiling the generated AWS HLS code ...";
        // system("g++ main.cpp -o out");
        LOG(CLEAN) << "Running Software simulation ...";
        // system("./out");
        LOG(CLEAN) << "Finished Software simulation";
        // system("rm out main.cpp");
        // FreeSharedMem(args, shmids, arg_sizes);
      });
  }

 private:
  LoweredFunc func_;
  std::string test_file_;
};

Module CreateAWSHLSModule(
    LoweredFunc func,
    std::string code) {

  std::shared_ptr<AWSHLSModuleNode> n =
    std::make_shared<AWSHLSModuleNode>(func, code);

  return Module(n);
}

} // namespace runtime
} // namespace TVM