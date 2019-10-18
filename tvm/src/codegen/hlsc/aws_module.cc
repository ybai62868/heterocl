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


inline std::string Type2WrapStr(TVMType t) {
  std::string str = "";
  if (t.code == kDLInt) {
    if (t.fracs > 0) {
      str += "ap_fixed<";
      str += std::to_string(static_cast<int>(t.bits + t.fracs));
    } else {
      str += "ap_int<";
      if      (t.bits <= 8)  str += std::to_string(static_cast<int>(t.bits));
      else if (t.bits <= 16) str += "16";
      else if (t.bits <= 32) str += "32";
      else                   str += "64";
    }     
    if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.bits)) + ">";
    else             str += ">";
  } else if (t.code == kDLUInt) {
    if (t.fracs > 0) {
      str += "ap_ufixed<";
      str += std::to_string(static_cast<int>(t.bits + t.fracs));
    } else {
      str += "ap_uint<";
      if      (t.bits <= 8)  str += std::to_string(static_cast<int>(t.bits));
      else if (t.bits <= 16) str += "16";
      else if (t.bits <= 32) str += "32";
      else                   str += "64"; 
    }
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

      stream << "arg_top_" << nth_arr;
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
      stream << "[i" << arr->ndim-1;
      int mul2 = 1;
      for (int j = arr->ndim-2; j >= 0; j--) {
        mul2 *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul2;
      }

      stream << "])";

      // for (int j = 0; j < arr->ndim; j++) {
      //   stream << "[i" << j << "]"; 
      // }
      // stream << ")";
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



void GenKernelCode(std::string test_file) {
  int indent = 0;
  std::ofstream stream;
  stream.open("/home/centos/src/project_data/lab_digitrec_aws/solution/src/kernel/knn_vhls.cpp");
  // stream.open("knn_vhls_auto.cpp");
  stream << test_file;
  stream.close();
}

void GenWrapperCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 LoweredFunc func) {
  std::ofstream stream;
  stream.open("/home/centos/src/project_data/lab_digitrec_aws/solution/src/kernel/digitrec.cpp");
  int indent = 0;
  // stream.open("digitrec.cpp");
  stream << "#include <stdio.h>\n";
  stream << "#include \"/home/centos/src/project_data/lab_digitrec_aws/solution/src/kernel/knn_vhls.cpp\"\n";
  stream << "\n\n";
  stream << "extern \"C\" \n";
  stream << "{\n";
  indent += 2;
  PrintIndent(stream, indent);
  stream << "void DigitRec( ";
  for (int i = 0;i < args.size();i++) {
    if (i!=args.size() - 1) {
      stream << Type2WrapStr(arg_types[i]);
      stream << "*";
      stream << " source_wrapper_" << i;
      stream << ", ";
    } else {
      stream << Type2WrapStr(arg_types[i]);
      stream << "*";
      stream << " source_wrapper_" << i;
      stream << " ) {\n";
    }
  }
  stream << "\n\n";
  PrintIndent(stream, indent);
  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "#pragma HLS INTERFACE m_axi port= ";
    stream << "source_wrapper_" << i;
    stream << " offset=slave bundle=gmem\n";
  }
  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "#pragma HLS INTERFACE s_axilite port= ";
    stream << "source_wrapper_" << i;
    stream << " bundle=control\n";
  }
  PrintIndent(stream, indent);
  stream << "#pragma HLS INTERFACE s_axilite port=return bundle=control\n";
  stream << "\n\n";
  for (int i = 1;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << Type2WrapStr(arg_types[i]);
    stream << " source_wrapper_temp_" << i;
    TVMArray* arr = args[i];
    for (int j = 0;j < arr->ndim;j++) {
      stream << "[" << arr->shape[j] << "]";
    }
    stream << ";\n";
  }

  for (int i = 1;i < args.size();i++) {
    TVMArray* arr = args[i];
    for (int j = 0;j < arr->ndim;j++) {
      PrintIndent(stream, indent);
      stream << "for ( int i" << j << " = 0; ";
      stream << "i" << j << " < " << arr->shape[j] << "; ";
      stream << "i" << j << "++) {\n";
      indent += 2;
      if (j == arr->ndim - 1) {
        PrintIndent(stream, indent);
        stream << "source_wrapper_temp_" << i;
        for (int k = 0;k < arr->ndim;k++) {
          stream << "[i" << k << "]";
        }
        stream << " = ";
        stream << "source_wrapper_" << j;
        stream << "[i" << arr->ndim-1;
        int mul = 1;
        for (int k = arr->ndim-2; k >= 0;k--) {
          mul *= arr->shape[k+1];
          stream << "+ i" << k << "*" << mul;
        }
        stream << "];\n";
      }
    }
    for (int j = 0;j < arr->ndim;j++) {
      indent -= 2;
      PrintIndent(stream, indent);
      stream << "}\n";
    }

  }

  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "default_function( ";
  for (int i = 0;i < args.size();i++) {
    if (i == 0) {
      stream << "source_wrapper_" << i;
      stream << "[0], ";
    } else if (i !=0 && i!=args.size() - 1){
      stream << "source_wrapper_temp_" << i;
      stream << ", ";
    } else {
      stream << "source_wrapper_temp_" << i;
      stream << ");\n";
    }

  }
  stream << "\n\n";

int index = args.size() - 1;
TVMArray* arr = args[index];
for (int i = 0;i < arr->ndim;i++) {
  PrintIndent(stream, indent);
  stream << "for ( int i" << i << " = 0; ";
  stream << "i" << i << " < " << arr->shape[i] <<  "; ";
  stream << "i" << i << "++) {\n";
  indent += 2;

  if (i == arr->ndim - 1) {
    PrintIndent(stream, indent);
    stream << "source_wrapper_" << index;
    stream << "[i" << arr->ndim-1;
    int mul = 1;
    for (int j = arr->ndim-2; j >= 0;j--) {
      mul *= arr->shape[j+1];
      stream << " + i" << j << "*" << mul;
    }
    stream << " ] = ";

    stream << "source_wrapper_temp_" << index;
    for (int j = 0;j < arr->ndim;j++) {
      stream << "[i" << j << "]";
    }
    stream <<";\n";
  }
}
for (int i = 0;i < arr->ndim;i++) {
    indent -= 2;
    PrintIndent(stream, indent);
    stream << "}\n";
  }
stream << "}\n";
indent -= 2;
stream << "}\n";



  stream.close();
}





void GenHostCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 LoweredFunc func,
                 std::string test_file) {
  int indent = 0;
  std::ofstream stream;
  // stream.open("digit_recognition.cpp");
  GenKernelCode(test_file);
  stream.open("/home/centos/src/project_data/lab_digitrec_aws/solution/src/host/digit_recognition.cpp");
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
  stream << "using namespace rosetta;\n";
  stream << "\n\n";
  stream << "//other headers\n";
  stream << "#include \"utils.h\"\n";
  // stream << "#include \"typedefs.h\"\n";
  stream << "int main(int argc, char ** argv) {\n";
  indent += 2;

  int cnt = 0; // label the constant value
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

      stream << "[";
      for (int j = 0; j < arr->ndim; j++) {
        //stream << "[" << arr->shape[j] << "]";
        if (j == arr->ndim-1) {
          stream << arr->shape[j];
        } else {
          stream << arr->shape[j];
          stream << " * ";
        }
      }
      stream << "];\n";
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
      stream << "arg_top_" << i;
      stream << " = (";
      stream << Type2Byte(arg_types[i]);

      stream << ")(arg_" << i << ")";
      if (arg_types[i].fracs > 0)
        stream << " >> " << static_cast<int>(arg_types[i].fracs);
      stream << ";\n";

      PrintIndent(stream, indent);
      stream << Type2Byte(arg_types[i]) << " ";
      stream << "fool_" << cnt << "[1] = { arg_top_" << i << " };\n";
      cnt += 1;
    }
    stream << "\n\n";
  }




  PrintIndent(stream,indent);
  stream << "printf(\"Digit Recognition Application\\n\");\n";

  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// parse command line arguments for opencl version\n";
  PrintIndent(stream, indent);
  stream << "std::string kernelFile(\"\");\n";
  PrintIndent(stream, indent);
  stream << "parse_sdaccel_command_line_args(argc, argv, kernelFile);\n";
  stream << "\n\n";





  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// create OpenCL world\n";
  PrintIndent(stream, indent);
  stream << "CLWorld digit_rec_world = CLWorld(TARGET_DEVICE, CL_DEVICE_TYPE_ACCELERATOR);\n";
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
  stream << "// create mem objects\n";
  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    if (cnt!=0) {
      stream << "CLMemObj source_" << i;
      stream << "((void*)fool_" << cnt - 1;
      stream << ", sizeof(" << Type2Byte(arg_types[i]) << "), ";
      stream << "1, ";
      stream << "CL_MEM_READ_WRITE);\n";
      cnt--;
      continue;
    }
    stream << "CLMemObj source_" << i;
    stream << "((void*)arg_top_" << i;
    stream << ", sizeof(" << Type2Byte(arg_types[i]) << "), ";
    // stream << ", sizeof(" << Type2ExtStr(arg_types[i]) << "), ";

    TVMArray* arr = args[i];
    for (int j = 0;j < arr->ndim;j++) {
      if (j==0) {
        stream << arr->shape[j] << " ";
      } else {
        stream << "* " << arr->shape[j];
      }
    }
    stream << ", ";
    stream << "CL_MEM_READ_WRITE);\n";
  }


  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// add them to the world\n";
  // TODO
  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "digit_rec_world.addMemObj(source_" << i;
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
  stream << "DigitRec.set_local(local_size);\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// add them to the world\n";
  PrintIndent(stream, indent);
  stream << "digit_rec_world.addKernel(DigitRec);\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// set kernel arguments\n";
  // TODO
  // PrintIndent(stream, indent);
  // stream << "digit_rec_world.setConstKernelArg(0, 0, arg_top_0);\n";
  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "digit_rec_world.setMemKernelArg(0, "<< i << ", " << i;
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
  PrintIndent(stream, indent);
  int read_index = args.size() - 1;
  // stream << "digit_rec_world.readMemObj(2);\n";
  stream << "digit_rec_world.readMemObj( " << read_index << " );\n";

  // copy to shared mem
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      PrintCopyBack(arr, stream, indent, i);
      // PrintCopyBack2(arr, stream, indent, i);
      PrintIndent(stream, indent);
      stream << "shmdt(";
      stream << "arg_" << i << ");\n";
    }
  }

  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "}\n";
  stream.close();

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
        GenWrapperCode(args, shmids, arg_types, func_);
        GenHostCode(args, shmids, arg_types, func_, test_file_);
        // TODO: find a better way to do the following
        LOG(CLEAN) << "Compiling the generated AWS HLS code ...";
        // system("g++ main.cpp -o out");
        LOG(CLEAN) << "Running Software simulation ...";
        system("source ./run_sw.sh");
        LOG(CLEAN) << "Finished Software simulation";
        // system("rm out main.cpp");
        FreeSharedMem(args, shmids, arg_sizes);
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
