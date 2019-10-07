#include "./codegen_aocl.h"
#include "./codegen_sdaccel.h"
#include "../build_common.h"
#include "./sdaccel_module.h"
#include "./aocl_module.h"

namespace TVM {
namespace codegen {


// Xilinx FPGA OpenCL runtime
#if HCL_SDACCEL_RUNTIME
runtime::Module BuildSDAccelSim(Array<LoweredFunc> funcs) {
  CodeAnalysOpenCLC ca;
  CodeGenSDACCEL cg;
  for (LoweredFunc f : funcs) {
    // 1st pass: Analyze AST and collect necessary information
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    // 2nd pass: Generate kernel code
    cg.AddFunction(f, map_arg_type);
  }
  std::string code = cg.Finish();
  return runtime::CreateSDAccelModule(funcs[0], code);
}

TVM_REGISTER_API("codegen.build_sdaccel_csim")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildSDAccelSim(args[0]);
  });
#endif


// Intel FPGA OpenCL runtime
#if HCL_AOCL_RUNTIME
runtime::Module BuildAOCLSim(Array<LoweredFunc> funcs) {
  CodeAnalysOpenCLC ca;
  CodeGenAOCL cg;
  for (LoweredFunc f : funcs) {
    // 1st pass: Analyze AST and collect necessary information
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    // 2nd pass: Generate kernel code
    cg.AddFunction(f, map_arg_type);
  }
  std::string code = cg.Finish();
  return runtime::CreateAOCLModule(funcs[0], code);
}

TVM_REGISTER_API("codegen.build_aocl_csim")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildAOCLSim(args[0]);
  });
#endif


template<class CodeGen>
std::string BuildOpenCL(Array<LoweredFunc> funcs){
    using TVM::runtime::Registry;
    CodeAnalysOpenCLC ca;
    CodeGen cg;
    for(LoweredFunc f: funcs){
        ca.AddFunction(f);
        str2tupleMap<std::string, Type>map_arg_type;
        map_arg_type = ca.Finish();

        cg.AddFunction(f, map_arg_type);
    }
    std::string code = cg.Finish();

    LOG(WARNING) << "OpenCL doesn't have runtime, return kernel code";
    return code;
}


TVM_REGISTER_API("codegen.build_sdaccel")
.set_body([]( TVMArgs args, TVMRetValue * rv ) {
    * rv = BuildOpenCL<CodeGenSDACCEL>(args[0]);
    });

TVM_REGISTER_API("codegen.build_aocl")
.set_body([]( TVMArgs args, TVMRetValue * rv ) {
    * rv = BuildOpenCL<CodeGenAOCL>(args[0]);
    });
} // namespace codegen
} // namespace TVM
