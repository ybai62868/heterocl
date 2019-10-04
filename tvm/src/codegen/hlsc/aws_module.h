#ifndef AWS_MODULE_H
#define AWS_MODULE_H

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include "../build_common.h"

namespace TVM {
namespace runtime {

Module CreateAWSModule(
    LoweredFunc func,
    std::string code);

} // namespace runtime
} // namespace TVM

#endif
