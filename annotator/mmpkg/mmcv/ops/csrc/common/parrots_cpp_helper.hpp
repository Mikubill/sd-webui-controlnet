#ifndef PARROTS_CPP_HELPER
#define PARROTS_CPP_HELPER
#include <parrots/darray/darraymath.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/darraylite.hpp>
#include <parrots/foundation/ssattrs.hpp>
#include <vector>

using namespace parrots;

#define PARROTS_PRIVATE_CASE_TYPE(prim_type, type, ...) \
  case prim_type: {                                     \
    using scalar_t = type;                              \
    return __VA_ARGS__();                               \
  }

#define PARROTS_DISPATCH_FLOATING_TYPES(TYPE, ...)                  \
  [&] {                                                             \
    const auto& the_type = TYPE;                                    \
    switch (the_type) {                                             \
      PARROTS_PRIVATE_CASE_TYPE(Prim::Float64, double, __VA_ARGS__) \
      PARROTS_PRIVATE_CASE_TYPE(Prim::Float32, float, __VA_ARGS__)  \
      default:                                                      \
        PARROTS_NOTSUPPORTED;                                       \
    }                                                               \
  }()

#define PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, ...)          \
  [&] {                                                              \
    const auto& the_type = TYPE;                                     \
    switch (the_type) {                                              \
      PARROTS_PRIVATE_CASE_TYPE(Prim::Float64, double, __VA_ARGS__)  \
      PARROTS_PRIVATE_CASE_TYPE(Prim::Float32, float, __VA_ARGS__)   \
      PARROTS_PRIVATE_CASE_TYPE(Prim::Float16, float16, __VA_ARGS__) \
      default:                                                       \
        PARROTS_NOTSUPPORTED;                                        \
    }                                                                \
  }()

#endif  // PARROTS_CPP_HELPER
