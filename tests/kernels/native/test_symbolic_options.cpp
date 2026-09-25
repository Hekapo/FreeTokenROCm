// Inert standalone regression source. No build target or automatic runner is registered.
// Metadata only: fake device tags below must never reach a runtime, allocator or kernel.
#include <freetoken/tensor.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename F>
void expect_panic(F&& action, std::string_view message) {
  try {
    std::forward<F>(action)();
  } catch (const host::PanicError& error) {
    require(std::string_view(error.what()).find(message) != std::string_view::npos,
            "unexpected validation diagnostic");
    return;
  }
  throw std::runtime_error("expected host::PanicError");
}

constexpr auto kInt32 = DLDataType{kDLInt, 32, 1};
constexpr auto kInt64 = DLDataType{kDLInt, 64, 1};
constexpr auto kUInt64 = DLDataType{kDLUInt, 64, 1};
constexpr auto kFloat32 = DLDataType{kDLFloat, 32, 1};

// Static storage: symbol options are spans and must outlive every matcher that reads them.
constexpr auto kFloat32Only = std::array<DLDataType, 1>{kFloat32};
constexpr auto kInt32Only = std::array<DLDataType, 1>{kInt32};
constexpr auto kCpuOnly = std::array<DLDevice, 1>{DLDevice{kDLCPU, host::details::kAnyDeviceID}};

// One element of shape [1]; non-zero shape keeps these cases independent of D6.
struct Metadata {
  std::array<std::int64_t, 1> shape{1};
  std::array<std::int64_t, 1> strides{1};
  std::array<std::int64_t, 2> storage{};
  DLTensor tensor{};

  explicit Metadata(DLDataType dtype, DLDevice device = DLDevice{kDLCPU, 0}) {
    tensor.data = storage.data();
    tensor.device = device;
    tensor.ndim = 1;
    tensor.dtype = dtype;
    tensor.shape = shape.data();
    tensor.strides = strides.data();
    tensor.byte_offset = 0;
  }

  Metadata(const Metadata&) = delete;
  Metadata& operator=(const Metadata&) = delete;

  auto view() const -> tvm::ffi::TensorView {
    return tvm::ffi::TensorView(&tensor);
  }
};

auto same(DLDataType lhs, DLDataType rhs) -> bool {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

// host::operator==(DLDevice, DLDevice) is not found by ADL from here; compare fields.
auto same(DLDevice lhs, DLDevice rhs) -> bool {
  return lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id;
}

// --- dtype ---------------------------------------------------------------

void fresh_dtype_symbol_rejects_outside_allowlist() {
  Metadata count{kInt32};
  host::SymbolicDType dtype;
  expect_panic([&] {
    host::TensorMatcher({1}).with_dtype<std::int64_t>(dtype).verify(count.view());
  }, "not in the allowed options");
}

void rejected_dtype_does_not_bind_symbol() {
  Metadata wrong{kInt32}, right{kInt64};
  host::SymbolicDType dtype;
  expect_panic([&] {
    host::TensorMatcher({1}).with_dtype<std::int64_t>(dtype).verify(wrong.view());
  }, "not in the allowed options");
  require(!dtype.has_value(), "rejected dtype must not bind the symbol");
  host::TensorMatcher({1}).with_dtype<std::int64_t>(dtype).verify(right.view());
  require(same(dtype.unwrap(), kInt64), "later allowed dtype must bind");
}

void fresh_dtype_symbol_accepts_member() {
  Metadata count{kInt64};
  host::SymbolicDType dtype;
  host::TensorMatcher({1}).with_dtype<std::int64_t>(dtype).verify(count.view());
  require(same(dtype.unwrap(), kInt64), "allowed dtype must bind");
}

void signedness_is_part_of_the_allowlist() {
  Metadata count{kUInt64};
  host::SymbolicDType dtype;
  expect_panic([&] {
    host::TensorMatcher({1}).with_dtype<std::int64_t>(dtype).verify(count.view());
  }, "not in the allowed options");
}

void multi_member_allowlist_still_requires_equality() {
  Metadata first{kInt32}, second{kInt64};
  host::SymbolicDType dtype;
  expect_panic([&] {
    host::TensorMatcher({1}).with_dtype<std::int32_t, std::int64_t>(dtype)
        .verify(first.view()).verify(second.view());
  }, "DType mismatch");
  require(same(dtype.unwrap(), kInt32), "first allowed dtype stays bound");
}

void bound_dtype_outside_new_allowlist_rejects() {
  Metadata indices{kInt32};
  host::SymbolicDType dtype;
  host::TensorMatcher({1}).with_dtype(dtype).verify(indices.view());
  expect_panic([&] {
    host::TensorMatcher({1}).with_dtype<std::int64_t>(dtype).verify(indices.view());
  }, "not in the allowed options");
  require(same(dtype.unwrap(), kInt32), "earlier binding must survive");
}

void bound_dtype_inside_allowlist_but_unequal_rejects() {
  Metadata first{kInt32}, second{kInt64};
  host::SymbolicDType dtype;
  host::TensorMatcher({1}).with_dtype(dtype).verify(first.view());
  expect_panic([&] {
    host::TensorMatcher({1}).with_dtype<std::int32_t, std::int64_t>(dtype).verify(second.view());
  }, "DType mismatch");
}

void symbol_options_are_not_overwritten() {
  Metadata count{kInt64};
  host::SymbolicDType dtype;
  dtype.set_options(kInt32Only);
  expect_panic([&] {
    host::TensorMatcher({1}).with_dtype<std::int32_t, std::int64_t>(dtype).verify(count.view());
  }, "not in the allowed options");
  require(!dtype.has_value(), "symbol options must still reject");
}

void disjoint_dtype_constraints_reject_everything() {
  Metadata as_int{kInt64}, as_float{kFloat32};
  host::SymbolicDType dtype;
  dtype.set_options(kFloat32Only);
  // Empty intersection must reject both sides, never collapse to "unconstrained".
  expect_panic([&] {
    host::TensorMatcher({1}).with_dtype<std::int64_t>(dtype).verify(as_int.view());
  }, "not in the allowed options");
  expect_panic([&] {
    host::TensorMatcher({1}).with_dtype<std::int64_t>(dtype).verify(as_float.view());
  }, "not in the allowed options");
  require(!dtype.has_value(), "no value may bind under disjoint constraints");
}

void empty_pack_keeps_plain_symbol_equality() {
  Metadata data{kFloat32};
  host::SymbolicDType dtype;
  host::TensorMatcher({1}).with_dtype(dtype).verify(data.view());
  require(same(dtype.unwrap(), kFloat32), "plain symbol must stay unconstrained");
}

void no_symbol_dtype_overload_is_unchanged() {
  Metadata data{kFloat32};
  expect_panic([&] {
    host::TensorMatcher({1}).with_dtype<std::int32_t>().verify(data.view());
  }, "not in the allowed options");
}

// --- device --------------------------------------------------------------

void fresh_device_symbol_rejects_outside_allowlist() {
  Metadata indices{kInt32, DLDevice{kDLCPU, 0}};
  host::SymbolicDevice device;
  expect_panic([&] {
    host::TensorMatcher({1}).with_device<kDLROCM>(device).verify(indices.view());
  }, "not in the allowed options");
  require(!device.has_value(), "rejected device must not bind the symbol");
}

void device_allowlist_wildcards_the_id() {
  Metadata indices{kInt32, DLDevice{kDLROCM, 1}};
  host::SymbolicDevice device;
  host::TensorMatcher({1}).with_device<kDLCUDA, kDLROCM>(device).verify(indices.view());
  require(same(device.unwrap(), DLDevice{kDLROCM, 1}), "exact device must bind");
}

void bound_device_keeps_exact_id() {
  Metadata first{kInt32, DLDevice{kDLROCM, 1}}, second{kInt32, DLDevice{kDLROCM, 0}};
  host::SymbolicDevice device;
  host::TensorMatcher({1}).with_device<kDLROCM>(device).verify(first.view());
  expect_panic([&] {
    host::TensorMatcher({1}).with_device<kDLROCM>(device).verify(second.view());
  }, "Device mismatch");
}

void bound_device_outside_new_allowlist_rejects() {
  Metadata indices{kInt32, DLDevice{kDLCPU, 0}};
  host::SymbolicDevice device;
  host::TensorMatcher({1}).with_device(device).verify(indices.view());
  expect_panic([&] {
    host::TensorMatcher({1}).with_device<kDLCUDA, kDLROCM>(device).verify(indices.view());
  }, "not in the allowed options");
}

void disjoint_device_constraints_reject_everything() {
  Metadata host_side{kInt32, DLDevice{kDLCPU, 0}}, gpu_side{kInt32, DLDevice{kDLROCM, 0}};
  host::SymbolicDevice device;
  device.set_options(kCpuOnly);
  expect_panic([&] {
    host::TensorMatcher({1}).with_device<kDLROCM>(device).verify(host_side.view());
  }, "not in the allowed options");
  expect_panic([&] {
    host::TensorMatcher({1}).with_device<kDLROCM>(device).verify(gpu_side.view());
  }, "not in the allowed options");
  require(!device.has_value(), "no device may bind under disjoint constraints");
}

// --- per-bank copy shape of use (metadata only; the kernel is never called) ---

void per_bank_count_dtype_is_enforced() {
  const auto gpu = DLDevice{kDLROCM, 0};
  Metadata src_indices{kInt32, gpu}, dst_indices{kInt32, gpu}, count{kInt32, gpu};
  host::SymbolicDType indices_dtype, count_dtype;
  host::SymbolicDevice device;
  host::TensorMatcher({1}).with_dtype<std::int32_t, std::int64_t>(indices_dtype)
      .with_device<kDLCUDA, kDLROCM>(device)
      .verify(src_indices.view()).verify(dst_indices.view());
  // An int32 count would later be read through const int64_t*; it must be rejected here.
  expect_panic([&] {
    host::TensorMatcher({1}).with_dtype<std::int64_t>(count_dtype)
        .with_device<kDLCUDA, kDLROCM>(device).verify(count.view());
  }, "not in the allowed options");
}

using Test = std::pair<const char*, void (*)()>;
const Test cases[] = {
    {"fresh_dtype_symbol_rejects_outside_allowlist", fresh_dtype_symbol_rejects_outside_allowlist},
    {"rejected_dtype_does_not_bind_symbol", rejected_dtype_does_not_bind_symbol},
    {"fresh_dtype_symbol_accepts_member", fresh_dtype_symbol_accepts_member},
    {"signedness_is_part_of_the_allowlist", signedness_is_part_of_the_allowlist},
    {"multi_member_allowlist_still_requires_equality", multi_member_allowlist_still_requires_equality},
    {"bound_dtype_outside_new_allowlist_rejects", bound_dtype_outside_new_allowlist_rejects},
    {"bound_dtype_inside_allowlist_but_unequal_rejects", bound_dtype_inside_allowlist_but_unequal_rejects},
    {"symbol_options_are_not_overwritten", symbol_options_are_not_overwritten},
    {"disjoint_dtype_constraints_reject_everything", disjoint_dtype_constraints_reject_everything},
    {"empty_pack_keeps_plain_symbol_equality", empty_pack_keeps_plain_symbol_equality},
    {"no_symbol_dtype_overload_is_unchanged", no_symbol_dtype_overload_is_unchanged},
    {"fresh_device_symbol_rejects_outside_allowlist", fresh_device_symbol_rejects_outside_allowlist},
    {"device_allowlist_wildcards_the_id", device_allowlist_wildcards_the_id},
    {"bound_device_keeps_exact_id", bound_device_keeps_exact_id},
    {"bound_device_outside_new_allowlist_rejects", bound_device_outside_new_allowlist_rejects},
    {"disjoint_device_constraints_reject_everything", disjoint_device_constraints_reject_everything},
    {"per_bank_count_dtype_is_enforced", per_bank_count_dtype_is_enforced},
};

} // namespace

int main() {
  int failed = 0;
  for (const auto& [name, test] : cases) {
    try {
      test();
    } catch (const std::exception& error) {
      ++failed;
      std::cerr << name << ": " << error.what() << '\n';
    }
  }
  return failed == 0 ? 0 : 1;
}
