// Inert standalone regression source. No build target or automatic runner is registered.
#include <freetoken/tensor.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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

struct Metadata {
  std::array<std::int64_t, 2> shape;
  std::array<std::int64_t, 2> strides;
  std::array<std::int32_t, 64> storage{};
  DLTensor tensor{};

  explicit Metadata(std::int64_t n, std::int64_t stride = 1)
      : shape{n, 1}, strides{stride, 1} {
    tensor.data = storage.data();
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = 1;
    tensor.dtype = DLDataType{kDLInt, 32, 1};
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

void unbound_rejects_unwrap() {
  host::SymbolicSize size{"length"};
  require(!size.has_value(), "fresh symbol must be unbound");
  require(!size.get_value().has_value(), "unbound optional must be empty");
  require(size.value_or_name("shape", 0) == "length(shape#0)", "unbound name");
  expect_panic([&] { (void)size.unwrap(); }, "Size value is not set");
}

void zero_is_a_bound_value() {
  host::SymbolicSize size{"length"};
  size.set_value(0);
  require(size.has_value(), "zero must bind");
  require(size.get_value().has_value(), "optional zero is engaged");
  require(size.get_value().value() == 0, "optional must contain zero");
  require(size.unwrap() == 0, "unwrap must return zero");
  require(size.value_or_name("shape", 0) == "0", "bound zero diagnostic");
}

void repeated_zero_verify_is_allowed() {
  host::SymbolicSize size;
  size.verify(0, "shape", 0);
  size.verify(0, "shape", 0);
  require(size.has_value() && size.unwrap() == 0, "equal zeros must stay bound");
}

void equal_positive_verify_is_unchanged() {
  host::SymbolicSize size;
  size.verify(7, "shape", 0);
  size.verify(7, "shape", 0);
  require(size.unwrap() == 7, "equal positive sizes must match");
}

void unequal_positive_verify_rejects() {
  host::SymbolicSize size;
  size.verify(7, "shape", 0);
  expect_panic([&] { size.verify(8, "shape", 0); }, "Size mismatch");
  require(size.unwrap() == 7, "mismatch must not rebind");
}

void zero_then_positive_rejects() {
  host::SymbolicSize size;
  size.verify(0, "shape", 0);
  expect_panic([&] { size.verify(7, "shape", 0); }, "expected 0 but got 7");
  require(size.unwrap() == 0, "zero binding must survive rejection");
}

void positive_then_zero_rejects() {
  host::SymbolicSize size;
  size.verify(7, "shape", 0);
  expect_panic([&] { size.verify(0, "shape", 0); }, "expected 7 but got 0");
}

void setting_zero_twice_rejects() {
  host::SymbolicSize size;
  size.set_value(0);
  expect_panic([&] { size.set_value(0); }, "Size value already set");
  require(size.unwrap() == 0, "second setter must not mutate state");
}

void setting_positive_after_zero_rejects() {
  host::SymbolicSize size;
  size.set_value(0);
  expect_panic([&] { size.set_value(7); }, "Size value already set");
  require(size.unwrap() == 0, "second setter must not overwrite zero");
}

void signed_scalar_values_are_not_sentinels() {
  // Signed scalar bindings are not negative tensor-shape support.
  for (const auto value : {std::int64_t{-1}, std::numeric_limits<std::int64_t>::min()}) {
    host::SymbolicSize size;
    size.set_value(value);
    require(size.has_value() && size.unwrap() == value, "do not steal a signed value");
  }
}

void empty_pair_binds_zero() {
  Metadata left{0}, right{0};
  host::SymbolicSize length;
  host::TensorMatcher({length}).with_dtype<std::int32_t>().with_device<kDLCPU>()
      .verify(left.view()).verify(right.view());
  require(length.has_value() && length.unwrap() == 0, "empty pair must bind zero");
}

void pair_zero_positive_rejects() {
  Metadata left{0}, right{7};
  host::SymbolicSize length;
  expect_panic([&] {
    host::TensorMatcher({length}).verify(left.view()).verify(right.view());
  }, "expected 0 but got 7");
}

void pair_positive_zero_rejects() {
  Metadata left{7}, right{0};
  host::SymbolicSize length;
  expect_panic([&] {
    host::TensorMatcher({length}).verify(left.view()).verify(right.view());
  }, "expected 7 but got 0");
}

void literal_zero_is_not_a_wildcard() {
  Metadata empty{0}, nonempty{7};
  host::TensorMatcher({0}).verify(empty.view());
  expect_panic([&] { host::TensorMatcher({0}).verify(nonempty.view()); }, "Size mismatch");
  require(host::TensorMatcher({0}).debug_str() == "Tensor<0>", "literal zero diagnostic");
}

void empty_total_does_not_bypass_other_axes() {
  Metadata tensor{0};
  tensor.tensor.ndim = 2;
  tensor.shape[1] = 5;
  tensor.strides[0] = 5;
  expect_panic([&] { host::TensorMatcher({0, 4}).verify(tensor.view()); }, "expected 4 but got 5");
}

void literal_zero_stride_is_enforced() {
  Metadata broadcast{3, 0}, dense{3, 1};
  host::TensorMatcher({3}).with_strides({0}).verify(broadcast.view());
  expect_panic([&] {
    host::TensorMatcher({3}).with_strides({0}).verify(dense.view());
  }, "Size mismatch");
}

void shared_zero_stride_is_enforced() {
  Metadata broadcast{3, 0}, strided{3, 8};
  host::SymbolicSize stride;
  expect_panic([&] {
    host::TensorMatcher({3}).with_strides({stride})
        .verify(broadcast.view()).verify(strided.view());
  }, "expected 0 but got 8");
  require(stride.unwrap() == 0, "zero stride must not rebind");
}

void singleton_stride_exception_is_preserved() {
  Metadata first{1, 0}, second{1, 8};
  host::SymbolicSize stride;
  host::TensorMatcher({1}).with_strides({stride})
      .verify(first.view()).verify(second.view());
  require(stride.has_value() && stride.unwrap() == 0, "first singleton stride must bind");
  host::TensorMatcher({1}).with_strides({0}).verify(second.view());
  host::TensorMatcher({1}).with_strides({8}).verify(first.view());
}

void fresh_wildcards_accept_zero_and_positive() {
  Metadata empty{0}, nonempty{7};
  host::TensorMatcher({-1}).verify(empty.view());
  host::TensorMatcher({-1}).verify(nonempty.view());
}

void chained_wildcard_binds_once() {
  Metadata empty{0}, nonempty{7};
  expect_panic([&] {
    host::TensorMatcher({-1}).verify(empty.view()).verify(nonempty.view());
  }, "expected 0 but got 7");
}

void shared_dtype_still_requires_equality() {
  Metadata left{0}, right{0};
  right.tensor.dtype = DLDataType{kDLFloat, 32, 1};
  host::SymbolicDType dtype;
  expect_panic([&] {
    host::TensorMatcher({0}).with_dtype(dtype).verify(left.view());
    host::TensorMatcher({0}).with_dtype(dtype).verify(right.view());
  }, "DType mismatch");
}

void empty_dtype_allowlist_without_symbol_is_checked() {
  Metadata empty{0};
  empty.tensor.dtype = DLDataType{kDLFloat, 32, 1};
  expect_panic([&] {
    host::TensorMatcher({0}).with_dtype<std::int32_t>().verify(empty.view());
  }, "not in the allowed options");
}

void empty_device_allowlist_without_symbol_is_checked() {
  Metadata empty{0};
  // Rejected metadata only: this fake device tag must never reach a runtime or kernel.
  empty.tensor.device = DLDevice{kDLROCM, 0};
  expect_panic([&] {
    host::TensorMatcher({0}).with_device<kDLCPU>().verify(empty.view());
  }, "not in the allowed options");
}

void empty_rank_is_checked() {
  Metadata empty{0};
  expect_panic([&] { host::TensorMatcher({0, 1}).verify(empty.view()); }, "Tensor dimension mismatch");
}

void contiguous_default_is_preserved() {
  Metadata strided{3, 2};
  expect_panic([&] { host::TensorMatcher({3}).verify(strided.view()); }, "Tensor is not contiguous");
}

void separate_symbols_remain_independent() {
  host::SymbolicSize first, second;
  first.set_value(0);
  second.set_value(7);
  require(first.unwrap() == 0 && second.unwrap() == 7, "symbols must not share binding state");
}

using Test = std::pair<const char*, void (*)()>;
const Test cases[] = {
    {"unbound_rejects_unwrap", unbound_rejects_unwrap},
    {"zero_is_a_bound_value", zero_is_a_bound_value},
    {"repeated_zero_verify_is_allowed", repeated_zero_verify_is_allowed},
    {"equal_positive_verify_is_unchanged", equal_positive_verify_is_unchanged},
    {"unequal_positive_verify_rejects", unequal_positive_verify_rejects},
    {"zero_then_positive_rejects", zero_then_positive_rejects},
    {"positive_then_zero_rejects", positive_then_zero_rejects},
    {"setting_zero_twice_rejects", setting_zero_twice_rejects},
    {"setting_positive_after_zero_rejects", setting_positive_after_zero_rejects},
    {"signed_scalar_values_are_not_sentinels", signed_scalar_values_are_not_sentinels},
    {"empty_pair_binds_zero", empty_pair_binds_zero},
    {"pair_zero_positive_rejects", pair_zero_positive_rejects},
    {"pair_positive_zero_rejects", pair_positive_zero_rejects},
    {"literal_zero_is_not_a_wildcard", literal_zero_is_not_a_wildcard},
    {"empty_total_does_not_bypass_other_axes", empty_total_does_not_bypass_other_axes},
    {"literal_zero_stride_is_enforced", literal_zero_stride_is_enforced},
    {"shared_zero_stride_is_enforced", shared_zero_stride_is_enforced},
    {"singleton_stride_exception_is_preserved", singleton_stride_exception_is_preserved},
    {"fresh_wildcards_accept_zero_and_positive", fresh_wildcards_accept_zero_and_positive},
    {"chained_wildcard_binds_once", chained_wildcard_binds_once},
    {"shared_dtype_still_requires_equality", shared_dtype_still_requires_equality},
    {"empty_dtype_allowlist_without_symbol_is_checked", empty_dtype_allowlist_without_symbol_is_checked},
    {"empty_device_allowlist_without_symbol_is_checked", empty_device_allowlist_without_symbol_is_checked},
    {"empty_rank_is_checked", empty_rank_is_checked},
    {"contiguous_default_is_preserved", contiguous_default_is_preserved},
    {"separate_symbols_remain_independent", separate_symbols_remain_independent},
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
