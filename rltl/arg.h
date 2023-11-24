#pragma once
#include <utility>

#define RLTL_ARG(T, name)					                            \
 public:                                                                \
  inline auto name(const T& new_##name)->decltype(*this) { /* NOLINT */ \
    this->m_##name = new_##name;										\
    return *this;                                                       \
  }                                                                     \
  inline auto name(T&& new_##name)->decltype(*this) { /* NOLINT */      \
    this->m_##name = std::move(new_##name);                             \
    return *this;                                                       \
  }                                                                     \
  inline const T& name() const noexcept { /* NOLINT */                  \
    return this->m_##name;                                              \
  }                                                                     \
  inline T& name() noexcept { /* NOLINT */                              \
    return this->m_##name;                                              \
  }                                                                     \
 protected:                                                             \
  T m_##name /* NOLINT */

