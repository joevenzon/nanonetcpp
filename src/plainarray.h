#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <algorithm>

// A minimal, in-place "dynamic" array backed by a fixed-size std::array.
// - No heap allocation; capacity is a compile-time constant N.
// - No exceptions; contract violations are asserted.
// - Trivially copyable/movable when T is POD.
template <typename T, std::size_t N>
class PlainArray
{
public:
    static_assert(__is_trivially_copyable(T), "T must be trivially copyable (POD-like)");

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------
    PlainArray() noexcept = default;

    PlainArray(std::initializer_list<T> list) noexcept
    {
        assert(list.size() <= N && "Initializer list exceeds capacity");
        std::copy(list.begin(), list.end(), buffer.begin());
        count = list.size();
    }

    // -----------------------------------------------------------------------
    // Size & Capacity
    // -----------------------------------------------------------------------
    const std::size_t size() const { return count; }
    static constexpr std::size_t capacity() noexcept { return N; }
    bool empty()    const noexcept { return count == 0; }
    bool full()     const noexcept { return count == N; }

    // -----------------------------------------------------------------------
    // Element access (bounds-checked via assert)
    // -----------------------------------------------------------------------
    T & operator[](std::size_t i) noexcept
    {
        assert(i < count && "index out of range");
        return buffer[i];
    }
    const T & operator[](std::size_t i) const noexcept
    {
        assert(i < count && "index out of range");
        return buffer[i];
    }

    T & front() noexcept { assert(!empty() && "front() on empty array"); return buffer[0]; }
    T & back()  noexcept { assert(!empty() && "back() on empty array");  return buffer[count - 1]; }

    const T & front() const noexcept { assert(!empty() && "front() on empty array"); return buffer[0]; }
    const T & back()  const noexcept { assert(!empty() && "back() on empty array");  return buffer[count - 1]; }

    // Raw pointer access (useful for memcpy / C APIs)
    T * data()       noexcept { return buffer.data(); }
    const T * data() const noexcept { return buffer.data(); }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    // Append one element at the end.
    void push_back(const T & value) noexcept
    {
        assert(!full() && "push_back on full array");
        buffer[count++] = value;
    }

    // Remove the last element.
    void pop_back() noexcept
    {
        assert(!empty() && "pop_back on empty array");
        --count;
    }

    // Resize: grows (does NOT initialize new slots) or shrinks (truncates). O(1).
    void resize(std::size_t new_size) noexcept
    {
        assert(new_size <= N && "resize exceeds capacity");
        count = new_size;
    }

    // Drop all elements without touching memory.
    void clear() noexcept { count = 0; }

    // -----------------------------------------------------------------------
    // Iterators (raw pointers — valid for range-for and STL algorithms)
    // -----------------------------------------------------------------------
    T * begin()       noexcept { return buffer.data(); }
    T * end()         noexcept { return buffer.data() + count; }
    const T * begin() const noexcept { return buffer.data(); }
    const T * end()   const noexcept { return buffer.data() + count; }

private:
    // -----------------------------------------------------------------------
    // Data members
    // -----------------------------------------------------------------------
    std::array<T, N> buffer{};
    std::size_t      count{ 0 };
};