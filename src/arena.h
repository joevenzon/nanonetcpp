#pragma once

#include <vector>
#include <cassert>
#include <span>

// =============================================================================
// Arena<T>  —  bump (linear) allocator
//
// Allocations are O(1). Memory is never freed individually; the entire arena
// is reclaimed with reset(). Intended for per-batch scratch allocations.
//
// Typical usage for tensor data/grad buffers:
//
//   Arena<float> activation_arena;
//   activation_arena.init(1 << 20);
//
//   // forward pass:
//   float* buf = activation_arena.allocate(n_elements);
//   // ... write into buf ...
//
//   // start of next iteration:
//   activation_arena.reset();
// =============================================================================

template <typename T>
class Arena
{
public:
    Arena() = default;
    explicit Arena(size_t capacity) { resize(capacity); }

    void resize(size_t capacity)
    {
        buffer.resize(capacity);
        used = 0;
        high_water = 0;
    }

    // Allocate n contiguous elements.
    // Does NOT zero-initialize.
    std::span <T> allocate(size_t n)
    {
        assert(used + n <= buffer.size() && "Arena capacity exceeded; increase init() size");
        T * ptr = buffer.data() + used;
        used += n;
        return std::span(ptr, buffer.data() + used);
    }

    // Get the current allocation cursor
    size_t cursor() const { return used; }

    // Reset to a saved cursor position (e.g. to undo a tentative allocation).
    void reset_to(size_t saved_cursor)
    {
        assert(saved_cursor <= used);
        high_water = std::max(high_water, used);
        used = saved_cursor;
    }

    // Reclaim all memory in the arena.
    void reset()
    {
        reset_to(0);
    }

    size_t size()           const { return used; }
    size_t capacity()       const { return buffer.size(); }
    size_t high_water_mark() const { return high_water; }

    T & operator[](size_t i) { return buffer[i]; }
    const T & operator[](size_t i) const { return buffer[i]; }

    std::span<T>       span() { return { buffer.data(), used }; }
    std::span<const T> span() const { return { buffer.data(), used }; }

private:
    std::vector<T> buffer;
    size_t used{ 0 };
    size_t high_water{ 0 };
};
