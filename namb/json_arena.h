#pragma once
// A bump arena and an nlohmann allocator that draws from it, so a .nam document
// can be parsed somewhere other than the general heap.
//
// Why this exists: converting a model means holding a whole JSON DOM at once,
// and a SlimmableContainer runs to hundreds of KiB of text. On the target that
// is far more than the newlib heap is sized for, and fragmenting the heap the
// audio path allocates from -- once per import, with a 6-figure allocation
// count -- is not something to do casually. A bump arena parked in its own
// region sidesteps both: the DOM is built, read once, and thrown away by moving
// a single pointer.
//
//   static uint8_t store[kBytes];          // wherever the platform wants it
//   nam::json_arena::Arena arena(store, sizeof(store));
//   nam::json_arena::ScopedBind bind(arena);
//
//   auto doc = nam::json_arena::arena_json::parse(first, last, nullptr, false);
//   if (doc.is_discarded() || arena.Exhausted()) { ...reject... }
//   ...read what you need out of doc...
//   // doc dies, arena.Reset() reclaims everything in one store
//
// The arena does NOT own its memory. The platform supplies a span, which is
// what lets the firmware place the store by linker section (internal RAM today,
// RAM_EXT once the HyperRAM parts are reworked) while the host tools hand over
// an ordinary static buffer, with no #ifdefs in this file.
//
// ---------------------------------------------------------------------------
// Sizing
// ---------------------------------------------------------------------------
// deallocate() is a no-op, which is the point -- but it means a std::vector
// that grows geometrically leaves each of its old buffers behind. nlohmann
// builds arrays by push_back, so the weights array of an A2 model costs roughly
// TWICE its final size before the parse ends: the live buffer, the previous one
// still uncopied, and the abandoned smaller ones. Do not size the arena from
// sizeof(DOM); size it from a measured HighWater() and leave margin.
//
// Note also that this redirects the DOM itself -- the nodes, the object maps,
// the arrays -- but not std::string. Keys and string values still come from the
// global heap, because nlohmann bakes std::allocator into the default
// StringType. They are small and mostly inside the short-string optimisation;
// swap StringType if a measurement ever says otherwise.

#include <cstddef>
#include <cstdint>
#include <new>

#include <json.hpp>

namespace nam
{
namespace json_arena
{

// =============================================================================
// Arena
// =============================================================================

/// \brief A bump allocator over a caller-supplied span.
///
/// Allocation is a pointer add; release is all-or-nothing through Reset(). No
/// locking: a conversion runs in one foreground context, and adding a mutex
/// would imply a concurrency this is not designed for.
class Arena
{
public:
  Arena() = default;

  Arena(uint8_t* base, size_t size) { Init(base, size); }

  /// \brief Bind the arena to \p size bytes at \p base and discard any contents.
  void Init(uint8_t* base, size_t size)
  {
    _base = base;
    _size = (base == nullptr) ? 0 : size;
    _used = 0;
    _high_water = 0;
    _exhausted = false;
    _spilled = 0;
  }

  /// \brief Hand back every byte at once. Any object still living in the arena
  ///        dangles immediately, so destroy or abandon the DOM first.
  ///
  /// Deliberately does not clear \p _high_water: the whole reason to record it
  /// is to size the arena across a run of conversions, and a reset per
  /// conversion would throw that away. Use ResetStatistics() to zero it.
  void Reset()
  {
    _used = 0;
    _exhausted = false;
    _spilled = 0;
  }

  /// \brief Clear the high-water mark and spill accounting.
  void ResetStatistics()
  {
    _high_water = _used;
    _spilled = 0;
  }

  /// \brief Carve \p bytes at \p align, or nullptr if the arena cannot.
  ///
  /// A null return latches Exhausted(); the caller (ArenaAllocator) then falls
  /// back to the global heap so the parse still completes and can be rejected
  /// in one place, rather than faulting halfway through a document.
  void* Allocate(size_t bytes, size_t align)
  {
    if (_base == nullptr || bytes == 0)
    {
      if (bytes != 0)
        _exhausted = true;
      return nullptr;
    }

    // Round up the ADDRESS, not the offset. Aligning the offset only works if
    // the span itself is aligned, and a caller is free to hand over a byte
    // array with no alignment at all. align is a power of two here -- it comes
    // from alignof(T).
    const uintptr_t mask = static_cast<uintptr_t>(align) - 1u;
    const uintptr_t base = reinterpret_cast<uintptr_t>(_base);
    const uintptr_t aligned_addr = (base + _used + mask) & ~mask;
    const size_t aligned = static_cast<size_t>(aligned_addr - base);

    // Checked in this order so neither term can wrap: aligned cannot exceed
    // _size without the first test catching it, and only then is the length
    // added to it.
    if (aligned > _size || bytes > _size - aligned)
    {
      _exhausted = true;
      return nullptr;
    }

    _used = aligned + bytes;
    if (_used > _high_water)
      _high_water = _used;

    return _base + aligned;
  }

  /// \brief True if \p p came out of this arena, so Deallocate can tell an
  ///        arena block from a heap block without a header on every allocation.
  bool Owns(const void* p) const
  {
    const uint8_t* q = static_cast<const uint8_t*>(p);
    return (_base != nullptr) && (q >= _base) && (q < _base + _size);
  }

  /// \brief Record that \p bytes had to come from the heap instead.
  void NoteSpill(size_t bytes)
  {
    _exhausted = true;
    _spilled += bytes;
  }

  /// \brief Bytes handed out since the last Reset().
  size_t Used() const { return _used; }

  /// \brief Bytes the arena was given.
  size_t Capacity() const { return _size; }

  /// \brief The largest Used() ever reached. What to size the arena from.
  size_t HighWater() const { return _high_water; }

  /// \brief Bytes that spilled to the global heap after the arena filled up.
  size_t Spilled() const { return _spilled; }

  /// \brief Sticky: the arena ran out at some point since the last Reset().
  ///
  /// A DOM parsed while this was set is still structurally correct -- the
  /// overflow went to the heap -- but it means the arena is undersized, and a
  /// converter should say so rather than quietly depend on the heap having had
  /// room.
  bool Exhausted() const { return _exhausted; }

private:
  uint8_t* _base = nullptr;
  size_t _size = 0;
  size_t _used = 0;
  size_t _high_water = 0;
  size_t _spilled = 0;
  bool _exhausted = false;
};

// =============================================================================
// The bound arena
// =============================================================================
//
// nlohmann default-constructs its allocator every time it creates a node
// (basic_json::create<T>() makes an AllocatorType<T> on the stack), so there is
// no way to hand an allocator instance down through the parse. The allocator
// therefore has to be stateless and find its arena through a global. That is
// acceptable precisely because conversion is a single-threaded, one-at-a-time
// operation; ScopedBind keeps the window explicit.

/// \brief The arena ArenaAllocator draws from, or nullptr for the global heap.
inline Arena*& bound_arena()
{
  static Arena* current = nullptr;
  return current;
}

/// \brief Install \p a for the lifetime of this object, restoring the previous
///        binding on the way out.
class ScopedBind
{
public:
  explicit ScopedBind(Arena& a)
  : _previous(bound_arena())
  {
    bound_arena() = &a;
  }

  ~ScopedBind() { bound_arena() = _previous; }

  ScopedBind(const ScopedBind&) = delete;
  ScopedBind& operator=(const ScopedBind&) = delete;

private:
  Arena* _previous;
};

// =============================================================================
// ArenaAllocator
// =============================================================================

/// \brief Standard-library allocator that carves from the bound arena.
///
/// Stateless, so rebinding and copying are free and every instance compares
/// equal -- which is what lets a container allocated under one instance be
/// freed through another, as the containers inside a DOM routinely do.
///
/// Falls back to the global heap when no arena is bound or the arena is full,
/// and remembers that it had to. Deallocate distinguishes the two by address
/// range, so no per-allocation header is needed.
template<class T>
class ArenaAllocator
{
public:
  using value_type = T;

  ArenaAllocator() noexcept = default;

  template<class U>
  ArenaAllocator(const ArenaAllocator<U>&) noexcept
  {
  }

  T* allocate(size_t n)
  {
    const size_t bytes = n * sizeof(T);
    Arena* a = bound_arena();

    if (a != nullptr)
    {
      // alignof(T) can be under the pointer alignment for small T; keeping the
      // cursor at least pointer-aligned costs nothing and avoids handing a
      // misaligned block to anything that later placement-news into it.
      const size_t align = alignof(T) > alignof(void*) ? alignof(T) : alignof(void*);
      if (void* p = a->Allocate(bytes, align))
        return static_cast<T*>(p);

      // Exhausted, and there is deliberately no fall back to the heap.
      //
      // An earlier version spilled here, on the theory that the parse could
      // finish and be rejected afterwards. That theory does not survive
      // -fno-exceptions: operator new cannot report failure, it calls abort().
      // So spilling a DOM the arena could not hold does not produce a
      // rejectable result, it empties whatever heap the rest of the firmware
      // shares and then kills the process from inside the allocator.
      //
      // Returning null instead faults at the point of use, which is at least
      // local, diagnosable, and leaves the heap intact. Neither outcome is
      // acceptable in the field, so the arena is sized past the worst case and
      // the caller pre-flights against Capacity() rather than relying on this.
      a->NoteSpill(bytes);
      return nullptr;
    }

    // No arena bound: the host tools use this type without one.
    return static_cast<T*>(::operator new(bytes));
  }

  void deallocate(T* p, size_t) noexcept
  {
    if (p == nullptr)
      return;

    // An arena block is released by Reset(), never individually -- that is the
    // whole bargain. Only spilled blocks go back to the heap.
    Arena* a = bound_arena();
    if (a != nullptr && a->Owns(p))
      return;

    ::operator delete(p);
  }

  template<class U>
  bool operator==(const ArenaAllocator<U>&) const noexcept
  {
    return true;
  }

  template<class U>
  bool operator!=(const ArenaAllocator<U>&) const noexcept
  {
    return false;
  }
};

// =============================================================================
// The JSON type
// =============================================================================

/// \brief nlohmann's DOM with its nodes, object maps and arrays drawn from the
///        bound arena. Interface-compatible with nlohmann::json; the only
///        difference is where the memory comes from.
///
/// Parse it with allow_exceptions = false and check is_discarded(): the
/// firmware builds with -fno-exceptions, under which nlohmann's JSON_THROW
/// becomes std::abort().
using arena_json = nlohmann::basic_json<std::map, std::vector, std::string, bool, std::int64_t, std::uint64_t, double,
                                        ArenaAllocator>;

} // namespace json_arena
} // namespace nam
