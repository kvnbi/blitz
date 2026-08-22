#include "misc.h"
#include <cstdlib>
#include <cstring>
#include <sstream>

#if defined(__APPLE__)
    #include <sys/mman.h>
#elif defined(__linux__)
    #include <sys/mman.h>
#endif

namespace blitz {

void* aligned_large_pages_alloc(size_t size) {
    constexpr size_t Alignment = 4096;
    size = (size + Alignment - 1) / Alignment * Alignment;

    void* mem = nullptr;
    if (posix_memalign(&mem, Alignment, size) != 0) return nullptr;

#if defined(__linux__)

    madvise(mem, size, MADV_HUGEPAGE);
#endif
    return mem;
}

void aligned_large_pages_free(void* ptr) { std::free(ptr); }

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::istringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
        if (!item.empty()) out.push_back(item);
    return out;
}

std::string engine_info() { return "Blitz 1.0"; }

}
