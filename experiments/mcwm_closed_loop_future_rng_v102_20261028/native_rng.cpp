// SUMO 1.25.0 only. Bind the resident library's public C++ lane-RNG loader.
// No SUMO object layout, simulation reload, or RNG pointer is used here.
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI != 0
#error "The resident SUMO 1.25.0 exports the old std::string ABI."
#endif

namespace {
constexpr unsigned long long burn = 1000000;
constexpr const char* loader_symbol = "_ZN6MSLane12loadRNGStateEiRKSs";
using LoadLaneRNG = void (*)(int, const std::string&);
}

extern "C" int mcwm_apply_lane_rng(const char* library, int lane_count,
                                   int identity, std::uint32_t seed,
                                   char* error, std::size_t error_size) {
    void* handle = nullptr;
    try {
        if (lane_count <= 0 || (identity != 0 && identity != 1)) {
            throw std::runtime_error("Invalid lane count or identity mode");
        }
        if (!identity && seed > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max() - (lane_count - 1))) {
            throw std::runtime_error("Future seed plus lane index exceeds SUMO int seed range");
        }
        // RTLD_NOLOAD prevents a second SUMO instance from being loaded here.
        handle = dlopen(library, RTLD_NOW | RTLD_NOLOAD);
        if (handle == nullptr) {
            throw std::runtime_error(dlerror());
        }
        dlerror();
        auto load = reinterpret_cast<LoadLaneRNG>(dlsym(handle, loader_symbol));
        if (const char* message = dlerror()) {
            throw std::runtime_error(message);
        }
        if (!identity) {
            for (int index = 0; index < lane_count; ++index) {
                std::mt19937 engine(seed + static_cast<std::uint32_t>(index));
                engine.discard(burn);
                std::ostringstream state;
                // SUMO reads a full engine only when count >= SAVE_ONLY_COUNT.
                // This count is real: these draws were discarded above.
                state << burn << ' ' << engine;
                load(index, state.str());
            }
        }
        // F0 is identity: fork already retained every original engine and count.
        dlclose(handle);
        if (error_size > 0) error[0] = '\0';
        return 0;
    } catch (const std::exception& exception) {
        if (handle != nullptr) dlclose(handle);
        if (error_size > 0) std::snprintf(error, error_size, "%s", exception.what());
        return 1;
    }
}
