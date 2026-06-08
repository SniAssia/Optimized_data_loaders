cmake_minimum_required(VERSION 3.18)
project(cpp_dataloader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── LibTorch ──────────────────────────────────────────────────────────────
# Point CMAKE_PREFIX_PATH to your LibTorch installation, e.g.:
#   cmake -B build -DCMAKE_PREFIX_PATH=/opt/libtorch
find_package(Torch REQUIRED)

# ── Target ────────────────────────────────────────────────────────────────
# FIX: all files are in the project root (no src/ or include/ subdirs)
add_executable(dataloader_test main.cpp)

target_include_directories(dataloader_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(dataloader_test PRIVATE
    "${TORCH_LIBRARIES}"
)

# LibTorch requires C++17 ABI; propagate its compile flags
target_compile_options(dataloader_test PRIVATE
    ${TORCH_CXX_FLAGS}
)

# ── Optional: distributed support ─────────────────────────────────────────
# Uncomment if your LibTorch was built with USE_DISTRIBUTED=1:
# target_compile_definitions(dataloader_test PRIVATE USE_DISTRIBUTED)
