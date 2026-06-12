cmake_minimum_required(VERSION 3.18)
project(unified_dataloader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point this at your libtorch installation, e.g.:
#   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)

# Optional: build with multi-GPU / NCCL distributed support
option(USE_DISTRIBUTED "Enable torch::distributed (NCCL) support" OFF)

add_executable(dataloader_demo main.cpp)

target_link_libraries(dataloader_demo "${TORCH_LIBRARIES}")
target_compile_options(dataloader_demo PRIVATE ${TORCH_CXX_FLAGS})

if(USE_DISTRIBUTED)
    target_compile_definitions(dataloader_demo PRIVATE USE_DISTRIBUTED)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dataloader_demo PRIVATE -Wall -Wextra -O2)
endif()

# Threads (required by prefetcher.h)
find_package(Threads REQUIRED)
target_link_libraries(dataloader_demo Threads::Threads)
