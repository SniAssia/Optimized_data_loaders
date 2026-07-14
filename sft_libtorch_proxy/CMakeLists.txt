cmake_minimum_required(VERSION 3.18)
project(sft_dataloader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

# ── original custom pipeline ──────────────────────────────────
add_executable(dataloader_demo main_libtorch.cpp)
target_include_directories(dataloader_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dataloader_demo PRIVATE "${TORCH_LIBRARIES}")
target_compile_options(dataloader_demo PRIVATE ${TORCH_CXX_FLAGS})

# ── new LibTorch pipeline with full benchmark ─────────────────
add_executable(dataloader_libtorch main_libtorch.cpp)
target_include_directories(dataloader_libtorch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dataloader_libtorch PRIVATE "${TORCH_LIBRARIES}")
target_compile_options(dataloader_libtorch PRIVATE ${TORCH_CXX_FLAGS})
