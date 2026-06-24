cmake_minimum_required(VERSION 3.18)
project(sharded_dataloader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── LibTorch ──────────────────────────────────────────────────
# cmake -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch REQUIRED)

# ── Target ────────────────────────────────────────────────────
add_executable(dataloader_demo main.cpp)

target_include_directories(dataloader_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(dataloader_demo PRIVATE
    "${TORCH_LIBRARIES}"
)

target_compile_options(dataloader_demo PRIVATE
    ${TORCH_CXX_FLAGS}
)

# ── Optional distributed support ──────────────────────────────
# Uncomment if LibTorch was built with USE_DISTRIBUTED=1:
# target_compile_definitions(dataloader_demo PRIVATE USE_DISTRIBUTED)
