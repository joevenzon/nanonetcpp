cmake_minimum_required(VERSION 3.16)
project(nanonetcpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default build type (for single-config generators: Ninja, Makefiles, etc.)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(tests src/tests.cpp)
target_include_directories(tests PRIVATE src)
set_target_properties(tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Per-config compile options
target_compile_options(tests PRIVATE
  $<$<CONFIG:Debug>:/Od /Zi>
  $<$<CONFIG:Release>:/O2 /Ob2 /Zi>
)

# Per-config linker options
target_link_options(tests PRIVATE
  $<$<CONFIG:Release>:/debug>
)

# Per-config compile definitions
target_compile_definitions(tests PRIVATE
  $<$<CONFIG:Debug>:DEBUG>
  $<$<CONFIG:Release>:NDEBUG>
)

add_executable(example example/example.cpp)
target_include_directories(example PRIVATE src)
set_target_properties(example PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/example"
)

# Per-config compile options
target_compile_options(example PRIVATE
  $<$<CONFIG:Debug>:/Od /Zi>
  $<$<CONFIG:Release>:/O2 /Ob2 /arch:AVX2 /Zi>
)

# Per-config linker options
target_link_options(example PRIVATE
  $<$<CONFIG:Release>:/debug>
)

# Per-config compile definitions
target_compile_definitions(example PRIVATE
  $<$<CONFIG:Debug>:DEBUG>
  $<$<CONFIG:Release>:NDEBUG>
)
