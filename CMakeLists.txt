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

# Helper function to configure an example target
function(add_example target_name source_file)
  add_executable(${target_name} ${source_file})
  target_include_directories(${target_name} PRIVATE src)
  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/example"
  )
  target_compile_options(${target_name} PRIVATE
    $<$<CONFIG:Debug>:/Od /Zi>
    $<$<CONFIG:Release>:/O2 /arch:AVX2 /Zi>
  )
  target_link_options(${target_name} PRIVATE
    $<$<CONFIG:Release>:/debug>
  )
  target_compile_definitions(${target_name} PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
  )
endfunction()

# Automatically discover all example sources
file(GLOB EXAMPLE_SOURCES "example/*.cpp")

foreach(ex_src IN LISTS EXAMPLE_SOURCES)
  # Derive target name from filename (e.g. "example-sine.cpp" -> "example-sine")
  get_filename_component(ex_name "${ex_src}" NAME_WE)
  add_example(${ex_name} "${ex_src}")
endforeach()
