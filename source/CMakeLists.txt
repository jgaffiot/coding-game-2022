# ------------------------------------------------------------------------------------ #
# Check if cmake has the required version
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)

# ------------------------------------------------------------------------------------ #
# Declaring the project
project(spider-attack)
set(REPOSITORY_DIR ${PROJECT_SOURCE_DIR}/../)

# ------------------------------------------------------------------------------------ #
# Set default C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)
# Set warnings
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-long-long -Wno-vla"
)

# ------------------------------------------------------------------------------------ #
# Offer the user the choice of overriding the installation directories
set(CMAKE_INSTALL_PREFIX
    ${CMAKE_BINARY_DIR}/../../install/
    CACHE PATH "Prefix for installation" FORCE
)
set(INSTALL_LIB_DIR
    lib
    CACHE PATH "Installation directory for libraries" FORCE
)
set(INSTALL_INCLUDE_DIR
    include
    CACHE PATH "Installation directory for header files" FORCE
)
set(INSTALL_CMAKE_DIR
    CMake
    CACHE PATH "Installation directory for CMake files" FORCE
)

# ------------------------------------------------------------------------------------ #
# Defining default install dir
set(CMAKE_INSTALL_LIBDIR lib/) # Search for GNUInstallDirs in CMake documentation
set(CMAKE_INSTALL_BINDIR bin/) # Search for GNUInstallDirs in CMake documentation

# ------------------------------------------------------------------------------------ #
# Defining global default install dirs
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/arch)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)

# ------------------------------------------------------------------------------------ #
# Make relative paths absolute(needed later on)
foreach(p_ LIB BIN CMAKE INCLUDE CMAKE)
    set(VAR INSTALL_${p_}_DIR)
    if(NOT IS_ABSOLUTE "${${VAR}}")
        set(${VAR} "${CMAKE_INSTALL_PREFIX}/${${VAR}}")
    endif()
endforeach()

# ------------------------------------------------------------------------------------ #
# Set the headers directory
include_directories(${PROJECT_SOURCE_DIR}/inc ${PROJECT_BINARY_DIR})

# ------------------------------------------------------------------------------------ #
# Target executable

# Generating file lists
file(GLOB SPIDER_SRC ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB SPIDER_INC ${PROJECT_SOURCE_DIR}/inc/*.hh)

add_executable(spider-attack  ${SPIDER_SRC} ${SPIDER_INC})
set_target_properties(
    spider-attack PROPERTIES PUBLIC_HEADER "${SPIDER_INC}"
)

# Install the executable
install(TARGETS spider-attack)
