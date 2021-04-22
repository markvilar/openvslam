cmake_minimum_required(VERSION 3.1)
project(openvslam LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 OLD)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# -----------------------------------------------------------------------------
# ---- Build options ----------------------------------------------------------
# -----------------------------------------------------------------------------

option(BUILD_APPLICATIONS   "Build applications."      ON)
option(BUILD_EXAMPLES       "Build examples."          OFF)
option(BUILD_TESTS          "Build tests."             OFF)
option(USE_PANGOLIN_VIEWER  "Enable Pangolin Viewer."  ON)
option(USE_SOCKET_PUBLISHER "Enable Socket Publisher." OFF)

# -----------------------------------------------------------------------------
# ---- Set build type ---------------------------------------------------------
# -----------------------------------------------------------------------------

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" 
		"MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# -----------------------------------------------------------------------------
# ---- Git --------------------------------------------------------------------
# -----------------------------------------------------------------------------

find_package(Git QUIET REQUIRED)

if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init 
			--recursive WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} 
			RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR CONCAT "git submodule update --init failed "
				"with ${GIT_SUBMOD_RESULT}, please checkout submodules.")
        endif()
    endif()
endif()

# -----------------------------------------------------------------------------
# ---- Set options for debugging ----------------------------------------------
# -----------------------------------------------------------------------------

set(USE_CCACHE ON CACHE BOOL "Use ccache to accelerate build")
find_program(CCACHE_EXE ccache)
if(USE_CCACHE AND CCACHE_EXE)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_EXE}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_EXE}")
    message(STATUS "ccache: Enabled")
else()
    set(USE_CCACHE OFF)
    message(STATUS "ccache: Disabled")
endif()

set(USE_SANITIZER OFF CACHE BOOL 
	"Enable Address/Memory sanitizer (set env as ASAN_OPTIONS=detect_leaks=1)")
if(USE_SANITIZER)
    string(JOIN " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" 
		"-fno-omit-frame-pointer" "-fsanitize=address")
    string(JOIN " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" 
		"-fno-omit-frame-pointer" "-fsanitize=address")
    string(JOIN " " CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" 
		"-fno-omit-frame-pointer" "-fsanitize=address")
    string(JOIN " " CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" 
		"-fno-omit-frame-pointer" "-fsanitize=address")
    message(STATUS "Address/Memory sanitizer: ENABLED")
else()
    message(STATUS "Address/Memory sanitizer: DISABLED")
endif()

# -----------------------------------------------------------------------------
# ---- Set compiler options ---------------------------------------------------
# -----------------------------------------------------------------------------

# Generic flags.
string(JOIN " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" "-Wall" "-Wextra")
string(JOIN " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" "-Wall" "-Wextra")

# Debug flags.
string(JOIN " " CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}" "-Og")
string(JOIN " " CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" "-Og")

# Release flags.
string(JOIN " " CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}" 
	"-O3 -ffast-math")
string(JOIN " " CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" 
	"-O3 -ffast-math")

set(BUILD_WITH_MARCH_NATIVE OFF CACHE BOOL 
	"Enable architecture-aware optimization (-march=native)")
if(BUILD_WITH_MARCH_NATIVE)
    string(JOIN " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" "-mtune=native" 
		"-march=native")
    string(JOIN " " CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}" "-mtune=native" 
		"-march=native")
    message(STATUS "Architecture-aware optimization (-march=native): ENABLED")
else()
    string(JOIN " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" "-mtune=native")
    string(JOIN " " CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}" "-mtune=native")
    message(STATUS "Architecture-aware optimization (-march=native): DISABLED")
endif()

# -----------------------------------------------------------------------------
# ---- Find dependencies ------------------------------------------------------
# -----------------------------------------------------------------------------

# Threads
find_package(Threads REQUIRED)

# Eigen
find_package(Eigen3 3.3 REQUIRED)

# yaml-cpp
find_package(yaml-cpp REQUIRED)

# OpenCV
find_package(OpenCV 3.3.1 QUIET COMPONENTS core imgcodecs videoio features2d 
	calib3d highgui)
if(NOT OpenCV_FOUND)
	find_package(OpenCV 4.0 REQUIRED COMPONENTS core imgcodecs videoio 
		features2d calib3d highgui)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV >= 3.3.1 not found")
    endif()
endif()
message(STATUS "Use OpenCV ${OpenCV_VERSION}")

# -----------------------------------------------------------------------------
# ---- Install configuration --------------------------------------------------
# -----------------------------------------------------------------------------

include(CMakePackageConfigHelpers)

# Generate cmake configuration scripts
set(OPENVSLAM_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(OPENVSLAM_VERSION_CONFIG 
	${OPENVSLAM_GENERATED_DIR}/${PROJECT_NAME}ConfigVersion.cmake)
set(OPENVSLAM_PROJECT_CONFIG 
	${OPENVSLAM_GENERATED_DIR}/${PROJECT_NAME}Config.cmake)
set(OPENVSLAM_TARGETS_EXPORT_NAME ${PROJECT_NAME}Targets)
set(OPENVSLAM_CONFIG_INSTALL_DIR lib/cmake/${PROJECT_NAME})
set(OPENVSLAM_NAMESPACE "${PROJECT_NAME}::")
set(OPENVSLAM_VERSION 0.0.1)

# Create a version config file
write_basic_package_version_file(${OPENVSLAM_VERSION_CONFIG}
	VERSION ${OPENVSLAM_VERSION}
	COMPATIBILITY SameMajorVersion)

# Create a project config file
configure_file(${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in 
	${OPENVSLAM_PROJECT_CONFIG} @ONLY)

# Install to the specified directory
install(FILES ${OPENVSLAM_VERSION_CONFIG} ${OPENVSLAM_PROJECT_CONFIG}
	DESTINATION ${OPENVSLAM_CONFIG_INSTALL_DIR})
install(EXPORT ${OPENVSLAM_TARGETS_EXPORT_NAME}
	NAMESPACE ${OPENVSLAM_NAMESPACE}
	DESTINATION ${OPENVSLAM_CONFIG_INSTALL_DIR})

# Set standard installation directories
set(RUNTIME_DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
set(LIBRARY_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(ARCHIVE_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(INCLUDES_DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

# -----------------------------------------------------------------------------
# ---- Build selection --------------------------------------------------------
# -----------------------------------------------------------------------------

# Third-party.
#add_subdirectory(3rd)

# Library.
add_subdirectory(src)

if(BUILD_APPLICATIONS)
	message(STATUS "Building applications.")
	add_subdirectory(apps)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
