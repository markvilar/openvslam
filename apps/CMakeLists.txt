# -----------------------------------------------------------------------------
# ---- Definitions ------------------------------------------------------------
# -----------------------------------------------------------------------------

if(USE_PANGOLIN_VIEWER)
    message(STATUS "Viewer for examples: PangolinViewer")
else()
    message(STATUS "Viewer for examples: None")
endif()

set(ZED_CXX_FLAGS "")

# -----------------------------------------------------------------------------
# ---- Dependencies -----------------------------------------------------------
# -----------------------------------------------------------------------------

# Popl
set(popl_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rd/popl/include)

# Stack trace logger
set(USE_STACK_TRACE_LOGGER OFF CACHE BOOL 
	"Enable automatic stack trace logger of google-glog")

if(USE_STACK_TRACE_LOGGER)
    find_package(Glog REQUIRED)
    include_directories(${GLOG_INCLUDE_DIR})
    message(STATUS "Stack trace logger: ENABLED")
else()
    message(STATUS "Stack trace logger: DISABLED")
endif()

# Perftools
set(USE_GOOGLE_PERFTOOLS OFF CACHE BOOL "Enable profiler of google-perftools")
if(USE_GOOGLE_PERFTOOLS)
    find_package(Gperftools REQUIRED)
    include_directories(${GPERFTOOLS_INCLUDE_DIRS})
    message(STATUS "Google Perftools: ENABLED")
else()
    message(STATUS "Google Perftools: DISABLED")
endif()

# ZED
option(ZED_LINK_SHARED "Link with the ZED SDK shared executable" ON)
find_package(ZED 3 REQUIRED)

if (ZED_FOUND)
	message(STATUS "Found ZED SDK ${ZED_VERSION} at ${ZED_DIR}. "
		"ZED CUDA version: ${ZED_CUDA_VERSION}")
endif()

if(NOT ZED_LINK_SHARED AND MSVC)
	message(FATAL_ERROR "ZED_LINK_SHARED OFF : ZED SDK static libraries "
	"not available on Windows")
endif()

# Cuda
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)

# ZED libraries - static or dynamic linking
if(ZED_LINK_SHARED)
    	SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} 
		${CUDA_CUDART_LIBRARY})
else()
    	SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} 
		${CUDA_LIBRARY})
endif()

# -----------------------------------------------------------------------------
# ---- Executable -------------------------------------------------------------
# -----------------------------------------------------------------------------

add_executable(ZEDNativeSlam ZEDNativeSlam.cpp)
target_include_directories(ZEDNativeSlam PUBLIC 
	${CUDA_INCLUDE_DIRS}
	${ZED_INCLUDE_DIRS}
	${popl_INCLUDE_DIR})
target_compile_features(ZEDNativeSlam PRIVATE cxx_std_17 ${ZED_CXX_FLAGS})
target_link_libraries(ZEDNativeSlam PRIVATE 
	${PROJECT_NAME} 
	${ZED_LIBS} 
	${CUDA_LIBRARIES}
	opencv_imgcodecs
	opencv_videoio)

if(USE_PANGOLIN_VIEWER)
        target_compile_definitions(ZEDNativeSlam PRIVATE USE_PANGOLIN_VIEWER)
        target_link_libraries(ZEDNativeSlam PRIVATE pangolin_viewer)
endif()

add_executable(ZEDStereoSlam ZEDStereoSlam.cpp)
target_include_directories(ZEDStereoSlam PUBLIC 
	${CUDA_INCLUDE_DIRS}
	${ZED_INCLUDE_DIRS}
	${popl_INCLUDE_DIR})
target_compile_features(ZEDStereoSlam PRIVATE cxx_std_17 ${ZED_CXX_FLAGS})
target_link_libraries(ZEDStereoSlam PRIVATE 
	${PROJECT_NAME} 
	${ZED_LIBS} 
	${CUDA_LIBRARIES}
	opencv_imgcodecs
	opencv_videoio)

if(USE_PANGOLIN_VIEWER)
        target_compile_definitions(ZEDStereoSlam PRIVATE USE_PANGOLIN_VIEWER)
        target_link_libraries(ZEDStereoSlam PRIVATE pangolin_viewer)
endif()

