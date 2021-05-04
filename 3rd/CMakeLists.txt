if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/g2o/CMakeLists.txt")
    	message(FATAL_ERROR "Sennet submodule g2o was not downloaded! "
		"GIT_SUBMODULE was turned off or failed. Please update submodules and "
		"try again.")
endif()

add_subdirectory(g2o)
