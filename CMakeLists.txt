cmake_minimum_required(VERSION 3.13)
project(rltl)

set(IDLCPP_PATH "C:/idlcpp/bin/idlcpp.exe" CACHE FILEPATH "idlcpp path")
set(PAFCORE_DIR "C:/paf/pafcore" CACHE PATH "pafcore dir") 

if(NOT EXISTS ${IDLCPP_PATH})
	message("can not find file: " ${IDLCPP_PATH})
	return()
endif()

set(PAFCORE_KEY_FILE $CACHE{PAFCORE_DIR}/Variant.h)
if(NOT EXISTS ${PAFCORE_KEY_FILE})
	message("invalid entry PAFCORE_DIR : " ${PAFCORE_DIR})
	message("can not find file: " ${PAFCORE_KEY_FILE})
	return()
endif()

set(CMAKE_PREFIX_PATH "G:/libtorch_cpu_release/libtorch")

find_package(Torch REQUIRED)

#set(OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../bin)
set(DEBUG_OUTPUT_DIRECTORY G:/libtorch_cu118_debug/libtorch/lib)
set(RELEASE_OUTPUT_DIRECTORY G:/libtorch_cu118_release/libtorch/lib)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${DEBUG_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${DEBUG_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${DEBUG_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${RELEASE_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${RELEASE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${RELEASE_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${RELEASE_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${RELEASE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${RELEASE_OUTPUT_DIRECTORY})


set(CMAKE_DEBUG_POSTFIX _d)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(rltl)
add_subdirectory(test)


