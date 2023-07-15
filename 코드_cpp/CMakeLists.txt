cmake_minimum_required(VERSION 3.18)
project(cardfinder)

set(CMAKE_CXX_STANDARD 14)
set(pathInclude ${CMAKE_SOURCE_DIR}/include)

add_library(native-lib SHARED ${CMAKE_SOURCE_DIR}/native-lib.cpp)
add_library(card-finder SHARED ${CMAKE_SOURCE_DIR}/card-finder.cpp)


file(GLOB OPENCV_INCLUDE_DIRS "${pathInclude}/opencv")
file(GLOB PYTORCH_INCLUDE_DIRS "${pathInclude}/libtorch")
file(GLOB SOURCE_INCLUDE_DIRS "${pathInclude}/source")
file(GLOB LINK_DIRS "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}")


include_directories(
        ${OPENCV_INCLUDE_DIRS}
        ${PYTORCH_INCLUDE_DIRS}
        ${SOURCE_INCLUDE_DIRS})

find_library(
        OPENCV_LIBRARY opencv_java4
        PATHS ${LINK_DIRS}
        NO_CMAKE_FIND_ROOT_PATH)

find_library(
        PYTORCH_LIBRARY pytorch_jni
        PATHS ${LINK_DIRS}
        NO_CMAKE_FIND_ROOT_PATH)

find_library(
        FBJNI_LIBRARY fbjni
        PATHS ${LINK_DIRS}
        NO_CMAKE_FIND_ROOT_PATH)


target_link_libraries(
        card-finder
        ${OPENCV_LIBRARY}
        ${FBJNI_LIBRARY}
        ${PYTORCH_LIBRARY}
)



target_link_libraries(
        native-lib
        card-finder
)


