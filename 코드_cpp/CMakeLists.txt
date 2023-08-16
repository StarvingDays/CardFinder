
cmake_minimum_required(VERSION 3.4.1)
set(pathPROJECT C:/Users/Administrator/AndroidStudioProjects/CardFinder)
set(pathOPENCV ${pathPROJECT}/OpenCV)
set(pathLIBOPENCV_JAVA ${pathOPENCV}/native/libs/${ANDROID_ABI}/libopencv_java4.so)
set(CMAKE_CXX_STANDARD 14)
include_directories(
        ${pathOPENCV}/native/jni/include
        ${pathPROJECT}/app/src/main/cpp/include/source
        ${pathPROJECT}/app/src/main/cpp/include/boost_1_82_0_android
)

add_library( # Sets the name of the library.
        CardFinder
        SHARED
        ${pathPROJECT}/app/src/main/cpp/CardFinder.cpp )


add_library( lib_opencv SHARED IMPORTED )
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${pathLIBOPENCV_JAVA})

target_link_libraries( # Specifies the target library.
        CardFinder
        lib_opencv
     )