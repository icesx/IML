
include_directories(src/main/cpp/)
cmake_minimum_required(VERSION 3.4.1)
find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

### for mnist
add_library( # Sets the name of the library.
             mnist

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/tensorflow/mnist.cpp)
target_link_libraries( # Specifies the target library.
                        mnist

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib})
