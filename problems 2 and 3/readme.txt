This code is written for OpenCl 1.0. 
In the makefile change the directories to the local machine's location of the opencl header file and library accordingly.
Make sure that the header files and library are opencl 1.0 (not 2.0) as there are some differences between the versions.
To compile the bfs code, change test.cpp to bfs.cpp.

Test.cpp has 3 opencl codes
source string is simple mat mult.
source1 string is unrolled looop mat mult.
source2 is tiled mat mult 16x16 tiles.
Simply change clCreateProgramWithSource function to point to the desired source string to run it.

You can ignore the other files.

To run run app.exe