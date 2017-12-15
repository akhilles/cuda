This code is written for OpenCl 1.2. 
In the makefile change the directories to the local machine's location of the opencl header file and library accordingly.
Make sure that the header files and library are opencl 1.2 (not 2.0) as there are some differences between the versions.

To compile test.cpp the clBlas.dll has to replaced with the appropriate dll from github unless your computer is win 64bit, which is what the current dll is.

To compile the bfs code, run make.

to run type app.exe

To compile the matrix multplication code, test.cpp, run make blas.


To run , type app.exe "1" for global mem

To run , type app.exe "2" for global mem loop unrolling

To run , type app.exe "3" for local mem tiles.