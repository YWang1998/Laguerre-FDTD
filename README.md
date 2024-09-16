# The Laguerre-FDTD CEM Solver Suite

## Prerequiste

Download and install the [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
For system requirements and installation instructions of cuda toolkit, please refer to the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/), and the [Windows Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

Download and install the [Intel® oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html) and [Intel® oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html) for your corresponding platform.

## Windows

The Windows program is built using the Visual Studio 2022 IDE. Please include the ```Common``` folder when building the project for either platform.

## Linux (Ubuntu 23.10)

### Dependencies

To build on Linux, download and install Boost C++ library:

```
sudo apt-get install libboost-all-dev
```

Download and install the openGL [GLFW 3.4](https://www.glfw.org/docs/latest/compile.html) package. Another helpful resource to help you build the package is refered [here](https://stackoverflow.com/questions/17768008/how-to-build-install-glfw-3-and-use-it-in-a-linux-project).

Install GLEW and GLM library:

```
sudo apt-get install libglew-dev
sudo apt install libglm-dev
```

If you encountered the error when running the program:

```
while loading shared libraries: libiomp5.so: cannot open shared object file: No such file or directory
```

fix it by running the command:

```
sudo apt-get install libomp-dev
```

### Build

If you are using Clion IDE, download both the ```Linux``` and ```Common``` folders. Open the the project folder on Clion and build accordingly.

If you are not working with any IDE, the Linux program can be built using ```cmake``` command. To build the project, first type:

```
mkdir <DIR_NAME>
cd <DIR_NAME>
```

Depending on situation, run either ```Debug``` or ```Release``` mode:

```
cmake .. -DCMAKE_BUILD_TYPE=Debug/Release
make
```

The executable is named ```LFDTD_PARDISO_CUDA```
