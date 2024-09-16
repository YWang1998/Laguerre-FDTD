# The Laguerre-FDTD CEM Solver Suite

## Prerequiste

Download and install the [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
For system requirements and installation instructions of cuda toolkit, please refer to the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/), and the [Windows Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

Download and install the [Intel® oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html) and [Intel® oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html) for your corresponding platform.

## Windows

The Windows samples are built using the Visual Studio 2022 IDE. Please also download the Common folder when building the project.

## Linux (Ubuntu 23.10)

The Linux samples are built using Cmake. To build on Linux, download and install Boost C++ library:
```
sudo apt-get install libboost-all-dev
```

Download and install the openGL [GLFW 3.4](https://www.glfw.org/docs/latest/compile.html) package. Another helpful resource to help you build the package is refered [here](https://stackoverflow.com/questions/17768008/how-to-build-install-glfw-3-and-use-it-in-a-linux-project).

Install GLEW and GLM library:
```
sudo apt-get install libglew-dev
sudo apt install libglm-dev
```
