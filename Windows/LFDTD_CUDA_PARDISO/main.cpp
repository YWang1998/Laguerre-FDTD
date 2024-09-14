// includes, system

#include "global.h"
#include "LFDTD_Coe.h"
#include "LFDTD.h"

using namespace std;

int Nnode(0), NNZ(0);

double pi = M_PI;
double eps0 = 8.854e-12;
double mu0 = 4 * pi * 1e-7;
double v0 = 1 / sqrt(eps0 * mu0);

// Current density source & resistor

double R0(50), Rs(0), Rl(0);
int jDirecIndex[3] = { 0,0,1 };
int probeDirecIndex[3] = { 0,0,1 };

int maxit = 2500;
double tol = 1e-8;

int main()
{

    LFDTD_Coe _LFDTD_coe;
    //_LFDTD_coe.Mesh_Grid_PatchAntennaArray("../PatchAntennaArrayMesh"); // Set the grid assignment for the structure
    //_LFDTD_coe.Mesh_Grid_2PortFilter();
    _LFDTD_coe.Mesh_Grid_4PortFilter();
    //_LFDTD_coe.Mesh_Grid_4PortFilter_MS();

    _LFDTD_coe.Coe_SET();

    auto Mesh_Window = std::async(std::launch::async, &Grid::Mesh_Visual, &_LFDTD_coe); // Launch the mesh window asynchronously

    LFDTD _LFDTD;
    try {
        _LFDTD.Solver_Select();
    }
    catch (int)
    {
        std::cerr << "Program EXIT!" << std::endl;
        exit(-1);
    }

    _LFDTD.Sim_start(_LFDTD_coe);

    int Probe_Record;
    std::cout << "Do you want to save the probe profile? - 0: No | 1: Yes\n";
    std::cin >> Probe_Record;

    if (Probe_Record)
    {
        //_LFDTD.result_write("2PortFilter_Laguerre.txt", _LFDTD_coe);
        _LFDTD.Convergence_Profiler("PatchAntenna_Jacobi_Convergence.txt");
        std::cout << "Profile Saved!!\n";
    }

    auto status = Mesh_Window.wait_for(std::chrono::milliseconds(10));
    if (status !=  std::future_status::ready)
    {
        std::cout << "Please exit the mesh window to close the program.\n";
        Mesh_Window.wait();
    }
    
    return 0;
}

