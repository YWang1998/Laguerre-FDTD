//
// Created by Yifan Wang on 10/20/23.
//
#include "global.h"
#include "LFDTD_Coe.h"

glm::vec3 Grid::cameraPos = glm::vec3(0.0f, 0.0f, 10.0f);
glm::vec3 Grid::cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 Grid::cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float Grid::cameraSpeed;

float Grid::deltaTime = 0.0f;	// time between current frame and last frame
double Grid::xpos_ini{ Grid::SCR_WIDTH / 2 }, Grid::ypos_ini{ Grid::SCR_HEIGHT / 2 };
double Grid::xpos, Grid::ypos;


void Grid::Mesh_Grid(const std::string& InputFile)
{
    nx = 22;
    ny = 140;
    nz = 13;

    _eps.resize(boost::extents[nx + 1][ny + 1][nz + 1]);
    _mu.resize(boost::extents[nx + 1][ny + 1][nz + 1]);

    _sigmax.resize(boost::extents[nx][ny][nz]);
    _sigmay.resize(boost::extents[nx][ny][nz]);
    _sigmaz.resize(boost::extents[nx][ny][nz]);

    _dxh = std::make_unique<double[]>(nx + 1);
    _dyh = std::make_unique<double[]>(ny + 1);
    _dzh = std::make_unique<double[]>(nz + 1);
    _dxe = std::make_unique<double[]>(nx + 1);
    _dye = std::make_unique<double[]>(ny + 1);
    _dze = std::make_unique<double[]>(nz + 1);

    MatrixImport_1D(InputFile + "/dxh.txt", nx + 1, _dxh.get());
    MatrixImport_1D(InputFile + "/dyh.txt", ny + 1, _dyh.get());
    MatrixImport_1D(InputFile + "/dzh.txt", nz + 1, _dzh.get());

    MatrixImport_1D(InputFile + "/dxe.txt", nx + 1, _dxe.get());
    MatrixImport_1D(InputFile + "/dye.txt", ny + 1, _dye.get());
    MatrixImport_1D(InputFile + "/dze.txt", nz + 1, _dze.get());

    MatrixImport_3D(InputFile + "/sigmax.txt", nx, ny, nz, _sigmax);
    MatrixImport_3D(InputFile + "/sigmay.txt", nx, ny, nz, _sigmay);
    MatrixImport_3D(InputFile + "/sigmaz.txt", nx, ny, nz, _sigmaz);

    fill_3D_array<double_array_type, double>(_eps, nx + 1, ny + 1, nz + 1, eps0);
    fill_3D_array<double_array_type, double>(_mu, nx + 1, ny + 1, nz + 1, mu0);
}

void Grid::Mesh_Grid_PatchAntennaArray(const std::string& InputFile)
{
    nx = 74; ny = 192; nz = 11; PML = 5;

    qstop = 300;

    t = 6e-9; // Simulation time duration
    dt = 1e-12;
    tStep = t / dt;

    s = 5e11; fc = 30e9; td = 1 / (1.5 * fc); tc = 0.5e-9;

    scale = 230 / (s * (tStep - 1) * dt / 2); // Similar as in Myunghyun's paper

    pulseType = 3; // 1 - Gaussian, 2 - Gaussian Derivative, 3 - Modulated Gaussian

    _eps.resize(boost::extents[nx + 1][ny + 1][nz + 1]);
    _mu.resize(boost::extents[nx + 1][ny + 1][nz + 1]);

    _sigmax.resize(boost::extents[nx][ny][nz]);
    _sigmay.resize(boost::extents[nx][ny][nz]);
    _sigmaz.resize(boost::extents[nx][ny][nz]);
    _Jx.resize(boost::extents[nx + 1][ny][nz + 1]);
    _Jy.resize(boost::extents[nx + 1][ny][nz + 1]);
    _Jz.resize(boost::extents[nx + 1][ny + 1][nz]);
    _Rz.resize(boost::extents[nx + 1][ny + 1][nz]);

    _dxh = std::make_unique<double[]>(nx + 1);
    _dyh = std::make_unique<double[]>(ny + 1);
    _dzh = std::make_unique<double[]>(nz + 1);
    _dxe = std::make_unique<double[]>(nx + 1);
    _dye = std::make_unique<double[]>(ny + 1);
    _dze = std::make_unique<double[]>(nz + 1);

    MatrixImport_1D(InputFile + "/dxh.txt", nx + 1, _dxh.get());
    MatrixImport_1D(InputFile + "/dyh.txt", ny + 1, _dyh.get());
    MatrixImport_1D(InputFile + "/dzh.txt", nz + 1, _dzh.get());

    MatrixImport_1D(InputFile + "/dxe.txt", nx + 1, _dxe.get());
    MatrixImport_1D(InputFile + "/dye.txt", ny + 1, _dye.get());
    MatrixImport_1D(InputFile + "/dze.txt", nz + 1, _dze.get());

    MatrixImport_3D(InputFile + "/sigmax.txt", nx, ny, nz, _sigmax);
    MatrixImport_3D(InputFile + "/sigmay.txt", nx, ny, nz, _sigmay);

    MatrixImport_3D(InputFile + "/eps.txt", nx + 1, ny + 1, nz + 1, _eps);
    fill_3D_array<double_array_type, double>(_mu, nx + 1, ny + 1, nz + 1, mu0);

    int jCellIndex[] = { 24 + PML + 1,40 + PML + 1,1 + PML,1 + PML,1 + 1,4 + 1 }; // jCellIndex is assumed to be 1-indexed
    int jResistorIndex[] = {0}; // jResistorIndex is assumed to be 1-indexed
    _JCount = (jCellIndex[1] - jCellIndex[0] + 1) * (jCellIndex[3] - jCellIndex[2] + 1);// Number of parallel current sources
    num_Resistor = 0; num_probe = 1;
    //_jResistorIndex = std::make_unique<int[]>(num_Resistor * 6);

    for (size_t i = 0; i < num_Resistor; ++i)
    {
        for (size_t count = 0; count < 6; ++count) _jResistorIndex[count + i * 6] = jResistorIndex[count + i * 6];
    }

    _jCellIndex = std::make_unique<int[]>(6);
    for (size_t i = 0; i < 6; i++) _jCellIndex[i] = jCellIndex[i];


    if (jDirecIndex[2] == 1) // jCellIndex/jResistorIndex is assumed to be 1-indexed
    {
        for (int i = _jCellIndex[0] - 1; i <= _jCellIndex[1] - 1; ++i) {
            for (int j = _jCellIndex[2] - 1; j <= _jCellIndex[3] - 1; ++j) {
                for (int k = _jCellIndex[4] - 1; k <= _jCellIndex[5] - 1; ++k) {
                    _Jz[i][j][k] = 1;
                }
            }
        }
        for (size_t count = 0; count < num_Resistor; count++)
        {
            for (int i = _jResistorIndex[0 + 6 * count] - 1; i <= _jResistorIndex[1 + 6 * count] - 1; ++i) {
                for (int j = _jResistorIndex[2 + 6 * count] - 1; j <= _jResistorIndex[3 + 6 * count] - 1; ++j) {
                    for (int k = _jResistorIndex[4 + 6 * count] - 1; k <= _jResistorIndex[5 + 6 * count] - 1; ++k) {
                        _Rz[i][j][k] = 1;
                    }
                }
            }
        }
    }

    // Probe Definition

    int probeCell[] = { 32 + PML + 1,32 + PML + 1,15 + PML,15 + PML,3 + 1,3 + 1 };

    _probeCell = std::make_unique<int[]>(num_probe * 6);
    for (size_t i = 0; i < num_probe; ++i)
    {
        for (size_t count = 0; count < 6; ++count)
        {
            _probeCell[count + i * 6] = probeCell[count + i * 6];

        }
    }

}

void Grid::Mesh_Grid()
{
    nx = 22; ny = 140; nz = 13; PML = 8;

    qstop = 300;

    t = 6e-9; // Simulation time duration
    dt = 2e-12;
    tStep = t / dt;

    s = 3e11; fc = 40e9; td = 1 / (2 * fc); tc = 0.2e-9;

    scale = 230 / (s * (tStep - 1) * dt / 2); // Similar as in Myunghyun's paper

    pulseType = 1; // 1 - Gaussian, 2 - Gaussian Derivative, 3 - Modulated Gaussian

    _eps.resize(boost::extents[nx + 1][ny + 1][nz + 1]);
    _mu.resize(boost::extents[nx + 1][ny + 1][nz + 1]);

    _sigmax.resize(boost::extents[nx][ny][nz]);
    _sigmay.resize(boost::extents[nx][ny][nz]);
    _sigmaz.resize(boost::extents[nx][ny][nz]);
    _Jx.resize(boost::extents[nx + 1][ny][nz + 1]);
    _Jy.resize(boost::extents[nx + 1][ny][nz + 1]);
    _Jz.resize(boost::extents[nx + 1][ny + 1][nz]);
    _Rz.resize(boost::extents[nx + 1][ny + 1][nz]);

    _dxh = std::make_unique<double[]>(nx + 1);
    _dyh = std::make_unique<double[]>(ny + 1);
    _dzh = std::make_unique<double[]>(nz + 1);
    _dxe = std::make_unique<double[]>(nx + 1);
    _dye = std::make_unique<double[]>(ny + 1);
    _dze = std::make_unique<double[]>(nz + 1);

    for (int i = 0; i < PML; ++i)
    {
        _dxe[i] = 0.06e-3;
        _dye[i] = 0.06e-3;
    }

    _dxe[PML] = 0.05e-3; // Input line
    for (int i = 1 + PML; i < 5 + PML; ++i) _dxe[i] = 0.06e-3; // Input line
    _dxe[5 + PML] = 0.07e-3; // Input line
    for (int i = 6 + PML; i < nx + 1; ++i) _dxe[i] = 0.06e-3; // PML Region

    for (int j = PML; j < ny + 1; ++j) _dye[j] = 0.06e-3; // Input port


    // Flip and copy dy assignment
    for (int j = 0; j < ny / 2; ++j)
    {
        _dye[ny - j - 1] = _dye[j];
    }
    _dye[ny] = 0.06e-3;

    _dze[0] = 0.1e-3;
    for (int k = 1; k < 3; ++k) _dze[k] = 0.0635e-3;
    for (int k = 3; k < nz + 1; ++k) _dze[k] = 0.1e-3;

    fill_3D_array<double_array_type, double>(_eps, nx + 1, ny + 1, nz + 1, eps0);
    fill_3D_array<double_array_type, double>(_mu, nx + 1, ny + 1, nz + 1, mu0);


    // Ground Plane
    for (int i = PML; i < nx - PML; ++i) {
        for (int j = PML; j < ny - PML + 1; ++j) {
            _sigmax[i][j][1] = 5.8e8;
        }
    }
    for (int i = PML; i < nx - PML + 1; ++i) {
        for (int j = PML; j < ny - PML; ++j) {
            _sigmay[i][j][1] = 5.8e8;
        }
    }

    // TL 1 
    for (int i = 1 + PML - 1; i < 6 + PML; ++i)
    {
        for (int j = PML; j < ny - PML + 1; ++j)
        {
            _sigmax[i][j][3] = 5.8e8;
        }
    }
    for (int i = 1 + PML - 1; i < 6 + PML + 1; ++i)
    {
        for (int j = PML; j < ny - PML; ++j)
        {
            _sigmay[i][j][3] = 5.8e8;
        }
    }

    // Dielectric Substrate
    for (int i = PML; i < nx - PML; ++i) {
        for (int j = PML; j < ny - PML; ++j) {
            for (int k = 1; k < 3; ++k) {
                _eps[i][j][k] *= 2.2;
            }
        }
    }

    for (int i = 0; i < nx + 1; ++i)
    {
        if (i == 0) {
            _dxh[i] = _dxe[i];
        }
        else
        {
            _dxh[i] = (_dxe[i] + _dxe[i - 1]) / 2;
        }
    }

    for (int i = 0; i < ny + 1; ++i)
    {
        if (i == 0) {
            _dyh[i] = _dye[i];
        }
        else
        {
            _dyh[i] = (_dye[i] + _dye[i - 1]) / 2;
        }
    }

    for (int i = 0; i < nz + 1; ++i)
    {
        if (i == 0) {
            _dzh[i] = _dze[i];
        }
        else
        {
            _dzh[i] = (_dze[i] + _dze[i - 1]) / 2;
        }
    }

    int jCellIndex[] = { PML + 1,PML + 6 + 1,1 + PML,1 + PML,1 + 1,2 + 1 }; // jCellIndex is assumed to be 1-indexed
    int jResistorIndex[] = { PML + 1,PML + 6 + 1,124 + PML + 1,124 + PML + 1,1 + 1,2 + 1 }; // jResistorIndex is assumed to be 1-indexed
    _JCount = (jCellIndex[1] - jCellIndex[0] + 1) * (jCellIndex[3] - jCellIndex[2] + 1);// Number of parallel current sources
    num_Resistor = 1; num_probe = 2;
    _jResistorIndex = std::make_unique<int[]>(num_Resistor * 6);

    for (size_t i = 0; i < num_Resistor; ++i)
    {
        for (size_t count = 0; count < 6; ++count) _jResistorIndex[count + i * 6] = jResistorIndex[count + i * 6];
    }

    _jCellIndex = std::make_unique<int[]>(6);
    for (size_t i = 0; i < 6; i++) _jCellIndex[i] = jCellIndex[i];


    if (jDirecIndex[2] == 1) // jCellIndex/jResistorIndex is assumed to be 1-indexed
    {
        for (int i = _jCellIndex[0] - 1; i <= _jCellIndex[1] - 1; ++i) {
            for (int j = _jCellIndex[2] - 1; j <= _jCellIndex[3] - 1; ++j) {
                for (int k = _jCellIndex[4] - 1; k <= _jCellIndex[5] - 1; ++k) {
                    _Jz[i][j][k] = 1;
                }
            }
        }
        for (size_t count = 0; count < num_Resistor; count++)
        {
            for (int i = _jResistorIndex[0 + 6 * count] - 1; i <= _jResistorIndex[1 + 6 * count] - 1; ++i) {
                for (int j = _jResistorIndex[2 + 6 * count] - 1; j <= _jResistorIndex[3 + 6 * count] - 1; ++j) {
                    for (int k = _jResistorIndex[4 + 6 * count] - 1; k <= _jResistorIndex[5 + 6 * count] - 1; ++k) {
                        _Rz[i][j][k] = 1;
                    }
                }
            }
        }
    }

    // Probe Definition

    int probeCell[] = { PML + 1,PML + 6 + 1,11 + PML,11 + PML,1 + 1,2 + 1,
                       PML + 1,PML + 6 + 1,114 + PML + 1,114 + PML + 1,1 + 1,2 + 1 };

    _probeCell = std::make_unique<int[]>(num_probe * 6);
    for (size_t i = 0; i < num_probe; ++i)
    {
        for (size_t count = 0; count < 6; ++count)
        {
            _probeCell[count + i * 6] = probeCell[count + i * 6];

        }
    }
    

}

void Grid::Mesh_Grid_2PortFilter()
{
    nx = 63; ny = 135; nz = 16; PML = 6;

    qstop = 200;

    t = 3e-9; // Simulation time duration
    dt = 2e-12;
    tStep = t / dt;

    s = 1.5e11; fc = 25e9; td = 1 / (1.5 * fc); tc = 0.2e-9;

    scale = 230 / (s * (tStep - 1) * dt / 2); // Similar as in Myunghyun's paper

    pulseType = 1; // 1 - Gaussian, 2 - Gaussian Derivative, 3 - Modulated Gaussian

    _eps.resize(boost::extents[nx + 1][ny + 1][nz + 1]);
    _mu.resize(boost::extents[nx + 1][ny + 1][nz + 1]);

    _sigmax.resize(boost::extents[nx][ny][nz]);
    _sigmay.resize(boost::extents[nx][ny][nz]);
    _sigmaz.resize(boost::extents[nx][ny][nz]);
    _Jx.resize(boost::extents[nx + 1][ny][nz + 1]);
    _Jy.resize(boost::extents[nx + 1][ny][nz + 1]);
    _Jz.resize(boost::extents[nx + 1][ny + 1][nz]);
    _Rz.resize(boost::extents[nx + 1][ny + 1][nz]);

    _dxh = std::make_unique<double[]>(nx + 1);
    _dyh = std::make_unique<double[]>(ny + 1);
    _dzh = std::make_unique<double[]>(nz + 1);
    _dxe = std::make_unique<double[]>(nx + 1);
    _dye = std::make_unique<double[]>(ny + 1);
    _dze = std::make_unique<double[]>(nz + 1);


    for (int i = 0; i < PML - 1; ++i)
    {
        _dxe[i] = 0.15e-3;
        _dye[i] = 0.15e-3;
    }
    _dze[0] = 0.15e-3;

    // Upper Region
    for (int i = PML - 1; i < 25 + PML; ++i) { _dxe[i] = 0.05e-3 * 2; }
    // Feed Line
    _dxe[25 + PML] = 0.05e-3;
    // Lower Region
    for (int i = 26 + PML; i < nx - PML + 1; ++i) { _dxe[i] = 0.05e-3 * 2; }

    // Feed Line 1
    for (int j = PML - 1; j < 20 + PML; ++j) { _dye[j] = 0.05e-3 * 2; }
    // Stub 1
    for (int j = 20 + PML; j < 23 + PML; ++j) { _dye[j] = 0.05e-3; }
    // Center TL till TL 1 for Patch 1
    for (int j = 23 + PML; j < 40 + PML; ++j) { _dye[j] = 0.05e-3 * 2; }
    // TL 1 for Patch 1
    _dye[40 + PML] = 0.05e-3;
    // TL 1 for Patch 1 to TL 2 for Patch 2
    for (int j = 41 + PML; j < 61 + PML; ++j) { _dye[j] = 0.05e-3 * 2; }
    // TL 2 for Patch 2
    _dye[61 + PML] = 0.05e-3;
    // TL 2 for Patch 2 to TL 3 for Patch 3
    for (int j = 62 + PML; j < 82 + PML; ++j) { _dye[j] = 0.05e-3 * 2; }
    // TL 3 for Patch 3
    _dye[82 + PML] = 0.05e-3;
    // TL 3 for Patch 3 to Stub 2
    for (int j = 83 + PML; j < 100 + PML; ++j) { _dye[j] = 0.05e-3 * 2; }
    // Stub 2
    for (int j = 100 + PML; j < 103 + PML; ++j) { _dye[j] = 0.05e-3; }
    // Feed Line 2
    for (int j = 103 + PML; j < ny - PML + 1; ++j) { _dye[j] = 0.05e-3 * 2; }

    for (int k = 1; k < 6; ++k) { _dze[k] = 0.05e-3; }

    for (int i = nx - PML + 1; i < nx + 1; ++i) { _dxe[i] = 0.15e-3; }
    for (int j = ny - PML + 1; j < ny + 1; ++j) { _dye[j] = 0.15e-3; }
    for (int k = 6; k < nz + 1; ++k) { _dze[k] = 0.15e-3; }

    fill_3D_array<double_array_type, double>(_eps, nx + 1, ny + 1, nz + 1, eps0);
    fill_3D_array<double_array_type, double>(_mu, nx + 1, ny + 1, nz + 1, mu0);


    // Ground Plane
    for (int i = PML; i < nx - PML; ++i) {
        for (int j = PML; j < ny - PML + 1; ++j) {
            _sigmax[i][j][2] = 5.8e8;
        }
    }
    for (int i = PML; i < nx - PML + 1; ++i) {
        for (int j = PML; j < ny - PML; ++j) {
            _sigmay[i][j][2] = 5.8e8;
        }
    }

    // TL 1 
    for (int i = 25 + PML - 1; i < 27 + PML; ++i)
    {
        for (int j = PML; j < 20 + PML + 1; ++j)
        {
            _sigmax[i][j][5] = 5.8e8;
        }
    }
    for (int i = 25 + PML - 1; i < 27 + PML + 1; ++i)
    {
        for (int j = PML; j < 20 + PML; ++j)
        {
            _sigmay[i][j][5] = 5.8e8;
        }
    }

    // Stub 1 
    for (int i = 9 + PML - 1; i < 43 + PML; ++i)
    {
        for (int j = 21 + PML - 1; j < 23 + PML + 1; ++j)
        {
            _sigmax[i][j][5] = 5.8e8;
        }
    }
    for (int i = 9 + PML - 1; i < 43 + PML + 1; ++i)
    {
        for (int j = 21 + PML - 1; j < 23 + PML; ++j)
        {
            _sigmay[i][j][5] = 5.8e8;
        }
    }

    // Center TL 
    for (int i = 26 + PML - 1; i < 26 + PML; ++i)
    {
        for (int j = 24 + PML - 1; j < 100 + PML + 1; ++j)
        {
            _sigmax[i][j][5] = 5.8e8;
        }
    }
    for (int i = 26 + PML - 1; i < 26 + PML + 1; ++i)
    {
        for (int j = 24 + PML - 1; j < 100 + PML; ++j)
        {
            _sigmay[i][j][5] = 5.8e8;
        }
    }

    // Stub 2
    for (int i = 9 + PML - 1; i < 43 + PML; ++i)
    {
        for (int j = 101 + PML - 1; j < 103 + PML + 1; ++j)
        {
            _sigmax[i][j][5] = 5.8e8;
        }
    }
    for (int i = 9 + PML - 1; i < 43 + PML + 1; ++i)
    {
        for (int j = 101 + PML - 1; j < 103 + PML; ++j)
        {
            _sigmay[i][j][5] = 5.8e8;
        }
    }

    // TL 2
    for (int i = 25 + PML - 1; i < 27 + PML; ++i)
    {
        for (int j = 104 + PML - 1; j < 123 + PML + 1; ++j)
        {
            _sigmax[i][j][5] = 5.8e8;
        }
    }
    for (int i = 25 + PML - 1; i < 27 + PML + 1; ++i)
    {
        for (int j = 104 + PML - 1; j < 123 + PML; ++j)
        {
            _sigmay[i][j][5] = 5.8e8;
        }
    }

    // Patch 1
    for (int i = 6 + PML - 1; i < 17 + PML; ++i)
    {
        for (int j = 37 + PML - 1; j < 45 + PML + 1; ++j)
        {
            _sigmax[i][j][5] = 5.8e8;
        }
    }
    for (int i = 6 + PML - 1; i < 17 + PML + 1; ++i)
    {
        for (int j = 37 + PML - 1; j < 45 + PML; ++j)
        {
            _sigmay[i][j][5] = 5.8e8;
        }
    }

    // TL to Patch 1
    for (int i = 18 + PML - 1; i < 25 + PML; ++i)
    {
        for (int j = 41 + PML - 1; j < 41 + PML + 1; ++j)
        {
            _sigmax[i][j][5] = 5.8e8;
        }
    }
    for (int i = 18 + PML - 1; i < 25 + PML + 1; ++i)
    {
        for (int j = 41 + PML - 1; j < 41 + PML; ++j)
        {
            _sigmay[i][j][5] = 5.8e8;
        }
    }

    // Patch 2
    for (int i = 36 + PML - 1; i < 47 + PML; ++i)
    {
        for (int j = 58 + PML - 1; j < 66 + PML + 1; ++j)
        {
            _sigmax[i][j][5] = 5.8e8;
        }
    }
    for (int i = 36 + PML - 1; i < 47 + PML + 1; ++i)
    {
        for (int j = 58 + PML - 1; j < 66 + PML; ++j)
        {
            _sigmay[i][j][5] = 5.8e8;
        }
    }

    // TL to Patch 2
    for (int i = 27 + PML - 1; i < 35 + PML; ++i)
    {
        for (int j = 62 + PML - 1; j < 62 + PML + 1; ++j)
        {
            _sigmax[i][j][5] = 5.8e8;
        }
    }
    for (int i = 27 + PML - 1; i < 35 + PML + 1; ++i)
    {
        for (int j = 62 + PML - 1; j < 62 + PML; ++j)
        {
            _sigmay[i][j][5] = 5.8e8;
        }
    }

    // Patch 3
    for (int i = 6 + PML - 1; i < 17 + PML; ++i)
    {
        for (int j = 79 + PML - 1; j < 87 + PML + 1; ++j)
        {
            _sigmax[i][j][5] = 5.8e8;
        }
    }
    for (int i = 6 + PML - 1; i < 17 + PML + 1; ++i)
    {
        for (int j = 79 + PML - 1; j < 87 + PML; ++j)
        {
            _sigmay[i][j][5] = 5.8e8;
        }
    }

    // TL to Patch 3
    for (int i = 18 + PML - 1; i < 25 + PML; ++i)
    {
        for (int j = 83 + PML - 1; j < 83 + PML + 1; ++j)
        {
            _sigmax[i][j][5] = 5.8e8;
        }
    }
    for (int i = 18 + PML - 1; i < 25 + PML + 1; ++i)
    {
        for (int j = 83 + PML - 1; j < 83 + PML; ++j)
        {
            _sigmay[i][j][5] = 5.8e8;
        }
    }


    // Dielectric Substrate
    for (int i = PML; i < nx - PML; ++i) {
        for (int j = PML; j < ny - PML; ++j) {
            for (int k = 2; k <= 4; ++k) {
                _eps[i][j][k] *= 2.94;
            }
        }
    }


    for (int i = 0; i < nx + 1; ++i)
    {
        if (i == 0) {
            _dxh[i] = _dxe[i];
        }
        else
        {
            _dxh[i] = (_dxe[i] + _dxe[i - 1]) / 2;
        }
    }

    for (int i = 0; i < ny + 1; ++i)
    {
        if (i == 0) {
            _dyh[i] = _dye[i];
        }
        else
        {
            _dyh[i] = (_dye[i] + _dye[i - 1]) / 2;
        }
    }

    for (int i = 0; i < nz + 1; ++i)
    {
        if (i == 0) {
            _dzh[i] = _dze[i];
        }
        else
        {
            _dzh[i] = (_dze[i] + _dze[i - 1]) / 2;
        }
    }


    int jCellIndex[] = { 25 + PML,27 + PML + 1,1 + PML,1 + PML,2 + 1,4 + 1 }; // jCellIndex is assumed to be 1-indexed
    int jResistorIndex[] = { 25 + PML,27 + PML + 1,123 + PML + 1,123 + PML + 1,2 + 1,4 + 1 }; // jResistorIndex is assumed to be 1-indexed
    _JCount = (jCellIndex[1] - jCellIndex[0] + 1) * (jCellIndex[3] - jCellIndex[2] + 1);// Number of parallel current sources
    num_Resistor = 1; num_probe = 2;
    _jResistorIndex = std::make_unique<int[]>(num_Resistor * 6);

    for (size_t i = 0; i < num_Resistor; ++i)
    {
        for (size_t count = 0; count < 6; ++count) _jResistorIndex[count + i * 6] = jResistorIndex[count + i * 6];
    }

    _jCellIndex = std::make_unique<int[]>(6);
    for (size_t i = 0; i < 6; i++) _jCellIndex[i] = jCellIndex[i];


    if (jDirecIndex[2] == 1) // jCellIndex/jResistorIndex is assumed to be 1-indexed
    {
        for (int i = _jCellIndex[0] - 1; i <= _jCellIndex[1] - 1; ++i) {
            for (int j = _jCellIndex[2] - 1; j <= _jCellIndex[3] - 1; ++j) {
                for (int k = _jCellIndex[4] - 1; k <= _jCellIndex[5] - 1; ++k) {
                    _Jz[i][j][k] = 1;
                }
            }
        }
        for (size_t count = 0; count < num_Resistor; count++)
        {
            for (int i = _jResistorIndex[0 + 6 * count] - 1; i <= _jResistorIndex[1 + 6 * count] - 1; ++i) {
                for (int j = _jResistorIndex[2 + 6 * count] - 1; j <= _jResistorIndex[3 + 6 * count] - 1; ++j) {
                    for (int k = _jResistorIndex[4 + 6 * count] - 1; k <= _jResistorIndex[5 + 6 * count] - 1; ++k) {
                        _Rz[i][j][k] = 1;
                    }
                }
            }
        }
    }

    // Probe Definition

    int probeCell[] = { 25 + PML,27 + PML + 1,11 + PML,11 + PML,2 + 1,4 + 1,
                        25 + PML,27 + PML + 1,113 + PML + 1,113 + PML + 1,2 + 1,4 + 1 };

    _probeCell = std::make_unique<int[]>(num_probe * 6);
    for (size_t i = 0; i < num_probe; ++i)
    {
        for (size_t count = 0; count < 6; ++count)
        {
            _probeCell[count + i * 6] = probeCell[count + i * 6];

        }
    }

}

void Grid::Mesh_Grid_4PortFilter()
{
    nx = 53; ny = 89; nz = 13; PML = 5;

    qstop = 400;

    t = 6e-9; // Simulation time duration
    dt = 2e-12;
    tStep = t / dt;

    s = 1.5e11; fc = 10e9; td = 1 / (1.5 * fc); tc = 0.5e-9;

    scale = 230 / (s * (tStep - 1) * dt / 2); // Similar as in Myunghyun's paper

    pulseType = 1; // 1 - Gaussian, 2 - Gaussian Derivative, 3 - Modulated Gaussian

    _eps.resize(boost::extents[nx + 1][ny + 1][nz + 1]);
    _mu.resize(boost::extents[nx + 1][ny + 1][nz + 1]);

    _sigmax.resize(boost::extents[nx][ny][nz]);
    _sigmay.resize(boost::extents[nx][ny][nz]);
    _sigmaz.resize(boost::extents[nx][ny][nz]);
    _Jx.resize(boost::extents[nx + 1][ny][nz + 1]);
    _Jy.resize(boost::extents[nx + 1][ny][nz + 1]);
    _Jz.resize(boost::extents[nx + 1][ny + 1][nz]);
    _Rz.resize(boost::extents[nx + 1][ny + 1][nz]);

    _dxh = std::move(std::make_unique<double[]>(nx + 1));
    _dyh = std::move(std::make_unique<double[]>(ny + 1));
    _dzh = std::move(std::make_unique<double[]>(nz + 1));
    _dxe = std::move(std::make_unique<double[]>(nx + 1));
    _dye = std::move(std::make_unique<double[]>(ny + 1));
    _dze = std::move(std::make_unique<double[]>(nz + 1));

    for (int i = 0; i < PML; ++i)
    {
        _dxe[i] = 1e-3;
        _dye[i] = 1e-3;
    }

    // Substrate Edge Offset
    for (int i = PML; i < PML + 5; ++i) { _dxe[i] = 1e-3; }
    // Upper Resonator Block
    for (int i = PML + 5; i < PML + 9; ++i) { _dxe[i] = 0.52e-3; }
    _dxe[PML + 9] = 0.075e-3;
    _dxe[PML + 10] = 0.09e-3;
    _dxe[PML + 11] = 0.18e-3;
    _dxe[PML + 12] = 0.09e-3;
    _dxe[PML + 13] = 0.075e-3;
    for (int i = PML + 14; i < PML + 18; ++i) { _dxe[i] = 0.52e-3; }
    // Taper to middle resonator block
    _dxe[PML + 18] = 1.446e-3;
    // Middle Resonator Block
    for (int i = PML + 19; i < PML + 21; ++i) { _dxe[i] = 1e-3; }
    _dxe[PML + 21] = 0.4e-3;
    // Flip and copy the x directional grid
    for (int i = 0; i < nx / 2; ++i) { _dxe[nx - i - 1] = _dxe[i]; }

    // TL
    for (int i = PML; i < PML + 10; ++i) { _dye[i] = 1e-3; }
    // First middle resonator offset
    _dye[PML + 10] = 1.15625e-3;
    // Upper Resonator Block
    for (int i = PML + 11; i < PML + 13; ++i) { _dye[i] = 0.99375e-3; }
    _dye[PML + 13] = 0.5e-3;
    for (int i = PML + 14; i < PML + 16; ++i) { _dye[i] = 0.99375e-3; }
    // First Taper Transition
    _dye[PML + 16] = 1.5625e-3;
    for (int i = PML + 17; i < PML + 20; ++i) { _dye[i] = 1e-3; }
    _dye[PML + 20] = 1.575e-3;
    for (int i = PML + 21; i < PML + 23; ++i) { _dye[i] = 1.43125e-3; }
    // Upper Resonator Block
    for (int i = PML + 23; i < PML + 25; ++i) { _dye[i] = 0.99375e-3; }
    _dye[PML + 25] = 0.5e-3;
    for (int i = PML + 26; i < PML + 28; ++i) { _dye[i] = 0.99375e-3; }
    // Second Taper Transition
    for (int i = PML + 28; i < PML + 30; ++i) { _dye[i] = 1.43125e-3; }
    for (int i = PML + 30; i < PML + 35; ++i) { _dye[i] = 1e-3; }
    _dye[PML + 35] = 1.2125e-3;
    _dye[PML + 36] = 0.925e-3;
    // Third Resonator Block (Half)
    for (int i = PML + 37; i < PML + 39; ++i) { _dye[i] = 0.99375e-3; }
    _dye[PML + 39] = 0.5e-3;
    // Flip and copy the y directional grid
    for (int i = 0; i < ny / 2; ++i) { _dye[ny - i - 1] = _dye[i]; }

    for (int i = 0; i < 2; ++i) { _dze[i] = 1e-3; }
    for (int i = 2; i < 4; ++i) { _dze[i] = 0.3175e-3; }
    for (int i = 4; i < nz; ++i) { _dze[i] = 1e-3; }

    fill_3D_array<double_array_type>(_eps, nx + 1, ny + 1, nz + 1, eps0);
    fill_3D_array<double_array_type>(_mu, nx + 1, ny + 1, nz + 1, mu0);

    // Dielectric Substrate
    for (int i = PML; i < nx - PML; ++i) {
        for (int j = PML; j < ny - PML; ++j) {
            for (int k = 2; k < 4; ++k) {
                _eps[i][j][k] *= 10.2;
            }
        }
    }

    // Ground Plane
    for (int i = PML; i < nx - PML; ++i) {
        for (int j = PML; j < ny - PML + 1; ++j) {
            _sigmax[i][j][2] = 5.8e8;
        }
    }
    for (int i = PML; i < nx - PML + 1; ++i) {
        for (int j = PML; j < ny - PML; ++j) {
            _sigmay[i][j][2] = 5.8e8;
        }
    }

    /*** Sigma X Section ***/

    // TL
    for (int i = PML + 9; i < PML + 14; ++i) {
        for (int j = PML; j < PML + 12; ++j) {
            _sigmax[i][j][4] = 5.8e8;
        }
    }

    // Resonator Block
    for (int i = PML + 5; i < PML + 18; ++i) {
        for (int j = PML + 11; j < PML + 17; ++j) {
            _sigmax[i][j][4] = 5.8e8;
        }
    }

    // 1st Taper
    for (int i = PML + 10; i < PML + 13; ++i) {
        for (int j = PML + 16; j < PML + 24; ++j) {
            _sigmax[i][j][4] = 5.8e8;
        }
    }

    // Resonator Block
    for (int i = PML + 5; i < PML + 18; ++i) {
        for (int j = PML + 23; j < PML + 29; ++j) {
            _sigmax[i][j][4] = 5.8e8;
        }
    }

    // 2nd Taper
    for (int i = PML + 11; i < PML + 12; ++i) {
        for (int j = PML + 28; j < PML + 38; ++j) {
            _sigmax[i][j][4] = 5.8e8;
        }
    }

    // Resonator Block
    for (int i = PML + 5; i < PML + 18; ++i) {
        for (int j = PML + 37; j < PML + 41; ++j) {
            _sigmax[i][j][4] = 5.8e8;
        }
    }

    // Taper to middle resonator
    for (int i = PML + 18; i < PML + 19; ++i) {
        for (int j = PML + 13; j < PML + 15; ++j) {
            _sigmax[i][j][4] = 5.8e8;
        }
    }
    for (int i = PML + 18; i < PML + 19; ++i) {
        for (int j = PML + 25; j < PML + 27; ++j) {
            _sigmax[i][j][4] = 5.8e8;
        }
    }
    for (int i = PML + 18; i < PML + 19; ++i) {
        for (int j = PML + 39; j < PML + 41; ++j) {
            _sigmax[i][j][4] = 5.8e8;
        }
    }

    // 1st Middle resonator
    for (int i = PML + 19; i < PML + 22; ++i) {
        for (int j = PML + 10; j < PML + 18; ++j) {
            _sigmax[i][j][4] = 5.8e8;
        }
    }

    // 2nd Middle resonator
    for (int i = PML + 19; i < PML + 22; ++i) {
        for (int j = PML + 21; j < PML + 31; ++j) {
            _sigmax[i][j][4] = 5.8e8;
        }
    }
    // 3rd Middle resonator
    for (int i = PML + 19; i < PML + 22; ++i) {
        for (int j = PML + 36; j < PML + 41; ++j) {
            _sigmax[i][j][4] = 5.8e8;
        }
    }

    // Flip and copy
    for (int i = 0; i < nx / 2; ++i) {
        for (int j = 0; j < ny; ++j) {
            _sigmax[nx - i - 1][j][4] = _sigmax[i][j][4];
        }
    }
    int Reverse_y = PML + 38;
    for (int j = PML + 41; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            _sigmax[i][j][4] = _sigmax[i][Reverse_y][4];
        }
        Reverse_y--;
    }

    /*** Sigma Y Section ***/
    // TL
    for (int i = PML + 9; i < PML + 14 + 1; ++i) {
        for (int j = PML; j < PML + 12 - 1; ++j) {
            _sigmay[i][j][4] = 5.8e8;
        }
    }
    // Resonator Block
    for (int i = PML + 5; i < PML + 18 + 1; ++i) {
        for (int j = PML + 11; j < PML + 17 - 1; ++j) {
            _sigmay[i][j][4] = 5.8e8;
        }
    }
    // 1st Taper
    for (int i = PML + 10; i < PML + 13 + 1; ++i) {
        for (int j = PML + 16; j < PML + 24 - 1; ++j) {
            _sigmay[i][j][4] = 5.8e8;
        }
    }
    // Resonator Block
    for (int i = PML + 5; i < PML + 18 + 1; ++i) {
        for (int j = PML + 23; j < PML + 29 - 1; ++j) {
            _sigmay[i][j][4] = 5.8e8;
        }
    }
    // 2nd Taper
    for (int i = PML + 11; i < PML + 12 + 1; ++i) {
        for (int j = PML + 28; j < PML + 38 - 1; ++j) {
            _sigmay[i][j][4] = 5.8e8;
        }
    }
    // Resonator Block
    for (int i = PML + 5; i < PML + 18 + 1; ++i) {
        for (int j = PML + 37; j < PML + 41 - 1; ++j) {
            _sigmay[i][j][4] = 5.8e8;
        }
    }
    // Taper to middle resonator
    for (int i = PML + 18; i < PML + 19 + 1; ++i) {
        for (int j = PML + 13; j < PML + 15 - 1; ++j) {
            _sigmay[i][j][4] = 5.8e8;
        }
    }
    for (int i = PML + 18; i < PML + 19 + 1; ++i) {
        for (int j = PML + 25; j < PML + 27 - 1; ++j) {
            _sigmay[i][j][4] = 5.8e8;
        }
    }
    for (int i = PML + 18; i < PML + 19 + 1; ++i) {
        for (int j = PML + 39; j < PML + 41 - 1; ++j) {
            _sigmay[i][j][4] = 5.8e8;
        }
    }

    // 1st Middle resonator
    for (int i = PML + 19; i < PML + 22 + 1; ++i) {
        for (int j = PML + 10; j < PML + 18 - 1; ++j) {
            _sigmay[i][j][4] = 5.8e8;
        }
    }

    // 2nd Middle resonator
    for (int i = PML + 19; i < PML + 22 + 1; ++i) {
        for (int j = PML + 21; j < PML + 31 - 1; ++j) {
            _sigmay[i][j][4] = 5.8e8;
        }
    }
    // 3rd Middle resonator
    for (int i = PML + 19; i < PML + 22 + 1; ++i) {
        for (int j = PML + 36; j < PML + 41 - 1; ++j) {
            _sigmay[i][j][4] = 5.8e8;
        }
    }

    // Flip and copy
    int Reverse_x = PML + 20;
    for (int i = PML + 23; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            _sigmay[i][j][4] = _sigmay[Reverse_x][j][4];
        }
        Reverse_x--;
    }

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny / 2; ++j) {
            _sigmay[i][ny - 1 - j][4] = _sigmay[i][j][4];
        }
    }


    for (int i = 0; i < nx + 1; ++i)
    {
        if (i == 0) {
            _dxh[i] = _dxe[i];
        }
        else
        {
            _dxh[i] = (_dxe[i] + _dxe[i - 1]) / 2;
        }
    }

    for (int i = 0; i < ny + 1; ++i)
    {
        if (i == 0) {
            _dyh[i] = _dye[i];
        }
        else
        {
            _dyh[i] = (_dye[i] + _dye[i - 1]) / 2;
        }
    }

    for (int i = 0; i < nz + 1; ++i)
    {
        if (i == 0) {
            _dzh[i] = _dze[i];
        }
        else
        {
            _dzh[i] = (_dze[i] + _dze[i - 1]) / 2;
        }
    }


    int jCellIndex[] = { PML + 10, PML + 15, PML + 1, PML + 1, 2 + 1, 3 + 1 }; // jCellIndex is assumed to be 1-indexed
    int jResistorIndex[] = { PML + 10, PML + 15, ny - PML + 1, ny - PML + 1, 2 + 1, 3 + 1,
                             PML + 30, PML + 35, PML + 1, PML + 1, 2 + 1, 3 + 1,
                             PML + 30, PML + 35, ny - PML + 1, ny - PML + 1, 2 + 1, 3 + 1 }; // jResistorIndex is assumed to be 1-indexed
    _JCount = (jCellIndex[1] - jCellIndex[0] + 1) * (jCellIndex[3] - jCellIndex[2] + 1);// Number of parallel current sources
    num_Resistor = 3; num_probe = 2;
    _jResistorIndex = std::make_unique<int[]>(num_Resistor * 6);

    for (size_t i = 0; i < num_Resistor; ++i)
    {
        for (size_t count = 0; count < 6; ++count) _jResistorIndex[count + i * 6] = jResistorIndex[count + i * 6];
    }

    _jCellIndex = std::make_unique<int[]>(6);
    for (size_t i = 0; i < 6; i++) _jCellIndex[i] = jCellIndex[i];


    if (jDirecIndex[2] == 1) // jCellIndex/jResistorIndex is assumed to be 1-indexed
    {
        for (int i = _jCellIndex[0] - 1; i <= _jCellIndex[1] - 1; ++i) {
            for (int j = _jCellIndex[2] - 1; j <= _jCellIndex[3] - 1; ++j) {
                for (int k = _jCellIndex[4] - 1; k <= _jCellIndex[5] - 1; ++k) {
                    _Jz[i][j][k] = 1;
                }
            }
        }
        for (size_t count = 0; count < num_Resistor; count++)
        {
            for (int i = _jResistorIndex[0 + 6 * count] - 1; i <= _jResistorIndex[1 + 6 * count] - 1; ++i) {
                for (int j = _jResistorIndex[2 + 6 * count] - 1; j <= _jResistorIndex[3 + 6 * count] - 1; ++j) {
                    for (int k = _jResistorIndex[4 + 6 * count] - 1; k <= _jResistorIndex[5 + 6 * count] - 1; ++k) {
                        _Rz[i][j][k] = 1;
                    }
                }
            }
        }
    }

    // Probe Definition

    int probeCell[] = { PML + 10, PML + 15, PML + 6, PML + 6, 2 + 1, 3 + 1,
                        PML + 10, PML + 15, ny - PML - 6, ny - PML - 6, 2 + 1, 3 + 1 };

    _probeCell = std::make_unique<int[]>(num_probe * 6);
    for (size_t i = 0; i < num_probe; ++i)
    {
        for (size_t count = 0; count < 6; ++count)
        {
            _probeCell[count + i * 6] = probeCell[count + i * 6];
        }
    }
}

void Grid::Mesh_Grid_4PortFilter_MS()
{
    nx = 53; ny = 89; nz = 13; PML = 5;

    qstop = 200;

    t = 6e-9; // Simulation time duration
    dt = 2e-12;
    tStep = t / dt;

    s = 1.5e11; fc = 10e9; td = 1 / (1.5 * fc); tc = 0.5e-9;

    scale = 230 / (s * (tStep - 1) * dt / 2); // Similar as in Myunghyun's paper

    pulseType = 1; // 1 - Gaussian, 2 - Gaussian Derivative, 3 - Modulated Gaussian

    _eps.resize(boost::extents[nx + 1][ny + 1][nz + 1]);
    _mu.resize(boost::extents[nx + 1][ny + 1][nz + 1]);

    _sigmax.resize(boost::extents[nx][ny][nz]);
    _sigmay.resize(boost::extents[nx][ny][nz]);
    _sigmaz.resize(boost::extents[nx][ny][nz]);
    _Jx.resize(boost::extents[nx + 1][ny][nz + 1]);
    _Jy.resize(boost::extents[nx + 1][ny][nz + 1]);
    _Jz.resize(boost::extents[nx + 1][ny + 1][nz]);
    _Rz.resize(boost::extents[nx + 1][ny + 1][nz]);

    _dxh = std::move(std::make_unique<double[]>(nx + 1));
    _dyh = std::move(std::make_unique<double[]>(ny + 1));
    _dzh = std::move(std::make_unique<double[]>(nz + 1));
    _dxe = std::move(std::make_unique<double[]>(nx + 1));
    _dye = std::move(std::make_unique<double[]>(ny + 1));
    _dze = std::move(std::make_unique<double[]>(nz + 1));

    for (int i = 0; i < PML; ++i)
    {
        _dxe[i] = 1e-3;
        _dye[i] = 1e-3;
    }

    // Substrate Edge Offset
    for (int i = PML; i < PML + 5; ++i) { _dxe[i] = 1e-3; }
    // Upper Resonator Block
    for (int i = PML + 5; i < PML + 9; ++i) { _dxe[i] = 0.52e-3; }
    _dxe[PML + 9] = 0.075e-3;
    _dxe[PML + 10] = 0.09e-3;
    _dxe[PML + 11] = 0.18e-3;
    _dxe[PML + 12] = 0.09e-3;
    _dxe[PML + 13] = 0.075e-3;
    for (int i = PML + 14; i < PML + 18; ++i) { _dxe[i] = 0.52e-3; }
    // Taper to middle resonator block
    _dxe[PML + 18] = 1.446e-3;
    // Middle Resonator Block
    for (int i = PML + 19; i < PML + 21; ++i) { _dxe[i] = 1e-3; }
    _dxe[PML + 21] = 0.4e-3;
    // Flip and copy the x directional grid
    for (int i = 0; i < nx / 2; ++i) { _dxe[nx - i - 1] = _dxe[i]; }

    // TL
    for (int i = PML; i < PML + 10; ++i) { _dye[i] = 1e-3; }
    // First middle resonator offset
    _dye[PML + 10] = 1.15625e-3;
    // Upper Resonator Block
    for (int i = PML + 11; i < PML + 13; ++i) { _dye[i] = 0.99375e-3; }
    _dye[PML + 13] = 0.5e-3;
    for (int i = PML + 14; i < PML + 16; ++i) { _dye[i] = 0.99375e-3; }
    // First Taper Transition
    _dye[PML + 16] = 1.5625e-3;
    for (int i = PML + 17; i < PML + 20; ++i) { _dye[i] = 1e-3; }
    _dye[PML + 20] = 1.575e-3;
    for (int i = PML + 21; i < PML + 23; ++i) { _dye[i] = 1.43125e-3; }
    // Upper Resonator Block
    for (int i = PML + 23; i < PML + 25; ++i) { _dye[i] = 0.99375e-3; }
    _dye[PML + 25] = 0.5e-3;
    for (int i = PML + 26; i < PML + 28; ++i) { _dye[i] = 0.99375e-3; }
    // Second Taper Transition
    for (int i = PML + 28; i < PML + 30; ++i) { _dye[i] = 1.43125e-3; }
    for (int i = PML + 30; i < PML + 35; ++i) { _dye[i] = 1e-3; }
    _dye[PML + 35] = 1.2125e-3;
    _dye[PML + 36] = 0.925e-3;
    // Third Resonator Block (Half)
    for (int i = PML + 37; i < PML + 39; ++i) { _dye[i] = 0.99375e-3; }
    _dye[PML + 39] = 0.5e-3;
    // Flip and copy the y directional grid
    for (int i = 0; i < ny / 2; ++i) { _dye[ny - i - 1] = _dye[i]; }

    for (int i = 0; i < 2; ++i) { _dze[i] = 1e-3; }
    for (int i = 2; i < 4; ++i) { _dze[i] = 0.3175e-3; }
    for (int i = 4; i < nz; ++i) { _dze[i] = 1e-3; }

    fill_3D_array<double_array_type>(_eps, nx + 1, ny + 1, nz + 1, eps0);
    fill_3D_array<double_array_type>(_mu, nx + 1, ny + 1, nz + 1, mu0);

    // Dielectric Substrate
    for (int i = PML; i < nx - PML; ++i) {
        for (int j = PML; j < ny - PML; ++j) {
            for (int k = 2; k < 4; ++k) {
                _eps[i][j][k] *= 10.2;
            }
        }
    }

    // Ground Plane
    for (int i = PML; i < nx - PML; ++i) {
        for (int j = PML; j < ny - PML + 1; ++j) {
            _sigmax[i][j][2] = 5.8e8;
        }
    }
    for (int i = PML; i < nx - PML + 1; ++i) {
        for (int j = PML; j < ny - PML; ++j) {
            _sigmay[i][j][2] = 5.8e8;
        }
    }

    /*** Sigma X Section ***/

    // TL
    for (int i = PML + 9; i < PML + 14; ++i) {
        for (int j = PML; j < ny - PML + 1; ++j) {
            _sigmax[i][j][4] = 5.8e8;
        }
    }

    /*** Sigma Y Section ***/

    // TL
    for (int i = PML + 9; i < PML + 14 + 1; ++i) {
        for (int j = PML; j < ny - PML; ++j) {
            _sigmay[i][j][4] = 5.8e8;
        }
    }

    for (int i = 0; i < nx + 1; ++i)
    {
        if (i == 0) {
            _dxh[i] = _dxe[i];
        }
        else
        {
            _dxh[i] = (_dxe[i] + _dxe[i - 1]) / 2;
        }
    }

    for (int i = 0; i < ny + 1; ++i)
    {
        if (i == 0) {
            _dyh[i] = _dye[i];
        }
        else
        {
            _dyh[i] = (_dye[i] + _dye[i - 1]) / 2;
        }
    }

    for (int i = 0; i < nz + 1; ++i)
    {
        if (i == 0) {
            _dzh[i] = _dze[i];
        }
        else
        {
            _dzh[i] = (_dze[i] + _dze[i - 1]) / 2;
        }
    }


    int jCellIndex[] = { PML + 10, PML + 15, PML + 1, PML + 1, 2 + 1, 3 + 1 }; // jCellIndex is assumed to be 1-indexed
    int jResistorIndex[] = { PML + 10, PML + 15, ny - PML + 1, ny - PML + 1, 2 + 1, 3 + 1}; // jResistorIndex is assumed to be 1-indexed
    _JCount = (jCellIndex[1] - jCellIndex[0] + 1) * (jCellIndex[3] - jCellIndex[2] + 1);// Number of parallel current sources
    num_Resistor = 1; num_probe = 2;
    _jResistorIndex = std::make_unique<int[]>(num_Resistor * 6);

    for (size_t i = 0; i < num_Resistor; ++i)
    {
        for (size_t count = 0; count < 6; ++count) _jResistorIndex[count + i * 6] = jResistorIndex[count + i * 6];
    }

    _jCellIndex = std::make_unique<int[]>(6);
    for (size_t i = 0; i < 6; i++) _jCellIndex[i] = jCellIndex[i];


    if (jDirecIndex[2] == 1) // jCellIndex/jResistorIndex is assumed to be 1-indexed
    {
        for (int i = _jCellIndex[0] - 1; i <= _jCellIndex[1] - 1; ++i) {
            for (int j = _jCellIndex[2] - 1; j <= _jCellIndex[3] - 1; ++j) {
                for (int k = _jCellIndex[4] - 1; k <= _jCellIndex[5] - 1; ++k) {
                    _Jz[i][j][k] = 1;
                }
            }
        }
        for (size_t count = 0; count < num_Resistor; count++)
        {
            for (int i = _jResistorIndex[0 + 6 * count] - 1; i <= _jResistorIndex[1 + 6 * count] - 1; ++i) {
                for (int j = _jResistorIndex[2 + 6 * count] - 1; j <= _jResistorIndex[3 + 6 * count] - 1; ++j) {
                    for (int k = _jResistorIndex[4 + 6 * count] - 1; k <= _jResistorIndex[5 + 6 * count] - 1; ++k) {
                        _Rz[i][j][k] = 1;
                    }
                }
            }
        }
    }

    // Probe Definition

    int probeCell[] = { PML + 10, PML + 15, PML + 6, PML + 6, 2 + 1, 3 + 1,
                        PML + 10, PML + 15, ny - PML - 6, ny - PML - 6, 2 + 1, 3 + 1 };

    _probeCell = std::make_unique<int[]>(num_probe * 6);
    for (size_t i = 0; i < num_probe; ++i)
    {
        for (size_t count = 0; count < 6; ++count)
        {
            _probeCell[count + i * 6] = probeCell[count + i * 6];
        }
    }
}

void Grid::Mesh_Visual()
{
    if (!glfwInit()) {
        std::cerr << "glfw init err" << std::endl;
        exit(-1);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#ifdef __APPLE__ // For MacOS only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Mesh View", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        std::cerr << "EEOR:Cannot set up Windows!!" << std::endl;
        exit(-1);
    }


    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, right_mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if (glewInit() != GLEW_OK)
    {
        std::cout << "Error!" << std::endl;
    }

    float mesh_ref = _dxe[0]; // Set dxe[0] value as the plot meshing reference. In practice, it can be any value

    std::unique_ptr<float[]> dx_nrm = std::make_unique<float[]>(nx);
    std::unique_ptr<float[]> dy_nrm = std::make_unique<float[]>(ny);
    std::unique_ptr<float[]> dz_nrm = std::make_unique<float[]>(nz);

    for (int i = 0; i < nx; ++i) {
        dx_nrm[i] = (_dxe[i] / mesh_ref) * x_scale;
    }

    for (int j = 0; j < ny; ++j) {
        dy_nrm[j] = (_dye[j] / mesh_ref) * y_scale;
    }

    for (int k = 0; k < nz; ++k) {
        dz_nrm[k] = (_dze[k] / mesh_ref) * z_scale;
    }


    glGenBuffers(1, &VBO);
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, pos_count * sizeof(float), positions, GL_STATIC_DRAW);

    // position attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, 0);

    // color attribute
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (void*)(3 * sizeof(float)));


    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    Shader source("./Basic.shader");

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        if (WireView)
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, wire_count * sizeof(unsigned int), wire_indices, GL_STATIC_DRAW);

        }
        else
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_count * sizeof(unsigned int), indices, GL_STATIC_DRAW);
        }

        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        /* Render here */
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //glUseProgram(shader);
        source.use();
        // create transformations
        glm::mat4 projection = glm::mat4(1.0f);

        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        source.setMat4("view", view);
        // note: currently we set the projection matrix each frame, but since the projection matrix rarely changes it's often best practice to set it outside the main loop only once.
        source.setMat4("projection", projection);

        glBindVertexArray(VAO);

        for (int i = 0; i < nx - 1; ++i) {

            if (i == 0)
                dx = 0.5f * dx_nrm[i] + 0.5f * dx_nrm[i];
            else
                dx = 0.5f * dx_nrm[i - 1] + 0.5f * dx_nrm[i];

            lx += dx;

            for (int j = 0; j < ny - 1; ++j) {

                if (j == 0)
                    dy = 0.5f * dy_nrm[j] + 0.5f * dy_nrm[j];
                else
                    dy = 0.5f * dy_nrm[j - 1] + 0.5f * dy_nrm[j];
                ly -= dy;

                for (int k = 1; k < nz - 1; ++k) {

                    if (k == 0)
                        dz = 0.5f * dz_nrm[k] + 0.5f * dz_nrm[k];
                    else
                        dz = 0.5f * dz_nrm[k - 1] + 0.5f * dz_nrm[k];
                    lz += dz;

                    glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
                    glm::mat4 RGB = glm::mat4(1.0f);

                    if (_sigmay[i][j][k+1] && _sigmax[i][j][k+1] && _sigmay[i + 1][j][k+1] && _sigmax[i][j + 1][k+1] && _eps[i][j][k + 1] != eps0) // Bottom GND mesh view
                    {
                        RGB = glm::translate(RGB, glm::vec3(1.0f, 1.0f, 0.0f));
                        source.setMat4("RGB", RGB);
                    }
                    else if (_eps[i][j][k - 1] != eps0 && _sigmay[i][j][k] && _sigmax[i][j][k] && _sigmay[i + 1][j][k] && _sigmax[i][j + 1][k]) // Top trace mesh view
                    {
                        RGB = glm::translate(RGB, glm::vec3(1.0f, 1.0f, 0.0f));
                        source.setMat4("RGB", RGB);
                    }
                    else
                    {
                        if ((_eps[i][j][k] != eps0) && (!_Rz[i][j][k]) && (!_Jz[i][j][k]) )
                        {
                            RGB = glm::translate(RGB, glm::vec3(137.0f / 255.0f, 137.0f / 255.0f, 137.0f / 255.0f));
                            source.setMat4("RGB", RGB);
                        }
                        else if ((_Jz[i][j][k])) // Excitation port resistor
                        {
                            if (_Jz[i + 1][j][k])
                            {
                                RGB = glm::translate(RGB, glm::vec3(1.0f, 0.0f, 0.0f));
                                source.setMat4("RGB", RGB);
                            }
                            else
                            {
                                RGB = glm::translate(RGB, glm::vec3(137.0f / 255.0f, 137.0f / 255.0f, 137.0f / 255.0f));
                                source.setMat4("RGB", RGB);
                            }
                        }
                        else if (_Rz[i][j][k]) // Termination port resistor
                        {
                            if (_Rz[i + 1][j][k])
                            {
                                RGB = glm::translate(RGB,glm::vec3(0.0f, 0.0f, 1.0f));
                                source.setMat4("RGB",RGB);
                            }
                            else if (_eps[i][j+1][k] != eps0)
                            {
                                RGB = glm::translate(RGB, glm::vec3(137.0f / 255.0f, 137.0f / 255.0f, 137.0f / 255.0f));
                                source.setMat4("RGB", RGB);
                            }
                            else continue; // Skip drawing the vacuum grid to save memory
                        }
                        else continue; // Skip drawing the vacuum grid to save memory
                    }

                    for (size_t count = 0; count < num_probe; ++count) // Probe Location
                    {
                        if (i >= _probeCell[count * 6] - 1 && i <= _probeCell[count * 6 + 1] - 1 &&
                            j >= _probeCell[count * 6 + 2] - 1 && j <= _probeCell[count * 6 + 3] - 1 &&
                            k >= _probeCell[count * 6 + 4] - 1 && k <= _probeCell[count * 6 + 5] - 1)
                        {
                            RGB = glm::translate(RGB, glm::vec3(0.0f, 1.0f, 1.0f));
                            source.setMat4("RGB", RGB);
                        }
                    }

                    model = glm::rotate(model, glm::radians(x_angle), glm::vec3(1.0f, 0.0f, 0.0f));
                    model = glm::rotate(model, glm::radians(y_angle), glm::vec3(0.0f, 1.0f, 0.0f));
                    model = glm::translate(model, glm::vec3(lx, lz, ly)); // Flip y - z axis due to RHS of openGL coordinate
                    model = glm::scale(model, glm::vec3(dx_nrm[i], dz_nrm[k], dy_nrm[j]));

                    source.setMat4("model", model);

                    if (WireView) glDrawElements(GL_LINES, wire_count, GL_UNSIGNED_INT, 0);
                    else glDrawElements(GL_TRIANGLES, indices_count, GL_UNSIGNED_INT, 0);

                }

                lz = z_start_pos; // Clear the lz buffer

            }

            ly = y_start_pos; // Clear the ly buffer
        }

        lx = x_start_pos;

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    source.Delete();
    glfwTerminate();

}

void LFDTD_Coe::Coe_SET()
{
    _nodeNum.resize(boost::extents[nx + 1][ny + 1][nz + 1][3]);

    _cex.resize(boost::extents[nx + 1][ny + 1][nz + 1]);
    _cey.resize(boost::extents[nx + 1][ny + 1][nz + 1]);
    _cez.resize(boost::extents[nx + 1][ny + 1][nz + 1]);

    _chx.resize(boost::extents[nx + 1][ny + 1][nz + 1]);
    _chy.resize(boost::extents[nx + 1][ny + 1][nz + 1]);
    _chz.resize(boost::extents[nx + 1][ny + 1][nz + 1]);

    _waveform = std::make_unique<double[]>(tStep);

    _hx.resize(boost::extents[nx][ny][nz]);
    _hy.resize(boost::extents[nx][ny][nz]);
    _hz.resize(boost::extents[nx][ny][nz]);
    _sumHx.resize(boost::extents[nx][ny][nz]);
    _sumHy.resize(boost::extents[nx][ny][nz]);
    _sumHz.resize(boost::extents[nx][ny][nz]);


    for (int i = 0; i < nx + 1; ++i) {
        for (int j = 0; j < ny + 1; ++j) {
            for (int k = 0; k < nz + 1; ++k) {
                _cex[i][j][k] = 2 / (s * _eps[i][j][k] * _dxh[i]);
                _cey[i][j][k] = 2 / (s * _eps[i][j][k] * _dyh[j]);
                _cez[i][j][k] = 2 / (s * _eps[i][j][k] * _dzh[k]);
                _chx[i][j][k] = 2 / (s * _mu[i][j][k] * _dxe[i]);
                _chy[i][j][k] = 2 / (s * _mu[i][j][k] * _dye[j]);
                _chz[i][j][k] = 2 / (s * _mu[i][j][k] * _dze[k]);
            }
        }
    }


    Rs = R0 * (_jCellIndex[1] - _jCellIndex[0] + 1) * (_jCellIndex[3] - _jCellIndex[2] + 1) / (_jCellIndex[5] - _jCellIndex[4] + 1); // Source Resistance
    if (num_Resistor)
    {
        Rl = R0 * (_jResistorIndex[1] - _jResistorIndex[0] + 1) * (_jResistorIndex[3] - _jResistorIndex[2] + 1) / (_jResistorIndex[5] - _jResistorIndex[4] + 1); // Load Resistance
    }

    switch (pulseType) // Waveform Definition
    {
    case 1:
        for (int i = 0; i < tStep; ++i) {
            _waveform[i] = pow(exp(1), (-pow(((dt * (i + 1) - tc) / td), 2)));
        }
        break;

    case 3:
        for (int i = 0; i < tStep; ++i)
        {
            double coe = -(dt * (i + 1) - tc) * (dt * (i + 1) - tc) / (2 * (tc * tc / 32));
            _waveform[i] = sin(2 * pi * fc * (dt * (i + 1) - tc)) * pow(exp(1), coe);
        }
        break;

    default:
        break;
    };

    for (int i = 0; i < nx + 1; ++i) {
        for (int j = 0; j < ny + 1; ++j) {
            for (int k = 0; k < nz + 1; ++k) {

                if (i != nx)
                {
                    _nodeNum[i][j][k][0] = Nnode;
                    Nnode++;
                }

                if (j != ny)
                {
                    _nodeNum[i][j][k][1] = Nnode;
                    Nnode++;
                }

                if (k != nz)
                {
                    _nodeNum[i][j][k][2] = Nnode;
                    Nnode++;
                }

            }
        }
    }

    printf("Laguerre-FDTD Meshing completed with total %d nodes.\n\n", Nnode);

}



