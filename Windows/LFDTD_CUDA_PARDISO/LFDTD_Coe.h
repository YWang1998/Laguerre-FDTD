#pragma once

#include "FileReader.h"
#include "Shader.h"

static void right_mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

class Grid
{

public:

    Grid() = default; // Default constructor that does nothing
    void Mesh_Grid(); // Allocate mesh profile for grid assignment (Simple MS)
    void Mesh_Grid_2PortFilter(); // Allocate mesh profile for grid assignment (APS Filter)
    void Mesh_Grid_PatchAntennaArray(const std::string& InputFile); // Allocate mesh profile for grid assignment (IMS Patch Antenna Array)
    void Mesh_Grid_4PortFilter(); // Allocate mesh profile for grid assignment (4 Ports Filter)
    void Mesh_Grid_4PortFilter_MS(); // Allocate mesh profile for grid assignment (Simple Microstrip line for 4 Ports Filter case)

    void Mesh_Grid(const std::string& InputFile); // Allocate mesh profile for grid assignment based on an input file
    ~Grid() = default; // Destructor that delete all the dynamic allocated pointer variable
    void Mesh_Visual(); // Display the mesh profile using openGL

    // process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
    // ---------------------------------------------------------------------------------------------------------
    void processInput(GLFWwindow* window)
    {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS) // Reset View
        {
            cameraPos = cameraPos_Default;
            cameraFront = cameraFront_Default;
            cameraUp = cameraUp_Default;

            x_start_pos = x_pos;
            y_start_pos = y_pos;
            z_start_pos = z_pos;

            x_angle = 0.0f;
            y_angle = 0.0f;
        }

        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
            WireView = true;

        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
            WireView = false;


        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) // Left Click
        {
            //getting cursor position
            glfwGetCursorPos(window, &xpos, &ypos);

            // This will not move the object
            x_mouse_diff = static_cast<float>(xpos - xpos_ini);
            y_mouse_diff = static_cast<float>(ypos - ypos_ini);

            x_angle += y_mouse_diff * 0.1f;
            y_angle += x_mouse_diff * 0.1f;

            xpos_ini = xpos; ypos_ini = ypos;
        }

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) // Right Click
        {

            cameraSpeed = static_cast<float>(0.5 * deltaTime);

            //getting cursor position
            glfwGetCursorPos(window, &xpos, &ypos);

            x_mouse_diff = static_cast<float>(xpos - xpos_ini);
            y_mouse_diff = static_cast<float>(ypos - ypos_ini);

            cameraPos -= x_mouse_diff * glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;

            cameraPos += y_mouse_diff * cameraSpeed * cameraUp;

            xpos_ini = xpos; ypos_ini = ypos;


        }

    }

    // Whenever the mouse moves, get the cursor position
    // -------------------------------------------------------
    friend void right_mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
    {

        if ((glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE) \
            && (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)) // Right & Left Release
        {
            xpos = xposIn; ypos = yposIn;
            xpos_ini = xposIn; ypos_ini = yposIn;
        }
    }

    // glfw: whenever the mouse scroll wheel scrolls, this callback is called
    // ----------------------------------------------------------------------
    friend void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
    {
        cameraSpeed = static_cast<float>(2.5 * deltaTime);
        cameraPos += static_cast<float>(yoffset * cameraSpeed) * cameraFront;
    }

protected:

    double_array_type _eps;
    double_array_type _mu;

    double_array_type _sigmax;
    double_array_type _sigmay;
    double_array_type _sigmaz;
    double_array_type _Jz;
    double_array_type _Jx;
    double_array_type _Jy;
    double_array_type _Rz;

    std::unique_ptr<double[]> _dxh;
    std::unique_ptr<double[]> _dyh;
    std::unique_ptr<double[]> _dzh;
    std::unique_ptr<double[]> _dxe;
    std::unique_ptr<double[]> _dye;
    std::unique_ptr<double[]> _dze;

    int qstop, tStep;
    int nx, ny, nz, PML;
    int num_Resistor, _JCount, num_probe;
    int pulseType;

    double t, s, fc, td, tc, dt, scale;

    std::unique_ptr<int[]> _jCellIndex, _jResistorIndex, _probeCell;


private:

    // Window Width & Height
    static const unsigned int SCR_WIDTH{ 800 };
    static const unsigned int SCR_HEIGHT{ 600 };

    static const int pos_count{ 48 }, wire_count{ 48 }, indices_count{ 36 };

    unsigned int VBO, VAO, EBO;

    // Wire / patch view
    bool WireView = false;

    // camera
    static glm::vec3 cameraPos;
    static glm::vec3 cameraFront;
    static glm::vec3 cameraUp;

    const glm::vec3 cameraPos_Default = cameraPos;
    const glm::vec3 cameraFront_Default = cameraFront;
    const glm::vec3 cameraUp_Default = cameraUp;
    static float cameraSpeed;

    // timing
    static float deltaTime;	// time between current frame and last frame
    float lastFrame = 0.0f;

    // Cursor Position
    static double xpos, ypos;
    static double xpos_ini, ypos_ini;

    // Difference in mouse position

    float x_mouse_diff;
    float y_mouse_diff;

    // Initial start position of x/y/z coordinate position
    float x_pos = -3.5f;
    float y_pos = 2.0f;
    float z_pos = -2.0f;

    float x_start_pos = x_pos;
    float y_start_pos = y_pos;
    float z_start_pos = z_pos;

    float lx{ x_start_pos }, dx;
    float ly{ y_start_pos }, dy;
    float lz{ z_start_pos }, dz;

    // Scale factor of each cube
    float x_scale{ 0.2f }, y_scale{ 0.2f }, z_scale{ 0.2f };

    // Rotation angle for each cube
    float x_angle{ 0.0f }, y_angle{ 0.0f };

    float positions[pos_count] = {
            0.5f,  0.5f, -0.5f,0.0f, 0.0f,0.0f, // lower top right
            0.5f, -0.5f, -0.5f,0.0f, 0.0f,0.0f,// lower bottom right
            -0.5f, -0.5f, -0.5f,0.0f, 0.0f,0.0f,// lower bottom left
            -0.5f,  0.5f, -0.5f,0.0f, 0.0f,0.0f,// lower top left

            0.5f,  0.5f, 0.5f,0.0f, 0.0f,0.0f, // upper top right
            0.5f, -0.5f, 0.5f,0.0f, 0.0f,0.0f,// upper bottom right
            -0.5f, -0.5f, 0.5f,0.0f, 0.0f,0.0f,// upper bottom left
            -0.5f,  0.5f, 0.5f,0.0f, 0.0f,0.0f// upper top left
    };

    unsigned int wire_indices[wire_count] = {  // note that we start from 0!
            0,1,
            1,2,
            2,3,
            3,0,    // lower rectangle

            4,5,
            5,6,
            6,7,
            7,4,    // upper rectangle

            2,3,
            3,7,
            7,6,
            6,2,    // left rectangle

            0,1,
            1,5,
            5,4,
            4,0,    // right rectangle

            2,1,
            1,5,
            5,6,
            6,2,   // front rectangle

            0,3,
            3,7,
            7,4,
            4,0    // back rectangle

    };

    unsigned int indices[indices_count] = {  // note that we start from 0!
            0, 1, 3,   // lower first triangle
            1, 2, 3,    // lower second triangle

            4, 5, 7,   // upper first triangle
            5, 6, 7,    // upper second triangle

            2, 6, 3,   // left first triangle
            3, 7, 6,    // left second triangle

            0, 4, 5,   // right first triangle
            5, 1, 0,    // right second triangle

            2, 1, 5,   // front first triangle
            5, 6, 2,    // front second triangle

            3, 0, 4,   // back first triangle
            4, 7, 3,    // back second triangle
    };


};

class LFDTD_Coe : public Grid {

public:
    friend class LFDTD;
    LFDTD_Coe() = default; // Constructor
    void Coe_SET(); // Compute the Laguerre-FDTD coefficients
    ~LFDTD_Coe() = default;

private:

    node_array_type _nodeNum;

    double_array_type _cex;
    double_array_type _cey;
    double_array_type _cez;

    double_array_type _chx;
    double_array_type _chy;
    double_array_type _chz;

    double_array_type _hx;
    double_array_type _hy;
    double_array_type _hz;
    double_array_type _sumHx;
    double_array_type _sumHy;
    double_array_type _sumHz;

    std::unique_ptr<double[]> _waveform;

};