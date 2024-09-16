#shader vertex
#version 330 core

layout (location = 0) in vec3 aPos;   // the position variable has attribute position 0
layout (location = 1) in vec3 aColor; // the color variable has attribute position 1

out vec3 ourColor; // output a color to the fragment shader

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    ourColor = aColor; // set ourColor to the input color we got from the vertex data
}

#shader fragment
#version 330 core

out vec4 FragColor;
in vec3 ourColor;

uniform mat4 RGB;
uniform float Opacity = 0.1f;

void main()
{
    FragColor = RGB * vec4(ourColor, 1.0f);
}
