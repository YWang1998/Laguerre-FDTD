//
// Created by yifanw on 9/6/24.
//


#include "Shader.h"

Shader::Shader(const std::string &filepath)
{
    std::ifstream stream(filepath);

    if (!stream.is_open())
    {
        Shader_STATUS = false;
        return;
    }

    std::string line;
    std::stringstream ss[2];
    ShaderType type = ShaderType::NONE;
    while (getline(stream, line))
    {
        if (line.find("#shader") != std::string::npos)
        {
            if (line.find("vertex")!= std::string::npos)
                type = ShaderType::VERTEX;
            else if (line.find("fragment")!= std::string::npos)
                type = ShaderType::FRAGMENT;
        }
        else
        {
            ss[(int)type] << line << "\n";
        }
    }
    VER_FRAG_SHADER.VertexSource = ss[0].str();
    VER_FRAG_SHADER.FragmentSource = ss[1].str();

    CreateShader(VER_FRAG_SHADER.VertexSource,VER_FRAG_SHADER.FragmentSource);
}

unsigned int Shader::CompileShader(unsigned int type, const std::string& source)
{
    unsigned int id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        int length;
        char infoLog[512];
        glGetShaderInfoLog(id, 512, NULL, infoLog);
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        std::cout << "ERROR::SHADER::"<<(type == GL_VERTEX_SHADER ? "vertex" : "fragment")<<"::COMPILATION_FAILED\n" << infoLog << std::endl;
        glDeleteShader(id);
        return 0;
    }

    return id;
}

void Shader::CreateShader(const std::string &vertexShader, const std::string &fragmentShader)
{
    ID = glCreateProgram();
    unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);

    glAttachShader(ID,vs);
    glAttachShader(ID,fs);
    glLinkProgram(ID);
    glValidateProgram(ID);

    glDeleteShader(vs);
    glDeleteShader(fs);
}

void Shader::use()
{
    glUseProgram(ID);
}

void Shader::Delete()
{
    glDeleteProgram(ID);
}
