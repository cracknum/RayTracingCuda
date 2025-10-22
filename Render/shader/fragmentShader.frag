#version 330 core

in vec2 fTexCoord;

uniform sampler2D fTexture;
out vec4 fragColor;
void main()
{
    fragColor = texture(fTexture, fTexCoord);
}