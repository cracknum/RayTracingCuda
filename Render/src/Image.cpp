#include "Image.hpp"

#include <QOpenGLFunctions_4_4_Core.h>

#include <iostream>
struct Image::Impl final
{
  int16_t mWidth;
  int16_t mHeight;
  GLuint mProgram;
  GLuint mVAO;
  GLuint mVBO;
  GLuint mEBO;
  GLuint mImageTexture;
  const unsigned char* mImage;

  bool mImageNeedUpdate;

  std::vector<GLfloat> mVertices;
  std::vector<GLuint> mElementIndices;

  Impl(int16_t width = 0, int16_t height = 0)
      : mWidth(width),
        mHeight(height),
        mProgram(0),
        mVAO(0),
        mVBO(0),
        mEBO(0),
        mImageNeedUpdate(false),
        mImageTexture(0)
  {
    mVertices =
        {-1.0f, -1.0f, 0.0f, 0.f, 0.f, -1.0f, 1.0f, 0.0f, 0.f, 1.f,
         1.0f, 1.0f, 0.0f, 1.f, 1.f, 1.0f, -1.0f, 0.0f, 1.f, 0.f};
    mElementIndices = {0, 1, 3, 1, 2, 3};
  }
};

Image::Image()
{
  mImpl = std::make_unique<Impl>();
}

Image::~Image()
{
  if (mImpl->mProgram)
  {
    mContext->glDeleteProgram(mImpl->mProgram);
  }

  if (mImpl->mVAO)
  {
    mContext->glDeleteVertexArrays(1, &mImpl->mVAO);
  }
  if (mImpl->mVBO)
  {
    mContext->glDeleteBuffers(1, &mImpl->mVBO);
  }
  if (mImpl->mEBO)
  {
    mContext->glDeleteBuffers(1, &mImpl->mEBO);
  }
}

void Image::render()
{
  bind();
  drawOnImage();
  unbind();
}

void Image::setImage(unsigned int width, unsigned int height,
                     const unsigned char* imageContent)
{
  mImpl->mImage = imageContent;

  mImpl->mWidth = width;
  mImpl->mHeight = height;
  mImpl->mImageNeedUpdate = true;
}

void Image::initialize(QOpenGLFunctions_4_4_Core *gl)
{
  SuperClass::initialize(gl);

  // initialize program;
  auto vertexShaderSource = readFile(SHADER_DIR "/vertexShader.vert");
  auto fragmentShaderSource = readFile(SHADER_DIR "/fragmentShader.frag");

  GLuint vertexShader = mContext->glCreateShader(GL_VERTEX_SHADER);
  GLuint fragmentShader = mContext->glCreateShader(GL_FRAGMENT_SHADER);
  const char *vShaderC = vertexShaderSource.c_str();
  mContext->glShaderSource(vertexShader, 1, &vShaderC, nullptr);

  const char *fShaderC = fragmentShaderSource.c_str();
  mContext->glShaderSource(fragmentShader, 1, &fShaderC, nullptr);

  mContext->glCompileShader(vertexShader);

  mContext->glCompileShader(fragmentShader);

  mImpl->mProgram = mContext->glCreateProgram();
  mContext->glAttachShader(mImpl->mProgram, vertexShader);
  mContext->glAttachShader(mImpl->mProgram, fragmentShader);
  mContext->glLinkProgram(mImpl->mProgram);

  int success = 0;
  char infoLog[512];

  mContext->glGetProgramiv(mImpl->mProgram, GL_LINK_STATUS, &success);

  if (!success)
  {
    mContext->glGetProgramInfoLog(mImpl->mProgram, 512, nullptr, infoLog);
    std::cerr << infoLog << std::endl;
    mContext->glDeleteProgram(mImpl->mProgram);
    mImpl->mProgram = 0;
  }

  mContext->glDeleteShader(vertexShader);
  mContext->glDeleteShader(fragmentShader);

  // initialize vao, vbo, ebo

  mContext->glGenVertexArrays(1, &mImpl->mVAO);
  mContext->glBindVertexArray(mImpl->mVAO);

  mContext->glGenBuffers(1, &mImpl->mVBO);
  mContext->glBindBuffer(GL_ARRAY_BUFFER, mImpl->mVBO);
  mContext->glBufferData(GL_ARRAY_BUFFER,
                         mImpl->mVertices.size() * sizeof(GLfloat),
                         mImpl->mVertices.data(), GL_STATIC_DRAW);

  mContext->glGenBuffers(1, &mImpl->mEBO);
  mContext->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mImpl->mEBO);
  mContext->glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                         mImpl->mElementIndices.size() * sizeof(GLuint),
                         mImpl->mElementIndices.data(), GL_STATIC_DRAW);

  mContext->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat),
                                  nullptr);
  mContext->glEnableVertexAttribArray(0);

  mContext->glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat),
                                  reinterpret_cast<const void *>(3 * sizeof(GLfloat)));
  mContext->glEnableVertexAttribArray(1);
  mContext->glBindVertexArray(0);

  mContext->glGenTextures(1, &mImpl->mImageTexture);
  mContext->glBindTexture(GL_TEXTURE_2D, mImpl->mImageTexture);
  mContext->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  mContext->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  mContext->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  mContext->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  mInitialized = true;
}

unsigned int Image::getImageTexture()
{
    return mImpl->mImageTexture;
}

void Image::drawOnImage()
{
  mContext->glViewport(0, 0, mImpl->mWidth, mImpl->mHeight);
  if (mImpl->mImageNeedUpdate)
  {
    uploadImage();
  }

  mContext->glActiveTexture(GL_TEXTURE0);
  mContext->glBindTexture(GL_TEXTURE_2D, mImpl->mImageTexture);

  auto textureId = mContext->glGetUniformLocation(mImpl->mProgram, "fTexture");
  mContext->glUniform1i(textureId, 0);

  mContext->glDrawElements(GL_TRIANGLES, mImpl->mElementIndices.size(),
                           GL_UNSIGNED_INT, nullptr);

}

void Image::bind()
{
  mContext->glBindVertexArray(mImpl->mVAO);
  mContext->glUseProgram(mImpl->mProgram);
}

void Image::unbind()
{
  mContext->glBindVertexArray(0);
  mContext->glUseProgram(0);
}

void Image::uploadImage()
{
  mContext->glBindTexture(GL_TEXTURE_2D, mImpl->mImageTexture);

  mContext->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mImpl->mWidth,
                         mImpl->mHeight, 0, GL_RGB, GL_UNSIGNED_BYTE,
                         mImpl->mImage);
  mContext->glBindTexture(GL_TEXTURE_2D, 0);
  mImpl->mImageNeedUpdate = false;
}
