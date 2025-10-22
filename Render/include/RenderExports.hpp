#ifndef RENDER_EXPORT_OPENGL_H
#define RENDER_EXPORT_OPENGL_H

#ifdef RENDER_EXPORTS // 如果在项目中定义了此宏，则我们正在构建DLL版本
    #define RENDER_API __declspec(dllexport)
#else
    #define RENDER_API __declspec(dllimport)
#endif

#endif