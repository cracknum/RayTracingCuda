#ifndef OBJECT_EXPORT_OPENGL_H
#define OBJECT_EXPORT_OPENGL_H

#ifdef OBJECT_EXPORTS // 如果在项目中定义了此宏，则我们正在构建DLL版本
    #define OBJECT_API __declspec(dllexport)
#else
    #define OBJECT_API __declspec(dllimport)
#endif

#endif