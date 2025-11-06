#ifndef MATERIAL_EXPORT_OPENGL_H
#define MATERIAL_EXPORT_OPENGL_H

#ifdef MATERIAL_EXPORTS // 如果在项目中定义了此宏，则我们正在构建DLL版本
    #define MATERIAL_API __declspec(dllexport)
#else
    #define MATERIAL_API __declspec(dllimport)
#endif

#endif