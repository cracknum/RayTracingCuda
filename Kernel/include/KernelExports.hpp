#ifndef KERNEL_EXPORT_OPENGL_H
#define KERNEL_EXPORT_OPENGL_H

#ifdef KERNEL_EXPORTS // 如果在项目中定义了此宏，则我们正在构建DLL版本
    #define KERNEL_API __declspec(dllexport)
#else
    #define KERNEL_API __declspec(dllimport)
#endif

#endif