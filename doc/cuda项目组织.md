# CUDA项目组织

## 问题
### 1. 将cuda的__device__代码声明在.cuh文件中实现在.cu文件中时出现无法找到链接
在cmake中开启分离编译
```c++
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```
### 2. 在完成问题1后如果将cuda代码封装成多个动态库会在链接时也会导致无法找到链接
将cuda的device代码封装为库时必须为静态库，使用动态库时nvlink找不到对应的实现
