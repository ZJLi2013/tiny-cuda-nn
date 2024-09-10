// debug_config.h
#ifndef DEBUG_CONFIG_H
#define DEBUG_CONFIG_H

#define DEBUG_MODE 1
#include <iostream>

template<typename T>
inline void printCublasMatrix(const T* matrix, int rows, int cols, const char* matrix_name) {
    std::string root_path = "/workspace/tiny-rocm-nn/matrix_logs/";
    std::string local_path = std::string(matrix_name) + ".log"; 
    auto log_path = root_path + local_path ; 
    T* cpu_data = new T[rows * cols] ;
    cudaMemcpy(cpu_data, matrix, rows*cols*sizeof(T), cudaMemcpyDeviceToHost); 
    std::ofstream logfile(log_path, std::ios::app);
    auto status = logfile.is_open(); 
    logfile << "Open " << local_path << " for logging " << " [ " << rows << " , " << cols << " ]" << std::endl; 
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if( std::is_same<T, float>::value){
                logfile << static_cast<float>(cpu_data[i * cols + j]) << " ";
            }else if(std::is_same<T, __half>::value){
                logfile << __half2float(cpu_data[i * cols + j]) << " ";
            }else{
                std::cerr << "not supported data format in printCublasMatrix" << std::endl;  
            }
        }
        logfile << "\n" << std::endl ; 
    }
    logfile.close(); 
    delete[] cpu_data; 
}

template<typename T> 
inline void printCutlassMatrix(const T* matrix, int rows, int cols, const char* matrix_name) {
    std::string root_path = "/workspace/tiny-cuda-nn/matrix_logs/";
    std::string local_path = std::string(matrix_name) + ".log"; 
    auto log_path = root_path + local_path ; 
    T* cpu_data = new T[rows * cols] ;
    cudaMemcpy(cpu_data, matrix, rows*cols*sizeof(T), cudaMemcpyDeviceToHost); 
    std::ofstream logfile(log_path, std::ios::app);
    auto status = logfile.is_open(); 
    logfile << "Open " << local_path << " for logging " << " [ " << rows << " , " << cols << " ]" << std::endl; 
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if( std::is_same<T, float>::value){
                logfile << static_cast<float>(cpu_data[i * cols + j]) << " ";
            }else if(std::is_same<T, __half>::value){
                logfile << __half2float(cpu_data[i * cols + j]) << " ";
            }else{
                std::cerr << "not supported data format in printCublasMatrix" << std::endl;  
            }
        }
        logfile << "\n" <<  std::endl ; 
    }
    logfile.close(); 
    delete[] cpu_data; 
}

#endif // DEBUG_CONFIG_H
