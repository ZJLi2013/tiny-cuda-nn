// debug_config.h
#ifndef DEBUG_CONFIG_H
#define DEBUG_CONFIG_H

#define DEBUG_MODE 1
#include <iostream>

inline void printCublasMatrix(const float* matrix, int rows, int cols, const char* matrix_name) {
    std::string root_path = "/workspace/tiny-rocm-nn/matrix_logs/";
    std::string local_path = std::string(matrix_name) + ".log"; 
    auto log_path = root_path + local_path ; 
    std::ofstream logfile(log_path, std::ios::app);
    auto status = logfile.is_open(); 
    std::cout << "Open " << local_path << " for logging " << std::endl; 
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
            logfile << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
        logfile << std::endl ; 
    }
    logfile.close(); 
}

#endif // DEBUG_CONFIG_H
