#include <iostream>
#include <fstream>
#include <Eigen/Dense>

template <typename MatrixType>
void writeMatrixToTxt(const MatrixType & matrix, const std::string & filename, int precision = 6) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    Eigen::IOFormat fmt(
        Eigen::StreamPrecision,      
        Eigen::DontAlignCols,        
        " ",                         
        "\n",                        
        "", "",                      
        ""                           
    );

    file << matrix.format(fmt);  
    file.close();
}
