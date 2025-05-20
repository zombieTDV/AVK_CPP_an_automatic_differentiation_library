#pragma once
#include "TensorBase.h"
#include "Tensor0D.h"

class Tensor1D : public TensorBase {
public:
    Eigen::Tensor<float, 1> data, grad;

    // Constructor declarations
    Tensor1D(Float1D values, string operation = "", string name = "", bool parameter = false);
    Tensor1D(Eigen::Tensor<float, 1> tensor, string operation = "", string name = "", bool parameter = false);

    // Function declarations
    void backward() override;
    Tensor1D* operator+(TensorBase* other);
    Tensor1D* operator-();
    Tensor1D* operator*(TensorBase* other);
    Tensor1D* operator*(Tensor0D* other);
    Tensor1D* operator-(TensorBase* other);
    void printInfo() override;
};
