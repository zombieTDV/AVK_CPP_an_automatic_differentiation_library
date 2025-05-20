#pragma once
#include "TensorBase.h"
#include "Tensor0D.h"

class Tensor2D : public TensorBase {
public:
    Eigen::Tensor<float, 2> data, grad;
    int rows, cols;

    // Constructor declarations
    Tensor2D(Float2D values, string operation = "", string name = "", bool parameter = false);
    Tensor2D(Eigen::Tensor<float, 2> tensor, string operation = "", string name = "", bool parameter = false);

    // Function declarations
    void backward() override;
    Tensor2D* operator+(TensorBase* other);
    Tensor2D* operator-();
    Tensor2D* operator*(TensorBase* other);
    Tensor2D* operator*(Tensor0D* other);
    Tensor2D* contract(TensorBase* other, int first_contract_dims, int second_contract_dims);
    Tensor2D* dot(TensorBase* other);
    void printInfo() override;
};
