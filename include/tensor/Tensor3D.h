#pragma once
#include "TensorBase.h"
#include "Tensor0D.h"
#include "Tensor2D.h"

class Tensor3D : public TensorBase {
public:
    Eigen::Tensor<float, 3> data, grad;
    int batch, rows, cols;

    // Constructor declarations
    Tensor3D(Float3D values, string operation = "", string name = "", bool parameter = false);
    Tensor3D(Eigen::Tensor<float, 3> tensor, string operation = "", string name = "", bool parameter = false);

    // Function declarations
    void backward() override;
    Tensor3D* operator+(TensorBase* other);
    Tensor3D* operator-();
    Tensor3D* operator-(TensorBase* other);
    Tensor3D* operator*(TensorBase* other);
    Tensor3D* operator*(Tensor0D* other);
    Tensor2D* contract(TensorBase* other, int first_contract_dims, int second_contract_dims);
    Tensor3D* dot(Tensor3D* other);
    void printInfo() override;
};
