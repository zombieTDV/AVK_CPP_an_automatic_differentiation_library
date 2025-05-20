#pragma once //Báo cho trình biên dịch không khai báo lớp này quá 1 lần.
#include "TensorBase.h"

class Tensor0D : public TensorBase {
public:
    Eigen::Tensor<float, 0> data, grad;

    // Constructor declarations
    Tensor0D(float data, string operation = "", string name = "", bool parameter = false);
    Tensor0D(Eigen::Tensor<float, 0> tensor, string operation = "", string name = "", bool parameter = false);

    // Function declarations
    void backward() override;
    Tensor0D* operator+(TensorBase* other);
    Tensor0D* operator*(TensorBase* other);
    Tensor0D* operator-(TensorBase* other);
    Tensor0D* pow(int other);
    Tensor0D* operator-();
    Tensor0D* operator*(Tensor0D* other);
    void printInfo() override;
};