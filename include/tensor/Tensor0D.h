#pragma once //Báo cho trình biên dịch không khai báo lớp này quá 1 lần.
#include "TensorBase.h"

class Tensor1D;  // Forward declarations
class Tensor2D;
class Tensor3D;
class OptimizationFunc;

class Tensor0D : public TensorBase {
private:
    Eigen::Tensor<float, 0> data, grad;
public:
    friend class Tensor1D;  // Friend declarations
    friend class Tensor2D;
    friend class Tensor3D;
    friend class OptimizationFunc;
    
    // Constructor declarations
    Tensor0D(float data, string operation = "", string name = "", bool parameter = false);
    Tensor0D(Eigen::Tensor<float, 0> tensor, string operation = "", string name = "", bool parameter = false);


    // Getters and setters
    Eigen::Tensor<float, 0> getData() const { return data; }
    Eigen::Tensor<float, 0> getGrad() const { return grad; }
    void setData(const Eigen::Tensor<float, 0>& newData) { data = newData; }
    void setGrad(const Eigen::Tensor<float, 0>& newGrad) { grad = newGrad; }

    
    // Function declarations
    void backward() override;
    Tensor0D* operator+(TensorBase* other) override;
    Tensor0D* operator*(TensorBase* other) override;
    Tensor0D* operator-(TensorBase* other) override;
    Tensor0D* pow(int other) override;
    Tensor0D* operator-() override;
    
    Tensor0D* operator*(Tensor0D* other);
    void printInfo() override;
};