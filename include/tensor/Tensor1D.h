#pragma once
#include "TensorBase.h"
#include "Tensor0D.h"

class Tensor2D;  // Forward declarations
class Tensor3D;

class Tensor1D : public TensorBase {
private:
    Eigen::Tensor<float, 1> data, grad;
public:
    friend class Tensor0D;  // Friend declarations
    friend class Tensor2D;
    friend class Tensor3D;

    // Constructor declarations
    Tensor1D(Float1D values, string operation = "", string name = "", bool parameter = false);
    Tensor1D(Eigen::Tensor<float, 1> tensor, string operation = "", string name = "", bool parameter = false);

    // Getters and setters
    Eigen::Tensor<float, 1> getData() const { return data; }
    Eigen::Tensor<float, 1> getGrad() const { return grad; }
    void setData(const Eigen::Tensor<float, 1>& newData) { data = newData; }
    void setGrad(const Eigen::Tensor<float, 1>& newGrad) { grad = newGrad; }

    // Function declarations
    void backward() override;
    Tensor1D* operator+(TensorBase* other);
    Tensor1D* operator-();
    Tensor1D* operator*(TensorBase* other);
    Tensor1D* operator*(Tensor0D* other);
    Tensor1D* operator-(TensorBase* other);
    // Tensor1D* operator/(TensorBase* other) override;
    // Tensor1D* operator^(TensorBase* other) override;
    // Tensor1D* operator^(float other) override;
    // Tensor1D* operator^(int other) override;
    // Tensor1D* operator^(double other) override;
    // Tensor1D* operator^(long other) override;
    // Tensor1D* operator^(short other) override;
    // Tensor1D* pow(int other) override;

    void printInfo() override;
};
