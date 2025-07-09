#pragma once
#include "TensorBase.h"
#include "Tensor0D.h"

class Tensor2D;  // Forward declarations
class Tensor3D;
class OptimizationFunc;

class Tensor1D : public TensorBase {
private:
    Eigen::Tensor<float, 1> data, grad;
public:
    friend class Tensor0D;  // Friend declarations
    friend class Tensor2D;
    friend class Tensor3D;
    friend class OptimizationFunc;

    // Constructor declarations
    Tensor1D(Float1D values, string operation = "", bool parameter = false);
    Tensor1D(Eigen::Tensor<float, 1> tensor, string operation = "", bool parameter = false);

    ~Tensor1D();
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
    Tensor1D* pow(int other) override;
    Tensor1D* pow(double other) override;
    Tensor1D* pow(float other) override;
    Tensor1D* pow(Tensor0D* other) override;

    void printInfo() override;

    void printTensor1D(const Eigen::Tensor<float, 1>& tensor) const;

    friend std::ostream& operator<<(std::ostream& out, const Tensor1D* T){
        out << "Tensor 1D: \n\tOperation: " << T->getOperation() << "\t is parameter: " << T->isParameter() << '\n';
        out << "\tData: ";
        T->printTensor1D(T->getData());
        out << "\tGrad: ";
        T->printTensor1D(T->getGrad());
        return out;
    }
};
