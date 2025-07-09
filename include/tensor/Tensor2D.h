#pragma once
#include "TensorBase.h"
#include "Tensor0D.h"

class Tensor1D;  // Forward declarations
class Tensor3D;
class OptimizationFunc;

class Tensor2D : public TensorBase {
private:
    Eigen::Tensor<float, 2> data, grad;
    int rows, cols;
public:
    friend class Tensor0D;  // Friend declarations
    friend class Tensor1D;
    friend class Tensor3D;
    friend class OptimizationFunc;

    // Constructor declarations
    Tensor2D(Float2D values, string operation = "", bool parameter = false);
    Tensor2D(Eigen::Tensor<float, 2> tensor, string operation = "", bool parameter = false);

    ~Tensor2D();

    // Getters and setters
    Eigen::Tensor<float, 2> getData() const { return data; }
    Eigen::Tensor<float, 2> getGrad() const { return grad; }
    void setData(const Eigen::Tensor<float, 2>& newData) { data = newData; }
    void setGrad(const Eigen::Tensor<float, 2>& newGrad) { grad = newGrad; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    void setRows(int r) { rows = r; }
    void setCols(int c) { cols = c; }

    
    // Function declarations
    void backward() override;
    Tensor2D* operator+(TensorBase* other) override;
    Tensor2D* operator-() override;
    Tensor2D* operator-(TensorBase* other) override;
    Tensor2D* operator*(TensorBase* other) override;
    Tensor2D* operator*(Tensor0D* other) override;

    Tensor2D* pow(int other) override;
    Tensor2D* pow(double other) override;
    Tensor2D* pow(float other) override;
    Tensor2D* pow(Tensor0D* other) override;
    
    Tensor2D* contract(TensorBase* other, int first_contract_dims, int second_contract_dims);
    Tensor2D* dot(TensorBase* other);
    void printInfo() override;

    void printTensor2D(const Eigen::Tensor<float, 2>& tensor) const;

    friend std::ostream& operator<<(std::ostream& out, const Tensor2D* T){
        out << "Tensor 2D: \n\tOperation: " << T->getOperation() << "\t is parameter: " << T->isParameter() << '\n';
        out << "\tData: \n";
        T->printTensor2D(T->getData());
        out << "\tGrad: \n";
        T->printTensor2D(T->getGrad());
        return out;
    }
};
