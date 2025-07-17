#pragma once
#include "TensorBase.h"
#include "Tensor0D.h"
#include "Tensor2D.h"

class Tensor1D;  // Forward declarations
class OptimizationFunc;

class Tensor3D : public TensorBase {
private:
    Eigen::Tensor<float, 3> data, grad;
    int batch, rows, cols;
public:
    friend class Tensor0D;  // Friend declarations
    friend class Tensor1D;
    friend class Tensor2D;
    friend class OptimizationFunc;

    // Constructor declarations
    Tensor3D(Float3D values, string operation = "", bool parameter = false);
    Tensor3D(Eigen::Tensor<float, 3> tensor, string operation = "", bool parameter = false);

    ~Tensor3D();

    // Getters and setters
    Eigen::Tensor<float, 3> getData() const { return data; }
    Eigen::Tensor<float, 3> getGrad() const { return grad; }
    void setData(const Eigen::Tensor<float, 3>& newData) { data = newData; }
    void setGrad(const Eigen::Tensor<float, 3>& newGrad) { grad = newGrad; }
    int getBatch() const { return batch; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    void setBatch(int b) { batch = b; }
    void setRows(int r) { rows = r; }
    void setCols(int c) { cols = c; }

    
    // Function declarations
    void backward() override;
    Tensor3D* operator+(TensorBase* other) override;
    Tensor3D* operator-() override;
    Tensor3D* operator-(TensorBase* other) override;
    Tensor3D* operator*(TensorBase* other) override;
    Tensor3D* operator*(Tensor0D* other) override;

    Tensor3D* pow(int other) override;
    Tensor3D* pow(float other) override;
    Tensor3D* pow(double other) override;
    Tensor3D* pow(Tensor0D* other) override;

    Tensor0D* mean() override;
    Tensor0D* sum() override;

    Tensor2D* contract(Tensor2D* other, int first_contract_dims, int second_contract_dims);
    Tensor3D* dot(Tensor3D* other);
    void printInfo() override;    

    void applyGradientDescent(float learning_rate) override;

    void printTensor3D(const Eigen::Tensor<float, 3>& tensor) const;

    friend std::ostream& operator<<(std::ostream& out, const Tensor3D* T){
        out << "Tensor 3D: \n\tOperation: " << T->getOperation() << "\t is parameter: " << T->isParameter() << '\n';

        out << "\tData: \n";
        T->printTensor3D(T->getData());

        out << "\tGrad: \n";
        T->printTensor3D(T->getGrad());
        return out; 
    }
};
