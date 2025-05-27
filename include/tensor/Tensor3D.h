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
    Tensor3D(Float3D values, string operation = "", string name = "", bool parameter = false);
    Tensor3D(Eigen::Tensor<float, 3> tensor, string operation = "", string name = "", bool parameter = false);


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
    Tensor3D* operator+(TensorBase* other);
    Tensor3D* operator-();
    Tensor3D* operator-(TensorBase* other);
    Tensor3D* operator*(TensorBase* other);
    Tensor3D* operator*(Tensor0D* other);
    Tensor3D* pow(int other) override;

    Tensor2D* contract(TensorBase* other, int first_contract_dims, int second_contract_dims);
    Tensor3D* dot(Tensor3D* other);
    void printInfo() override;
};
