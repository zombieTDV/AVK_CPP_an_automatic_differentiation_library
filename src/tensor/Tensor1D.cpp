#include "../../include/tensor/Tensor1D.h"

Tensor1D::Tensor1D(Float1D values, string operation, string name, bool parameter) : 
    TensorBase(operation, name, parameter), 
    data(Eigen::Tensor<float, 1>(values.size()).setValues(values)),
    grad(Eigen::Tensor<float, 1>(values.size()).setZero())
{}

Tensor1D::Tensor1D(Eigen::Tensor<float, 1> tensor, string operation, string name, bool parameter) : 
    TensorBase(operation, name, parameter), 
    data(tensor),
    grad(Eigen::Tensor<float, 1>(tensor).setZero())
{}

void Tensor1D::backward() {
    vector<TensorBase*> visited;
    buildTopo(this->topo, visited);

    this->grad.setConstant(1);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->executeBackward();
    }
}

Tensor1D* Tensor1D::operator+(TensorBase* other) {
    Tensor1D* otherTensor = dynamic_cast<Tensor1D*>(other);
    Tensor1D* output = new Tensor1D((this->data + otherTensor->data), "+");

    output->children = {this, otherTensor};

    output->backwardFn = [output, this, otherTensor] () {
        this->grad += output->grad;
        otherTensor->grad += output->grad;
    };

    return output;
}

Tensor1D* Tensor1D::operator-() {
    Tensor0D* neg_one = new Tensor0D(-1.0f, "neg_one");
    Tensor1D* output = this->operator*(neg_one);
    return output;
}

Tensor1D* Tensor1D::operator*(TensorBase* other) {
    Tensor1D* otherTensor = dynamic_cast<Tensor1D*>(other);
    Tensor1D* output = new Tensor1D((this->data * otherTensor->data), "*");

    output->children = {this, otherTensor};

    output->backwardFn = [output, this, otherTensor] () {
        this->grad += output->grad * otherTensor->data;
        otherTensor->grad += output->grad * this->data;
    };

    return output;
}

Tensor1D* Tensor1D::operator*(Tensor0D* other) {
    // Create a 1D tensor with the same value as the scalar
    Eigen::Tensor<float, 1> scalar_tensor(this->data.dimensions());
    scalar_tensor.setConstant(other->getData()(0));
    
    Tensor1D* output = new Tensor1D((this->data * scalar_tensor), "*");
    output->children = {this, other};

    output->backwardFn = [output, this, other] () {
        this->grad += other->getData()(0) * output->grad;
        other->grad += (this->data * output->grad).sum();
    };

    return output;
}

Tensor1D* Tensor1D::operator-(TensorBase* other) {
    Tensor1D* neg = new Tensor1D((this->data).setConstant(-1), "Neg");
    Tensor1D* output = this->operator+(*neg * other);
    return output;
}

Tensor1D* Tensor1D::pow(int other) {
    Tensor1D* output = new Tensor1D((this->data.pow(other)), "pow");
    output->children = {this};
    output->backwardFn = [output, this, other] () {
        this->grad += other * (this->data.pow(other - 1)) * output->grad;
    };
    return output;
}

void Tensor1D::printInfo() {
    cout << this->name << ": \n" << "Data: \n" << this->data << '\n' << "Grad: \n" << this->grad << '\n';
}
 