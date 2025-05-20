#include "../../include/tensor/Tensor0D.h"

Tensor0D::Tensor0D(float data, string operation, string name, bool parameter) : 
    TensorBase(operation, name, parameter), 
    data(Eigen::Tensor<float, 0>().setConstant(data))
{
    this->grad.setZero();
}

Tensor0D::Tensor0D(Eigen::Tensor<float, 0> tensor, string operation, string name, bool parameter) : 
    TensorBase(operation, name, parameter), 
    data(Eigen::Tensor<float, 0>(tensor))
{
    this->grad.setZero();
}

void Tensor0D::backward() {
    vector<TensorBase*> visited;
    buildTopo(this->topo, visited);

    this->grad.setConstant(1);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->backwardFn();
    }
}

Tensor0D* Tensor0D::operator+(TensorBase* other) {
    Tensor0D* otherTensor = dynamic_cast<Tensor0D*>(other);
    Tensor0D* output = new Tensor0D((this->data + otherTensor->data), "+");

    output->children = {this, otherTensor};

    output->backwardFn = [output, this, otherTensor] () {
        this->grad += output->grad;
        otherTensor->grad += output->grad;
    };

    return output;
}

Tensor0D* Tensor0D::operator*(TensorBase* other) {
    Tensor0D* otherTensor = dynamic_cast<Tensor0D*>(other);
    Tensor0D* output = new Tensor0D((this->data * otherTensor->data), "*");

    output->children = {this, otherTensor};

    output->backwardFn = [output, this, otherTensor] () {
        this->grad += output->grad * otherTensor->data;
        otherTensor->grad += output->grad * this->data;
    };

    return output;
}

Tensor0D* Tensor0D::operator-(TensorBase* other) {
    Tensor0D* neg = new Tensor0D(-1, "Neg");
    Tensor0D* output = this->operator+(*neg * other);
    return output;
}

Tensor0D* Tensor0D::pow(int other) {
    Tensor0D* output = new Tensor0D((this->data.pow(other)), "pow");

    output->children = {this};

    output->backwardFn = [output, this, other] () {
        this->grad += other * (this->data.pow(other - 1)) * output->grad;
    };

    return output;
}

Tensor0D* Tensor0D::operator-() {
    Tensor0D* neg_one = new Tensor0D(-1.0f, "neg_one");
    Tensor0D* output = this->operator*(neg_one);
    return output;
}

Tensor0D* Tensor0D::operator*(Tensor0D* other) {
    Tensor0D* output = new Tensor0D((this->data * other->data), "*");
    output->children = {this, other};
    output->backwardFn = [output, this, other] () {
        this->grad += other->data * output->grad;
        other->grad += this->data * output->grad;
    };
    return output;
}

void Tensor0D::printInfo() {
    cout << this->name << ": " << "Data: " << this->data << ", " << " Grad: " << this->grad << '\n';
}
