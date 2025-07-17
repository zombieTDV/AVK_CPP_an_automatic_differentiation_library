#include "../../include/tensor/Tensor1D.h"

Tensor1D::Tensor1D(Float1D values, string operation, bool parameter) : 
    TensorBase(operation, parameter), 
    data(Eigen::Tensor<float, 1>(values.size()).setValues(values)),
    grad(Eigen::Tensor<float, 1>(values.size()).setZero())
{
    // this->grad.setZero();

    TensorBase::memoryUsage += sizeof(*this);
}

Tensor1D::Tensor1D(Eigen::Tensor<float, 1> tensor, string operation, bool parameter) : 
    TensorBase(operation, parameter), 
    data(tensor),
    grad(Eigen::Tensor<float, 1>(tensor).setZero())
{
    // this->grad.setZero();
    
    TensorBase::memoryUsage += sizeof(*this);
}

Tensor1D::~Tensor1D(){
    TensorBase::memoryUsage -= sizeof(*this);
}

void Tensor1D::backward() {
    vector<TensorBase*> visited;
    buildTopo(visited);

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

Tensor1D* Tensor1D::operator+(float other) {
    Tensor1D* output = new Tensor1D((this->data + other), "+");

    output->children = {this};

    output->backwardFn = [output, this] () {
        this->grad += output->grad;
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

Tensor1D* Tensor1D::operator-(TensorBase* other){
    return (*this) + (other->operator-());
}

Tensor1D* Tensor1D::operator-(float other){
    return (*this) + (-other);
}

Tensor1D* Tensor1D::operator/(TensorBase* other){
    return (*this) * other->pow(-1);
}
Tensor1D* operator/(float other){
    return (*this) * (1.0f / other);
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

Tensor1D* Tensor1D::pow(float other) {
    Tensor1D* output = new Tensor1D((this->data.pow(other)), "pow");
    output->children = {this};
    output->backwardFn = [output, this, other] () {
        this->grad += other * (this->data.pow(other - 1)) * output->grad;
    };
    return output;
}

Tensor1D* Tensor1D::pow(double other) {
    Tensor1D* output = new Tensor1D((this->data.pow(other)), "pow");
    output->children = {this};
    output->backwardFn = [output, this, other] () {
        this->grad += other * (this->data.pow(other - 1)) * output->grad;
    };
    return output;
}

Tensor1D* Tensor1D::pow(Tensor0D* other) {
    Tensor1D* output = this->pow(other->data(0));
    return output;
}

Tensor0D* Tensor1D::sum(){
    Tensor0D* output = new Tensor0D((this->data.sum()), "sum");
    output->children = {this};
    output->backwardFn = [output, this] () {
        this->grad += this->data.constant(1.0) * output->grad;
    };
    return output;
}

Tensor0D* Tensor1D::mean(){
    Tensor0D* output = this->sum() / this->data.size();
    return output;
}



void Tensor1D::applyGradientDescent(float learning_rate){
    if (parameter){
        data += -learning_rate * grad;
        grad.setZero();
    }
}

void Tensor1D::printTensor1D(const Eigen::Tensor<float, 1>& tensor) const {
    cout << "[ ";
    for (int i = 0; i < tensor.dimension(0); ++i) {
        cout << tensor(i) << " ";
    }
    cout << "]\n";
}

void Tensor1D::printInfo() {
    cout << ": \n" << "Data: \n";
    printTensor1D(this->data);
    cout << "Grad: \n";
    printTensor1D(this->grad);
}


 