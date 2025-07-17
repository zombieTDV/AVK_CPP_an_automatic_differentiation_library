#include "../../include/tensor/Tensor0D.h"

Tensor0D::Tensor0D(float data, string operation, bool parameter) : 
    TensorBase(operation, parameter), 
    data(Eigen::Tensor<float, 0>().setConstant(data))
{
    this->grad.setZero();

    TensorBase::memoryUsage += sizeof(*this);
}

Tensor0D::Tensor0D(Eigen::Tensor<float, 0> tensor, string operation, bool parameter) : 
    TensorBase(operation, parameter), 
    data(Eigen::Tensor<float, 0>(tensor))
{
    this->grad.setZero();

    TensorBase::memoryUsage += sizeof(*this);
}

Tensor0D::~Tensor0D(){
    TensorBase::memoryUsage -= sizeof(*this);
}

void Tensor0D::backward() {
    vector<TensorBase*> visited;
    buildTopo(visited);

    this->grad.setConstant(1);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->executeBackward();
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

Tensor0D* Tensor0D::pow(float other) {
    Tensor0D* output = new Tensor0D((this->data.pow(other)), "pow");
    output->children = {this};
    output->backwardFn = [output, this, other] () {
        this->grad += other * (this->data.pow(other - 1)) * output->grad;
    };
    return output;
}
Tensor0D* Tensor0D::pow(double other) {
    Tensor0D* output = new Tensor0D((this->data.pow(other)), "pow");

    output->children = {this};

    output->backwardFn = [output, this, other] () {
        this->grad += other * (this->data.pow(other - 1)) * output->grad;
    };

    return output;
}

Tensor0D* Tensor0D::pow(Tensor0D* other) {
    Tensor0D* output = this->pow(other->data(0));
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

Tensor0D* Tensor0D::operator/(float other) {
    if (other == 0.0f) {
        throw std::invalid_argument("Division by zero in Tensor0D::operator/(float)");
    }
    return new Tensor0D(this->data(0) / other, this->operation + "/", this->parameter);
}


void Tensor0D::applyGradientDescent(float learning_rate){
    if (parameter){
        data += -learning_rate * grad;
        grad.setZero();
    }
}

void Tensor0D::printInfo() {
    cout << "Data: " << this->data << ", " << " Grad: " << this->grad << '\n';
}
