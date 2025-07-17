#include "../../include/tensor/Tensor2D.h"

Tensor2D::Tensor2D(Float2D values, string operation, bool parameter) :
    TensorBase(operation, parameter)
{
    this->cols = values.begin()->size();
    this->rows = values.size();

    this->data = Eigen::Tensor<float, 2>(this->rows, this->cols).setValues(values);
    this->grad = Eigen::Tensor<float, 2>(this->rows, this->cols).setZero();

    TensorBase::memoryUsage += sizeof(*this);
}

Tensor2D::Tensor2D(Eigen::Tensor<float, 2> tensor, string operation, bool parameter) :
    TensorBase(operation, parameter),
    data(tensor), 
    grad(Eigen::Tensor<float, 2>(tensor).setZero())
{
    this->rows = this->data.dimension(0);
    this->cols = this->data.dimension(1);

    TensorBase::memoryUsage += sizeof(*this);
}

Tensor2D::~Tensor2D(){
    TensorBase::memoryUsage -= sizeof(*this);
}

void Tensor2D::backward() {
    vector<TensorBase*> visited;
    buildTopo(visited);

    this->grad.setConstant(1);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->executeBackward();
    }
}

Tensor2D* Tensor2D::operator+(TensorBase* other) {
    Tensor2D* otherTensor = dynamic_cast<Tensor2D*>(other);
    Tensor2D* output = new Tensor2D((this->data + otherTensor->data), "+");

    output->children = {this, otherTensor};

    output->backwardFn = [output, this, otherTensor] () {
        this->grad += output->grad;
        otherTensor->grad += output->grad;
    };

    return output;
}

Tensor2D* Tensor2D::operator-() {
    Tensor0D* neg_one = new Tensor0D(-1.0f, "neg_one");
    Tensor2D* output = this->operator*(neg_one);
    return output;
}

Tensor2D* Tensor2D::operator-(TensorBase* other) {
    Tensor2D* neg = new Tensor2D((this->data).setConstant(-1), "Neg");
    Tensor2D* output = this->operator+(*neg * other);
    return output;
}

Tensor2D* Tensor2D::operator*(TensorBase* other) {
    Tensor2D* otherTensor = dynamic_cast<Tensor2D*>(other);
    Tensor2D* output = new Tensor2D((this->data * otherTensor->data), "*");

    output->children = {this, otherTensor};

    output->backwardFn = [output, this, otherTensor] () {
        this->grad += otherTensor->data * output->grad;
        otherTensor->grad += this->data * output->grad;
    };

    return output;
}

Tensor2D* Tensor2D::operator*(Tensor0D* other) {
    // Create a 2D tensor with the same value as the scalar
    Eigen::Tensor<float, 2> scalar_tensor(this->data.dimensions());
    scalar_tensor.setConstant(other->data(0));

    Tensor2D* output = new Tensor2D((this->data * scalar_tensor), "*");
    output->children = {this, other};

    output->backwardFn = [output, this, other] () {
        this->grad += other->data(0) * output->grad;
        other->grad += (this->data * output->grad).sum();
    };

    return output;
}

Tensor2D* Tensor2D::contract(TensorBase* other, int first_contract_dims, int second_contract_dims) {
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(first_contract_dims, second_contract_dims)};

    Tensor2D* otherTensor = dynamic_cast<Tensor2D*>(other);
    Tensor2D* output = new Tensor2D((this->data.contract(otherTensor->data, contract_dims)), "contract");

    output->children = {this, otherTensor};

    output->backwardFn = [output, this, otherTensor, first_contract_dims, second_contract_dims] () {
        Eigen::array<Eigen::IndexPair<int>, 1> dims1{Eigen::IndexPair<int>(first_contract_dims,first_contract_dims)};
        Eigen::array<Eigen::IndexPair<int>, 1> dims2{Eigen::IndexPair<int>(second_contract_dims, second_contract_dims)};

        this->grad += output->grad.contract(otherTensor->data, dims1);
        otherTensor->grad += this->data.contract(output->grad, dims2);
    };

    return output;
}

Tensor2D* Tensor2D::dot(TensorBase* other) {
    Tensor2D* output = this->contract(other, 1, 0);
    return output;
}

Tensor2D* Tensor2D::pow(int other) {
    Tensor2D* output = new Tensor2D((this->data.pow(other)), "pow");
    output->children = {this};
    output->backwardFn = [output, this, other] () {
        this->grad += other * (this->data.pow(other - 1)) * output->grad;
    };
    return output;
}

Tensor2D* Tensor2D::pow(float other) {
    Tensor2D* output = new Tensor2D((this->data.pow(other)), "pow");
    output->children = {this};
    output->backwardFn = [output, this, other] () {
        this->grad += other * (this->data.pow(other - 1)) * output->grad;
    };
    return output;
}

Tensor2D* Tensor2D::pow(double other) {
    Tensor2D* output = new Tensor2D((this->data.pow(other)), "pow");
    output->children = {this};
    output->backwardFn = [output, this, other] () {
        this->grad += other * (this->data.pow(other - 1)) * output->grad;
    };
    return output;
}

Tensor2D* Tensor2D::pow(Tensor0D* other) {
    Tensor2D* output = this->pow(other->data(0));
    return output;
}


void Tensor2D::applyGradientDescent(float learning_rate){
    if (parameter){
        data += -learning_rate * grad;
        grad.setZero();
    }
}

void Tensor2D::printTensor2D(const Eigen::Tensor<float, 2>& tensor) const {
    for (int r = 0; r < tensor.dimension(0); ++r) {
        for (int c = 0; c < tensor.dimension(1); ++c) {
            cout << tensor(r, c) << " ";
        }
        cout << "\n";
    }
}

void Tensor2D::printInfo() {
    cout << ": \n" << "Data: \n";
    printTensor2D(this->data);
    cout << "Grad: \n";
    printTensor2D(this->grad);
}
