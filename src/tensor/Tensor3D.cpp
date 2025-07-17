#include "../../include/tensor/Tensor3D.h"

Tensor3D::Tensor3D(Float3D values, string operation, bool parameter) :
    TensorBase(operation, parameter)
{
    this->batch = values.size();
    this->rows = values.begin()->size();
    this->cols = values.begin()->begin()->size();

    this->data = Eigen::Tensor<float, 3>(this->batch, this->rows, this->cols).setValues(values);
    this->grad = Eigen::Tensor<float, 3>(this->batch, this->rows, this->cols).setZero();

    TensorBase::memoryUsage += sizeof(*this);
}

Tensor3D::Tensor3D(Eigen::Tensor<float, 3> tensor, string operation, bool parameter) :
    TensorBase(operation, parameter),
    data(tensor), 
    grad(Eigen::Tensor<float, 3>(tensor).setZero())
{
    this->batch = this->data.dimension(0);
    this->rows = this->data.dimension(1);
    this->cols = this->data.dimension(2);

    TensorBase::memoryUsage += sizeof(*this);
}

Tensor3D::~Tensor3D(){
    TensorBase::memoryUsage -= sizeof(*this);
}

void Tensor3D::backward() {
    vector<TensorBase*> visited;
    buildTopo(visited);

    this->grad.setConstant(1);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->executeBackward();
    }
}

Tensor3D* Tensor3D::operator+(TensorBase* other) {
    Tensor3D* otherTensor = dynamic_cast<Tensor3D*>(other);
    Tensor3D* output = new Tensor3D((this->data + otherTensor->data), "+");

    output->children = {this, otherTensor};

    output->backwardFn = [output, this, otherTensor] () {
        this->grad += output->grad;
        otherTensor->grad += output->grad;
    };

    return output;
}

Tensor3D* Tensor3D::operator-() {
    Tensor0D* neg_one = new Tensor0D(-1.0f, "neg_one");
    Tensor3D* output = this->operator*(neg_one);
    return output;
}

Tensor3D* Tensor3D::operator-(TensorBase* other) {
    Tensor3D* otherTensor = dynamic_cast<Tensor3D*>(other);
    return this->operator+(otherTensor->operator-());
}

Tensor3D* Tensor3D::operator*(TensorBase* other) {
    Tensor3D* otherTensor = dynamic_cast<Tensor3D*>(other);
    Tensor3D* output = new Tensor3D((this->data * otherTensor->data), "*");

    output->children = {this, otherTensor};

    output->backwardFn = [output, this, otherTensor] () {
        this->grad += otherTensor->data * output->grad;
        otherTensor->grad += this->data * output->grad;
    };

    return output;
}

Tensor3D* Tensor3D::operator*(Tensor0D* other) {
    // Create a 3D tensor with the same value as the scalar
    Eigen::Tensor<float, 3> scalar_tensor(this->data.dimensions());
    scalar_tensor.setConstant(other->data(0));
    
    Tensor3D* output = new Tensor3D((this->data * scalar_tensor), "*");
    output->children = {this, other};

    output->backwardFn = [output, this, other] () {
        this->grad += other->data(0) * output->grad;
        other->grad += (this->data * output->grad).sum();
    };

    return output;
}

Tensor2D* Tensor3D::contract(Tensor2D* other, int first_contract_dims, int second_contract_dims) {
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(first_contract_dims, second_contract_dims)};

    // Tensor2D* otherTensor = dynamic_cast<Tensor2D*>(other);
    Tensor2D* output = new Tensor2D((this->data.contract(other->data, contract_dims)), "contract");

    output->children = {this, other};

    output->backwardFn = [output, this, other, first_contract_dims, second_contract_dims] () {
        Eigen::array<Eigen::IndexPair<int>, 1> dims1{Eigen::IndexPair<int>(first_contract_dims,first_contract_dims)};
        Eigen::array<Eigen::IndexPair<int>, 1> dims2{Eigen::IndexPair<int>(second_contract_dims, second_contract_dims)};

        this->grad += output->grad.contract(other->data, dims1);
        other->grad += this->data.contract(output->grad, dims2);
    };

    return output;
}

Tensor3D* Tensor3D::dot(Tensor3D* other) {
    if (this->batch != other->batch) {
        cout << "Batch dimension not matching in order to perform Batch-wise matrix multiplication.";
    }

    Tensor3D* output = new Tensor3D(Eigen::Tensor<float, 3>(this->batch, this->rows, other->cols).setZero());
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};
    
    for (int i = 0; i < this->data.dimension(0); ++i) {
        output->data.chip<0>(i) = this->data.chip<0>(i).contract(other->data.chip<0>(i), contract_dims);
    }

    output->children = {this, other};

    output->backwardFn = [output, this, other]() {
        for (int i = 0; i < this->data.dimension(0); ++i) {
            Eigen::array<Eigen::IndexPair<int>, 1> dims1 = {Eigen::IndexPair<int>(1, 1)};
            Eigen::array<Eigen::IndexPair<int>, 1> dims2 = {Eigen::IndexPair<int>(0, 0)};
            
            // dA = dC * B^T
            this->grad.chip<0>(i) += output->grad.chip<0>(i).contract(
                other->data.chip<0>(i).shuffle(Eigen::array<int, 2>{1, 0}),
                Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)}
            );
            // dB = A^T * dC
            other->grad.chip<0>(i) += this->data.chip<0>(i).contract(output->grad.chip<0>(i), dims2);
        }
    };

    return output;
}

Tensor3D* Tensor3D::pow(int other) {
    Tensor3D* output = new Tensor3D((this->data.pow(other)), "pow");
    output->children = {this};
    output->backwardFn = [output, this, other] () {
        this->grad += other * (this->data.pow(other - 1)) * output->grad;
    };
    return output;
}

Tensor3D* Tensor3D::pow(float other) {
    Tensor3D* output = new Tensor3D((this->data.pow(other)), "pow");
    output->children = {this};
    output->backwardFn = [output, this, other] () {
        this->grad += other * (this->data.pow(other - 1)) * output->grad;
    };
    return output;
}

Tensor3D* Tensor3D::pow(double other) {
    Tensor3D* output = new Tensor3D((this->data.pow(other)), "pow");
    output->children = {this};
    output->backwardFn = [output, this, other] () {
        this->grad += other * (this->data.pow(other - 1)) * output->grad;
    };
    return output;
}

Tensor3D* Tensor3D::pow(Tensor0D* other) {
    Tensor3D* output = this->pow(other->data(0));
    return output;
}




void Tensor3D::applyGradientDescent(float learning_rate){
    if (parameter){
        data += -learning_rate * grad;
        grad.setZero();
    }
}

void Tensor3D::printTensor3D(const Eigen::Tensor<float, 3>& tensor) const {
    // Print batch headers
    for (int b = 0; b < tensor.dimension(0); ++b) {
        cout << "Batch " << b << ":\t";
    }
    cout << "\n";
    // Print rows side by side
    for (int r = 0; r < tensor.dimension(1); ++r) {
        for (int b = 0; b < tensor.dimension(0); ++b) {
            for (int c = 0; c < tensor.dimension(2); ++c) {
                cout << tensor(b, r, c) << " ";
            }
            cout << "\t\t";
        }
        cout << "\n";
    }

    cout << '\n';
}

void Tensor3D::printInfo() {
    cout << ": \n" << "Data: \n";
    printTensor3D(this->data);
    cout << "Grad: \n";
    printTensor3D(this->grad);
}