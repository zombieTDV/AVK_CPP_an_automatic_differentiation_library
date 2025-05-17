// #include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"
#include <iostream>
// #include <variant>
// #include <memory>

using std::vector;
using std::string;
using std::cout;

// using std::shared_ptr;
// using std::make_shared;

using Float1D = std::initializer_list<float>;
using Float2D = std::initializer_list<std::initializer_list<float>>;
using Float3D = std::initializer_list<std::initializer_list<std::initializer_list<float>>>;

using Pair1D = Eigen::array<Eigen::IndexPair<int>, 1>;

class TensorBase{
public:
    string operation;
    string name;
    bool parameter = false;

    vector<TensorBase*> children;
    vector<TensorBase*> topo;

    std::function<void()> backwardFn = [](){};

    TensorBase(string operation = "", string name = "", bool parameter = false) : operation(operation), name(name), parameter(parameter){}

    void buildTopo(vector<TensorBase*> &topo, vector<TensorBase*> &visited){
        if (std::find(visited.begin(), visited.end(), this) == visited.end()){
            visited.push_back(this);
            for (TensorBase* child : children){
                child->buildTopo(topo, visited);
            }
            topo.push_back(this);
        }
    }

    void deleteTopo(){
        for (TensorBase* pTensor : this->topo){
            if (pTensor->parameter != true){
                delete(pTensor);
            }else{
                continue;
            }
        }
    }

    virtual void backward(){}
    virtual void printInfo(){}

    void setName(string name){this->name = name;}

    void setCleaned(bool parameter){this->parameter = parameter;}
};

//------------------------------------------------------------

class Tensor0D : public TensorBase {
public:
    Eigen::Tensor<float, 0> data, grad;

    Tensor0D(float data, string operation = "", string name = "", bool parameter = false) : 
        TensorBase(operation, name, parameter), 
        data(Eigen::Tensor<float, 0> ().setConstant(data))
        {
            this->grad.setZero();
        }

    Tensor0D(Eigen::Tensor<float, 0> tensor, string operation = "", string name = "", bool parameter = false) : 
        TensorBase(operation, name, parameter), 
        data(Eigen::Tensor<float, 0> (tensor))
        {
            this->grad.setZero();
        }

    void backward() override{
        vector<TensorBase*> visited;
        buildTopo(this->topo, visited);

        this->grad.setConstant(1);

        for (auto it = topo.rbegin(); it != topo.rend(); ++it){
            (*it)->backwardFn();
    }
}

    Tensor0D* operator+(TensorBase* other){
        Tensor0D* otherTensor = dynamic_cast<Tensor0D*>(other);
        Tensor0D* output = new Tensor0D((this->data + otherTensor->data), "+");

        output->children = {this, otherTensor};

        output->backwardFn = [output, this, otherTensor] () {
            this->grad += output->grad;
            otherTensor->grad += output->grad;
        };

        return output;
    }

    Tensor0D* operator*(TensorBase* other){
        Tensor0D* otherTensor = dynamic_cast<Tensor0D*>(other);
        Tensor0D* output = new Tensor0D((this->data * otherTensor->data), "*");

        output->children = {this, otherTensor};

        output->backwardFn = [output, this, otherTensor] () {
            this->grad += output->grad * otherTensor->data;
            otherTensor->grad += output->grad * this->data;
        };

        return output;
    }

    Tensor0D* operator-(TensorBase* other){
        Tensor0D* neg = new Tensor0D(-1, "Neg");
        Tensor0D* output = this->operator+(*neg * other);

        return output;
    }

    Tensor0D* pow(int other){
        Tensor0D* output = new Tensor0D((this->data.pow(other)), "pow");

        output->children = {this};

        output->backwardFn = [output, this, other] () {
            this->grad += other * (this->data.pow(other - 1)) * output->grad;
        };

        return output;
    }


    void printInfo() override{
        cout << this->name << ": " << "Data: " << this->data << ", " << " Grad: " << this->grad << '\n';
    }
};

//------------------------------------------------------------

class Tensor1D : public TensorBase {
public:
    Eigen::Tensor<float, 1> data, grad;

    Tensor1D(Float1D values, string operation = "", string name = "", bool parameter = false) : 
        TensorBase(operation, name, parameter), 
        data(Eigen::Tensor<float, 1> (values.size()).setValues(values)),
        grad(Eigen::Tensor<float, 1> (values.size()).setZero()){}
 
    Tensor1D(Eigen::Tensor<float, 1> tensor, string operation = "", string name = "", bool parameter = false) : 
        TensorBase(operation, name, parameter), 
        data(tensor),
        grad(Eigen::Tensor<float, 1> (tensor).setZero()){}


    void backward() override{
        vector<TensorBase*> visited;
        buildTopo(this->topo, visited);

        this->grad.setConstant(1);

        for (auto it = topo.rbegin(); it != topo.rend(); ++it){
            (*it)->backwardFn();
    }
}

    Tensor1D* operator+(TensorBase* other){
        Tensor1D* otherTensor = dynamic_cast<Tensor1D*>(other);
        Tensor1D* output = new Tensor1D((this->data + otherTensor->data), "+");

        output->children = {this, otherTensor};

        output->backwardFn = [output, this, otherTensor] () {
            this->grad += output->grad;
            otherTensor->grad += output->grad;
        };

        return output;
    }

    Tensor1D* operator*(TensorBase* other){
        Tensor1D* otherTensor = dynamic_cast<Tensor1D*>(other);
        Tensor1D* output = new Tensor1D((this->data * otherTensor->data), "*");

        output->children = {this, otherTensor};

        output->backwardFn = [output, this, otherTensor] () {
            this->grad += output->grad * otherTensor->data;
            otherTensor->grad += output->grad * this->data;
        };

        return output;
    }

    Tensor1D* operator-(TensorBase* other){
        Tensor1D* neg = new Tensor1D((this->data).setConstant(-1), "Neg");
        Tensor1D* output = this->operator+(*neg * other);

        return output;
    }


    void printInfo() override{
        cout << this->name << ": \n" << "Data: \n" << this->data << '\n' << "Grad: \n" << this->grad << '\n';
    }
};

//------------------------------------------------------------

class Tensor2D : public TensorBase{
public:
    Eigen::Tensor<float, 2> data, grad;
    int rows, cols;

    Tensor2D(Float2D values, string operation = "", string name = "", bool parameter = false) :
        TensorBase(operation, name, parameter){
            this->cols = values.begin()->size();
            this->rows = values.size();

            this->data = Eigen::Tensor<float, 2> (this->rows, this->cols).setValues(values);
            this->grad = Eigen::Tensor<float, 2> (this->rows, this->cols).setZero();
        }

    Tensor2D(Eigen::Tensor<float, 2> tensor, string operation = "", string name = "", bool parameter = false) :
        TensorBase(operation, name, parameter),
        data(tensor), 
        grad(Eigen::Tensor<float, 2> (tensor).setZero()){
            this->rows = this->data.dimension(0);
            this->cols = this->data.dimension(1);
        }   

    void backward() override{
        vector<TensorBase*> visited;
        buildTopo(this->topo, visited);

        this->grad.setConstant(1);

        for (auto it = topo.rbegin(); it != topo.rend(); ++it){
            (*it)->backwardFn();
    }}

    Tensor2D* operator+(TensorBase* other){
        Tensor2D* otherTensor = dynamic_cast<Tensor2D*>(other);
        Tensor2D* output = new Tensor2D((this->data + otherTensor->data), "+");

        output->children = {this, otherTensor};

        output->backwardFn = [output, this, otherTensor] () {
            this->grad += output->grad;
            otherTensor->grad += output->grad;
        };

        return output;
    }

    Tensor2D* operator*(TensorBase* other){
        Tensor2D* otherTensor = dynamic_cast<Tensor2D*>(other);
        Tensor2D* output = new Tensor2D((this->data * otherTensor->data), "*");

        output->children = {this, otherTensor};
    
        output->backwardFn = [output, this, otherTensor] () {
            this->grad += otherTensor->data * output->grad;
            otherTensor->grad += this->data * output->grad;
        };

        return output;
    }

    Tensor2D* contract(TensorBase* other, int first_contract_dims, int second_contract_dims){
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

    Tensor2D* dot(TensorBase* other){
        Tensor2D* output = this->contract(other, 1, 0);

        output->setName("dot");

        return output;
    }


    void printInfo() override{
        cout << this->name << ": \n" << "Data: \n" << this->data << '\n' << "Grad: \n" << this->grad << '\n';
    }


};

//------------------------------------------------------------

class Tensor3D : public TensorBase{
public:
    Eigen::Tensor<float, 3> data, grad;
    int batch, rows, cols;

    Tensor3D(Float3D values, string operation = "", string name = "", bool parameter = false) :
        TensorBase(operation, name, parameter){
            this->batch = values.size();
            this->rows = values.begin()->size();
            this->cols = values.begin()->begin()->size();

            this->data = Eigen::Tensor<float, 3>(this->batch, this->rows, this->cols).setValues(values);
            this->grad = Eigen::Tensor<float, 3>(this->batch, this->rows, this->cols).setZero();
        }

    Tensor3D(Eigen::Tensor<float, 3> tensor, string operation = "", string name = "", bool parameter = false) :
        TensorBase(operation, name, parameter),
        data(tensor), 
        grad(Eigen::Tensor<float, 3> (tensor).setZero()){
            this->batch = this->data.dimension(0);
            this->rows = this->data.dimension(1);
            this->cols = this->data.dimension(2);
        }   

    void backward() override{
        vector<TensorBase*> visited;
        buildTopo(this->topo, visited);

        this->grad.setConstant(1);

        for (auto it = topo.rbegin(); it != topo.rend(); ++it){
            (*it)->backwardFn();
    }}

    Tensor3D* operator+(TensorBase* other){
        Tensor3D* otherTensor = dynamic_cast<Tensor3D*>(other);
        Tensor3D* output = new Tensor3D((this->data + otherTensor->data), "+");

        output->children = {this, otherTensor};

        output->backwardFn = [output, this, otherTensor] () {
            this->grad += output->grad;
            otherTensor->grad += output->grad;
        };

        return output;
    }

    Tensor3D* operator*(TensorBase* other){
        Tensor3D* otherTensor = dynamic_cast<Tensor3D*>(other);
        Tensor3D* output = new Tensor3D((this->data * otherTensor->data), "*");

        output->children = {this, otherTensor};
    
        output->backwardFn = [output, this, otherTensor] () {
            this->grad += otherTensor->data * output->grad;
            otherTensor->grad += this->data * output->grad;
        };

        return output;
    }

    Tensor3D* contract(TensorBase* other, int first_contract_dims, int second_contract_dims){
        Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(first_contract_dims, second_contract_dims)};

        Tensor3D* otherTensor = dynamic_cast<Tensor3D*>(other);
        Tensor3D* output = new Tensor3D((this->data.contract(otherTensor->data, contract_dims)), "contract");

        output->children = {this, otherTensor};
    
        output->backwardFn = [output, this, otherTensor, first_contract_dims, second_contract_dims] () {
            Eigen::array<Eigen::IndexPair<int>, 1> dims1{Eigen::IndexPair<int>(first_contract_dims,first_contract_dims)};
            Eigen::array<Eigen::IndexPair<int>, 1> dims2{Eigen::IndexPair<int>(second_contract_dims, second_contract_dims)};

            this->grad += output->grad.contract(otherTensor->data, dims1);
            otherTensor->grad += this->data.contract(output->grad, dims2);
        };

        return output;
    }

    Tensor3D* dot(TensorBase* other){
        Tensor3D* otherTensor = dynamic_cast<Tensor3D*>(other);
        if (!otherTensor) {
            cout << "Invalid type for dot operation";
            return nullptr;
        }
        if (this->batch != otherTensor->batch){
            cout << "Batch dimension not matching in order to perform dot.";
        }
        return this->contract(otherTensor, 2, 1);
    }


    void printInfo() override{
        cout << this->name << ": \n" << "Data: \n" << this->data << '\n' << "Grad: \n" << this->grad << '\n';
    }


};
//------------------------------------------------------------

class MeanSquaredErrorLoss{
public:
    Tensor0D* Tensor0D_Loss(Tensor0D* true_y, Tensor0D* predicted_y){
        Tensor0D* loss = (*true_y - predicted_y)->pow(2);
        return loss;
    }
};

class OptimizationFunc{
public:

    OptimizationFunc() {};

    void gradientDescent(vector<TensorBase*> topo, float learning_rate){
        for (auto it = topo.rbegin(); it != topo.rend(); ++it){
            Tensor0D* new_it = dynamic_cast<Tensor0D*>(*it);
            if ((*new_it).parameter){
                (*new_it).data += -learning_rate * (*new_it).grad;
                (*new_it).grad.setZero();
        }}  
    }
};
int main(){
    //-------------------------------------- Tensor0D testing ground
    // MeanSquaredErrorLoss Loss_func;
    // OptimizationFunc Opti_func;
    // Tensor0D* A0 = new Tensor0D(5, "", "A0", true);
    // Tensor0D* B0 = new Tensor0D(3, "", "B0", true);

    // for (int i = 0; i < 10; i++){
    //     Tensor0D* Y_hat = new Tensor0D(15, "", "Y-hat");
    //     Tensor0D* C0 = Loss_func.Tensor0D_Loss((*A0 + B0), Y_hat);
    //     C0->setName("C0");

    //     C0->backward();

    //     // A0->printInfo();
    //     // B0->printInfo();
    //     // C0->printInfo();

    //     Opti_func.gradientDescent(C0->topo, 0.1);

    //     A0->printInfo();
    //     B0->printInfo();
    //     C0->printInfo();

    //     cout << "Loss: " << C0->data << '\n';

    //     C0->deleteTopo();

    // }
    

    //-------------------------------------- Tensor1D testing ground
    // Tensor1D* A1 = new Tensor1D({3, 5}, "", "A1");
    // Tensor1D* B1 = new Tensor1D({6, 9}, "", "B1");

    // Tensor1D* C1 = *A1 * B1;
    // C1->setName("C1");

    // C1->backward();

    // A1->printInfo();
    // B1->printInfo();
    // C1->printInfo();

    // C1->deleteTopo();

    // //-------------------------------------- Tensor2D testing ground
    // Tensor2D* A2 = new Tensor2D({{1, 2}, 
    //                             {3, 4}}, "", "A2");
    // Tensor2D* B2 = new Tensor2D({{5, 6, 7}, 
    //                                 {8, 9, 10}}, "", "B2");

    // Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
    //     Eigen::IndexPair<int>(1, 0)
    // };

    // Tensor2D* C2 = A2->dot(B2);
    // // C1->setName("C1");

    // C2->backward();

    // A2->printInfo();
    // B2->printInfo();
    // C2->printInfo();

    // // C2->deleteTopo();

    //-------------------------------------- Tensor3D testing ground
    Tensor3D* A3 = new Tensor3D({{
        {1, 2}, {3, 4}
    }, {
        {5, 6}, {7, 8}
    }}, "", "A3");
    Tensor3D* B3 = new Tensor3D({{
        {9, 10}, {11, 12}
    }, {
        {13, 14}, {15, 16}
    }}, "", "B3");

    // Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
    //     Eigen::IndexPair<int>(1, 0)
    // };

    Tensor3D* C3 = A3->dot(B3);
    // // C1->setName("C1");

    C3->backward();

    A3->printInfo();
    B3->printInfo();
    C3->printInfo();

    // C3->deleteTopo();

    return 0;   
}
