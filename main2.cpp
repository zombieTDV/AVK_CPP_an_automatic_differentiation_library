#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"
#include <iostream>
#include <variant>
#include <memory>

using std::vector;
using std::string;
using std::cout;

using Float1D = std::initializer_list<float>;
using Float2D = std::initializer_list<std::initializer_list<float>>;
using Float3D = std::initializer_list<std::initializer_list<std::initializer_list<float>>>;

class TensorBase{
public:
    string operation;
    string name;
    bool parameter = false;

    int ranks = -1;

    vector<TensorBase*> children;
    vector<TensorBase*> topo;

    std::function<void()> backwardFn = [](){};

    TensorBase(string operation = "", string name = "", bool parameter = false) : operation(operation), name(name), parameter(parameter){}
    
    virtual ~TensorBase() = default;

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

    virtual Eigen::Tensor<float, 0> getData() const {
        return Eigen::Tensor<float,0> ();
    }

    virtual Eigen::Tensor<float, 0> getGrad() const {
        return Eigen::Tensor<float,0> ();
    }

    virtual Eigen::Tensor<float, 0> setGrad() const {
        return Eigen::Tensor<float,0> ();
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
            this->ranks = 0;
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
    }}

    Eigen::Tensor<float, 0> getData() const override{
        return this->data;
    }

    Eigen::Tensor<float, 0> getGrad() const override{
        return this->grad;
    }

    Eigen::Tensor<float, 0> setGrad() const override{
        return this->grad; 
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

    // Eigen::Tensor<float, 1> getData() const override{
    //     return this->data;
    // }

    // Eigen::Tensor<float, 1> getGrad() const override{
    //     return this->grad;
    // }

    // Eigen::Tensor<float, 1> setGrad() const override{
    //     return this->grad; 
    // }
};



//------------------------------------------------------------


class Tensor{
public:
    TensorBase* ptensor;

    Tensor(Tensor0D* ptr) : ptensor(ptr) {}


    void printAttribute() const {
            if (ptensor) ptensor->printInfo();
        }

    void casting(Tensor* other){
        if (other->ptensor->ranks == 0){
            Tensor0D* newTensor = dynamic_cast<Tensor0D*> (other->ptensor);
            
            if(newTensor){
                other->ptensor = newTensor;
            }
        }
    }

    Tensor* operator+(Tensor* other){
        casting(other);
        Tensor0D* output = new Tensor0D((this->ptensor->getData() + other->ptensor->getData()), "+");

        output->children = {this->ptensor, other->ptensor};

        output->backwardFn = [output, this, other] () {
            this->ptensor->getGrad() += output->grad;
            other->ptensor->getGrad() += output->grad;
        };

        return new Tensor(output);
    }
};


// int main(){

//     return 0;
// }