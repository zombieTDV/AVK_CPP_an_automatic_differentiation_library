#pragma once //Báo cho trình biên dịch không khai báo lớp này quá 1 lần.
#include "../../unsupported/Eigen/CXX11/Tensor"
#include <iostream>
// #include <functional>
// #include <algorithm>

using std::vector;
using std::string;
using std::cout;

// using std::shared_ptr;
// using std::make_shared;

using Float1D = std::initializer_list<float>;
using Float2D = std::initializer_list<std::initializer_list<float>>;
using Float3D = std::initializer_list<std::initializer_list<std::initializer_list<float>>>;

// using Pair1D = Eigen::array<Eigen::IndexPair<int>, 1>;

class Tensor0D;  // Forward declaration
class OptimizationFunc;

class TensorBase{
protected:
    const string operation;
    // string name;
    bool parameter = false;

    vector<TensorBase*> children;
    
    
    std::function<void()> backwardFn = [](){};
    
    static vector<TensorBase*> topo;
    static int instanceCount;
    static int memoryUsage;
public:
    friend class OptimizationFunc; 
    
    TensorBase(string operation = "", bool parameter = false) : operation(operation), parameter(parameter){
        TensorBase::instanceCount ++;
    }
    
    virtual ~TensorBase() {TensorBase::instanceCount --;}

    static int getInstanceCount() {return instanceCount;}
    static void printMemoryUsage() {cout << "Memory Usage: " << memoryUsage << " bytes (" << (float)memoryUsage/(1024*1024) << " MB) for " << instanceCount << " instances" << "\n";}

    string getOperation() const {return operation;}
    // string getName() const {return name;}
    bool isParameter() const {return parameter;}
    vector<TensorBase*> getChildren() const {return children;}
    vector<TensorBase*> getTopo() const {return topo;}
    std::function<void()> getBackwardFn() const {return backwardFn;}
    
    virtual TensorBase* operator+(TensorBase* other) = 0;
    virtual TensorBase* operator-(TensorBase* other) = 0;
    virtual TensorBase* operator-() = 0;
    virtual TensorBase* operator*(TensorBase* other) = 0;
    virtual TensorBase* operator*(Tensor0D* other) = 0;

    virtual TensorBase* pow(int other) = 0;
    virtual TensorBase* pow(double other) = 0;
    virtual TensorBase* pow(float other) = 0;
    virtual TensorBase* pow(Tensor0D* other) = 0;
    
    virtual void backward() = 0;
    virtual void printInfo() = 0;
    
    static void reserveTopo(int r_size){
        if(r_size < 0){
            throw std::runtime_error("Reserve size for topo must not be negative");
        }

        else{
            cout << "[[Reversed " << r_size << " memory slots for topo]]\n";
            TensorBase::topo.reserve(r_size);
        }
    }
    
    void buildTopo(vector<TensorBase*> &visited);
    void deleteTopo();
    // void setName(string name);
    void setParameter(bool parameter);

    void executeBackward() { backwardFn(); }

    friend std::ostream& operator<<(std::ostream& out, const TensorBase* T){
        out << "Tensor Base: \n\tOperation: " << T->getOperation() << "\t is parameter: " << T->isParameter() << '\n'; 
        return out; 
    }
};
