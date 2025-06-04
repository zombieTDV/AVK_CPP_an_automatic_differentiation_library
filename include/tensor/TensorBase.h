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
class Tensor1D;
class Tensor2D;
class Tensor3D;
class OptimizationFunc;

class TensorBase{
protected:
    string operation;
    string name;
    bool parameter = false;

    vector<TensorBase*> children;
    vector<TensorBase*> topo;

    std::function<void()> backwardFn = [](){};

    static int instanceCount;
    static int memoryUsage;
public:
    friend class OptimizationFunc; 
    
    TensorBase(string operation = "", string name = "", bool parameter = false) : operation(operation), name(name), parameter(parameter){
        instanceCount++;
        memoryUsage += sizeof(TensorBase);
    }
    
    virtual ~TensorBase();

    static int getInstanceCount() {return instanceCount;}
    static void printMemoryUsage() {cout << "Memory Usage: " << memoryUsage << " bytes (" << (float)memoryUsage/(1024*1024) << " MB) for " << instanceCount << " instances" << "\n";}

    string getOperation() const {return operation;}
    string getName() const {return name;}
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
    void buildTopo(vector<TensorBase*> &topo, vector<TensorBase*> &visited);
    void deleteTopo();
    void setName(string name);
    void setCleaned(bool parameter);

    void executeBackward() { backwardFn(); }
};
