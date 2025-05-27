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

class TensorBase{
protected:
    string operation;
    string name;
    bool parameter = false;

    vector<TensorBase*> children;
    vector<TensorBase*> topo;

    std::function<void()> backwardFn = [](){};

public:
    TensorBase(string operation = "", string name = "", bool parameter = false) : operation(operation), name(name), parameter(parameter){}
    virtual ~TensorBase() = default;


    string getOperation() {return operation;}
    string getName() {return name;}
    bool isParameter() {return parameter;}
    vector<TensorBase*> getChildren() {return children;}
    vector<TensorBase*> getTopo() {return topo;}
    std::function<void()> getBackwardFn() {return backwardFn;}
    
    virtual TensorBase* operator+(TensorBase* other) = 0;
    virtual TensorBase* operator-(TensorBase* other) = 0;
    virtual TensorBase* operator-() = 0;
    virtual TensorBase* operator*(TensorBase* other) = 0;
    // virtual TensorBase* operator/(TensorBase* other) = 0;
    // virtual TensorBase* operator^(TensorBase* other) = 0;
    // virtual TensorBase* operator^(float other) = 0;
    // virtual TensorBase* operator^(int other) = 0;
    // virtual TensorBase* operator^(double other) = 0;
    // virtual TensorBase* operator^(long other) = 0;
    // virtual TensorBase* operator^(short other) = 0;

    virtual TensorBase* pow(int other) = 0;
    
    virtual void backward(){}
    virtual void printInfo(){}
    void buildTopo(vector<TensorBase*> &topo, vector<TensorBase*> &visited);
    void deleteTopo();
    void setName(string name);
    void setCleaned(bool parameter);
    void executeBackward() { backwardFn(); }
};
