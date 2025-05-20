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
public:
    virtual ~TensorBase() = default;

    string operation;
    string name;
    bool parameter = false;

    vector<TensorBase*> children;
    vector<TensorBase*> topo;

    std::function<void()> backwardFn = [](){};

    // Constructor declarations
    TensorBase(string operation = "", string name = "", bool parameter = false) : operation(operation), name(name), parameter(parameter){}

     // Function declarations
    virtual void backward(){}
    virtual void printInfo(){}
    void buildTopo(vector<TensorBase*> &topo, vector<TensorBase*> &visited);
    void deleteTopo();
    void setName(string name);
    void setCleaned(bool parameter);
};
