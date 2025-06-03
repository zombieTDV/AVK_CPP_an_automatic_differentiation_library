#include "../../include/tensor/TensorBase.h"

int TensorBase::count = 0;  

void TensorBase::buildTopo(vector<TensorBase*> &topo, vector<TensorBase*> &visited){
    if (std::find(visited.begin(), visited.end(), this) == visited.end()){
        visited.push_back(this);
        for (TensorBase* child : children){
            child->buildTopo(topo, visited);
        }
        topo.push_back(this);
    }
}

TensorBase::~TensorBase(){
    count--;
}

void TensorBase::deleteTopo(){
    for (TensorBase* pTensor : topo){
        if (pTensor->parameter != true){
            delete(pTensor);
        }else{
            continue;
        }
    }
}

void TensorBase::setName(string name) {this->name = name;}

void TensorBase::setCleaned(bool parameter) {this->parameter = parameter;}