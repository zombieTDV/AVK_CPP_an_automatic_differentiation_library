#include "../../include/tensor/TensorBase.h"

int TensorBase::instanceCount = 0;  
int TensorBase::memoryUsage = 0;

vector<TensorBase*> TensorBase::topo;

void TensorBase::buildTopo(vector<TensorBase*> &visited){
    if (std::find(visited.begin(), visited.end(), this) == visited.end()){
        visited.push_back(this);
        for (TensorBase* child : children){
            child->buildTopo(visited);
        }
        topo.push_back(this);
    }
}

void TensorBase::deleteTopo(){
    for (TensorBase* pTensor : topo){
        if (pTensor->parameter != true){
            delete(pTensor);
        }else{
            continue;
        }
    }

    topo.clear();
}

// void TensorBase::setName(string name) {this->name = name;}

void TensorBase::setParameter(bool parameter) {this->parameter = parameter;}