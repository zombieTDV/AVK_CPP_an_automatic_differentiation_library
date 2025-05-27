// #include "Eigen/Dense"

#include "include/tensor/TensorBase.h"
#include "include/tensor/Tensor0D.h"
#include "include/tensor/Tensor1D.h"
#include "include/tensor/Tensor2D.h"
#include "include/tensor/Tensor3D.h"

#include <iostream>
// #include <variant>
// #include <memory>

using std::vector;
using std::string;
using std::cout;

//------------------------------------------------------------

class MeanSquaredErrorLoss{
public:
    // Friend declarations
    friend class Tensor0D;
    friend class Tensor1D;  
    friend class Tensor2D;
    friend class Tensor3D;
    Tensor0D* Tensor0D_Loss(Tensor0D* true_y, Tensor0D* predicted_y){
        Tensor0D* loss = (*true_y - predicted_y)->pow(2);
        return loss;
    }
};

class OptimizationFunc{
public:
    // Friend declarations
    friend class Tensor0D;
    friend class Tensor1D;  
    friend class Tensor2D;
    friend class Tensor3D;
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
        {1, 3}, 
        {2, 4}
    }, {
        {5, 7}, 
        {6, 8}
    }}, "", "A3");
    Tensor3D* B3 = new Tensor3D({{
        {1, 3}, 
        {2, 4}
    }, {
        {5, 7}, 
        {6, 8}
    }}, "", "B3");

    // Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
    //     Eigen::IndexPair<int>(1, 0)
    // };

    Tensor3D* C3 = A3->dot(B3);
    C3->setName("C3");

    C3->backward();

    A3->printInfo();
    B3->printInfo();
    C3->printInfo();

    // C3->deleteTopo();

    return 0;   
}
