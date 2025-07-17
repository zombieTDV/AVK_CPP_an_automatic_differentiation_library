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
    Tensor0D* Tensor0D_Loss(Tensor0D* true_y, Tensor0D* predicted_y){
        Tensor0D* loss = (*true_y - predicted_y)->pow(2);
        return loss;
    }

    Tensor3D* Tensor3D_Loss(Tensor3D* true_y, Tensor3D* predicted_y){
        Tensor3D* loss = (*true_y - predicted_y)->pow(2);
        return loss;
    }
};

class OptimizationFunc{
private:
public:
    OptimizationFunc() {};

    void gradientDescent(float learning_rate){
        for (auto it = TensorBase::topo.rbegin(); it != TensorBase::topo.rend(); ++it){
            (*it)->applyGradientDescent(learning_rate);
    }}
};

int main(){
    TensorBase::reserveTopo(10);
    MeanSquaredErrorLoss Loss_func;
    OptimizationFunc Opti_func;
    //-------------------------------------- Tensor0D testing ground
    // MeanSquaredErrorLoss Loss_func;
    // OptimizationFunc Opti_func;
    // Tensor0D* A0 = new Tensor0D(5, "", true);
    // Tensor0D* B0 = new Tensor0D(3, "", true);

    // for (int i = 0; i < 10; i++){
    //     Tensor0D* Y_hat = new Tensor0D(15, "");
    //     Tensor0D* Y = (*A0 + B0);
    //     Tensor0D* C0 = Loss_func.Tensor0D_Loss(Y, Y_hat);

    //     C0->backward();

    //     cout << A0;
    //     cout << B0;
    //     cout << Y;

    //     Opti_func.gradientDescent(0.1);


    //     cout << "Loss: " << C0->getData() << '\n';

    //     // TensorBase::printMemoryUsage();
    //     C0->deleteTopo();

    // }
    // TensorBase::printMemoryUsage();
    

    //-------------------------------------- Tensor1D testing ground
    // Tensor1D* A1 = new Tensor1D({3, 5});
    // Tensor1D* B1 = new Tensor1D({6, 9});

    // Tensor1D* C1 = *A1 * B1;

    // C1->backward();

    // cout << A1;
    // cout << B1;
    // cout << C1;

    // C1->deleteTopo();
    // TensorBase::printMemoryUsage();

    //-------------------------------------- Tensor2D testing ground
    // Tensor2D* A2 = new Tensor2D({{1, 2}, 
    //                             {3, 4}}, "");
    // Tensor2D* B2 = new Tensor2D({{5, 6, 7}, 
    //                                 {8, 9, 10}}, "", true);

    // Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
    //     Eigen::IndexPair<int>(1, 0)
    // };

    // Tensor2D* C2 = A2->dot(B2);

    // C2->backward();

    // cout << A2;
    // cout << B2;
    // cout << C2;

    // C2->deleteTopo();
    // TensorBase::printMemoryUsage();

    //-------------------------------------- Tensor3D testing ground
    Tensor3D* A3 = new Tensor3D({{
        {1, 2}, 
        {3, 4}
    }, {
        {5, 6}, 
        {7, 8}
    }}, "", true);
    Tensor3D* B3 = new Tensor3D({{
        {1, 2}, 
        {3, 4}
    }, {
        {5, 6}, 
        {7, 8}
    }}, "", true);

    // Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
    //     Eigen::IndexPair<int>(1, 0)
    // };
    for (int i = 0; i < 1; i++){
        Tensor3D* Y = new Tensor3D({{
            {0, 0}, 
            {0, 0}
        }, {
            {0, 0}, 
            {0, 0}
        }}, "");


        Tensor3D* Y_hat = A3->dot(B3);

        Tensor3D* Loss = Loss_func.Tensor3D_Loss(Y, Y_hat);
        Loss->backward();

        
        cout << A3;
        cout << B3;
        cout << Y_hat;
        cout << Loss;
        
        // Opti_func.gradientDescent(0.1);
        
        TensorBase::printMemoryUsage();
        Loss->deleteTopo();
        TensorBase::printMemoryUsage();
    }

    
    // TensorBase::printMemoryUsage();
    return 0;   
}
