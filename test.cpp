#include <iostream>
#include "unsupported/Eigen/CXX11/Tensor"

int main() {
    // Define two 2×2 “matrices” as rank-2 tensors:
    Eigen::Tensor<float, 2> A(2, 2);
    Eigen::Tensor<float, 2> B(2, 2);

    // Initialize A = [ [1, 2],
    //                  [3, 4] ]
    A.setValues({ {1.0f, 2.0f},
                  {3.0f, 4.0f} });

    // Initialize B = [ [5, 6],
    //                  [7, 8] ]
    B.setValues({ {5.0f, 6.0f},
                  {7.0f, 8.0f} });

    // Specify a single pair of dimensions to contract:
    //   - contract A’s 2nd axis (index 1) with B’s 1st axis (index 0)
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
        Eigen::IndexPair<int>(1, 0)
    };  // 2D contraction → matrix product :contentReference[oaicite:0]{index=0}

    // Perform the contraction: resulting tensor is rank 2 (2+2−2*1 = 2)
    Eigen::Tensor<float, 2> C = A.contract(B, contract_dims);

    // Output the result C = A * B
    std::cout << "C(0,0) = " << C(0,0) << "\n";  // = 1*5 + 2*7 = 19
    std::cout << "C(0,1) = " << C(0,1) << "\n";  // = 1*6 + 2*8 = 22
    std::cout << "C(1,0) = " << C(1,0) << "\n";  // = 3*5 + 4*7 = 43
    std::cout << "C(1,1) = " << C(1,1) << "\n";  // = 3*6 + 4*8 = 50

    return 0;
}
