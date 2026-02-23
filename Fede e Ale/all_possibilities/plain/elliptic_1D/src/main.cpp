#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "Elliptic1D.hpp"


//------------------------------------
//---------------1D--------------------
//---------------------------------------


void mesh_function(const bool silent,
                   const unsigned int r,
                   const std::vector<unsigned int> N_vector)


{
  ConvergenceTable table;
  
  std::vector<double> errors_L2;
  std::vector<double> errors_H1;
  
  for (const auto &N : N_vector)
  {
    Elliptic1D problem(N, r);
    
    std::streambuf* originalBuffer = std::cout.rdbuf();
    std::ofstream nullStream("dev/null");
    
    if (silent){
      std::cout.rdbuf(nullStream.rdbuf());
    }
    
    problem.setup();
    problem.assemble();
    problem.solve();
    problem.output();
    
    
    if (silent){
      std::cout.rdbuf(originalBuffer);
    }
    
    errors_L2.push_back(problem.compute_error(VectorTools::L2_norm));
    errors_H1.push_back(problem.compute_error(VectorTools::H1_norm));
  }
  
  // Print the errors and estimate the convergence order.

  std::cout << "==============================================="
            << std::endl;
    
  
  std::ofstream convergence_file("convergence.csv");
  convergence_file << "h,eL2,eH1" << std::endl;
  
  for (unsigned int i = 0; i < N_vector.size(); ++i)
  {
    const double h        = 1.0 / (N_vector[i] + 1.0);
    convergence_file << h << "," << errors_L2[i] << ","
                     << errors_H1[i] << std::endl;
    
    std::cout << std::scientific << "h = " << std::setw(4)
              << std::setprecision(2) << h;
    
    std::cout << std::scientific << " | eL2 = " << errors_L2[i];
    
    // Estimate the convergence order.
    if (i > 0)
    {
      const double p =
              std::log(errors_L2[i] / errors_L2[i - 1]) /
              std::log(h / (1.0 / (N_vector[i-1] + 1.0)));
      
      std::cout << " (" << std::fixed << std::setprecision(2)
                << std::setw(4) << p << ")";
    }
    else
      std::cout << " (  - )";
    
    std::cout << std::scientific << " | eH1 = " << errors_H1[i];
    
    // Estimate the convergence order.
    if (i > 0)
    {
      const double p =
              std::log(errors_H1[i] / errors_H1[i - 1]) /
              std::log(h / (1.0 / (N_vector[i-1] + 1.0)));
      
      std::cout << " (" << std::fixed << std::setprecision(2)
                << std::setw(4) << p << ")";
    }
    else
      std::cout << " (  - )";
    
    std::cout << "\n";
  }
  
}




void single_execution(const bool silent,
                      const unsigned int r,
                      const unsigned int N)


{
  
  Elliptic1D problem(N, r);
  
  std::streambuf* originalBuffer = std::cout.rdbuf();
  std::ofstream nullStream("dev/null");
  
  if (silent){
    std::cout.rdbuf(nullStream.rdbuf());
  }
  
  
  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();
  
  
  if (silent){
    std::cout.rdbuf(originalBuffer);
  }
  
}




// Main function.
int
main()
{
  
  const unsigned int r = 1;
  
  //-------1D--------
  
  const bool silent = true;
 
  const unsigned int N = 31;
  const std::vector<unsigned int> N_values = {9, 19, 39, 79, 159, 319};
  mesh_function(silent, r, N_values);
  
  system("rm -r *.vtu *.pvtu *.vtk");
  
  single_execution(silent, r, N);
  
  return 0;
}








