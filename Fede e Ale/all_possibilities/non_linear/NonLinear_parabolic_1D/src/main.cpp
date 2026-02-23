#include "NonLinearParabolic1D.hpp"

#include <fstream>
#include <iostream>
#include <deal.II/base/convergence_table.h>
#include <vector>

//------------------------------------
//---------------1D--------------------
//---------------------------------------

using SolverClass = NonLinearParabolic1D;

void time_function(const bool silent,
                   const unsigned int r,
                   const double T,
                   const double theta,
                   const unsigned int N,
                   const std::vector<double> deltat_vector,
                   const unsigned int mpi_rank)
{
  ConvergenceTable table;
  
  std::vector<double> errors_L2;
  std::vector<double> errors_H1;
  
  for (const auto &deltat : deltat_vector)
  {
    SolverClass problem(N, r, T, deltat);
    
    
    std::streambuf* originalBuffer = std::cout.rdbuf();
    std::ofstream nullStream("dev/null");
    
    if (silent){
      std::cout.rdbuf(nullStream.rdbuf());
    }
    
    problem.setup();
    problem.solve();
    
    if (silent){
      std::cout.rdbuf(originalBuffer);
    }
    
    errors_L2.push_back(problem.compute_error(VectorTools::L2_norm));
    errors_H1.push_back(problem.compute_error(VectorTools::H1_norm));
  }
  
  // Print the errors and estimate the convergence order.
  if (mpi_rank == 0)
  {
    std::cout << "==============================================="
              << std::endl;
    
    std::ofstream convergence_file("convergence.csv");
    convergence_file << "dt,eL2,eH1" << std::endl;
    
    for (unsigned int i = 0; i < deltat_vector.size(); ++i)
    {
      convergence_file << deltat_vector[i] << "," << errors_L2[i] << ","
                       << errors_H1[i] << std::endl;
      
      std::cout << std::scientific << "dt = " << std::setw(4)
                << std::setprecision(2) << deltat_vector[i];
      
      std::cout << std::scientific << " | eL2 = " << errors_L2[i];
      
      // Estimate the convergence order.
      if (i > 0)
      {
        const double p =
                std::log(errors_L2[i] / errors_L2[i - 1]) /
                std::log(deltat_vector[i] / deltat_vector[i - 1]);
        
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
                std::log(deltat_vector[i] / deltat_vector[i - 1]);
        
        std::cout << " (" << std::fixed << std::setprecision(2)
                  << std::setw(4) << p << ")";
      }
      else
        std::cout << " (  - )";
      
      std::cout << "\n";
    }
  }
}



void mesh_function(const bool silent,
                   const unsigned int r,
                   const double T,
                   const double theta,
                   const double deltat,
                   const std::vector<unsigned int> N_vector,
                   const unsigned int mpi_rank)
{
  ConvergenceTable table;
  
  std::vector<double> errors_L2;
  std::vector<double> errors_H1;
  
  for (const auto &N : N_vector)
  {
    SolverClass problem(N, r, T, deltat);
    
    std::streambuf* originalBuffer = std::cout.rdbuf();
    std::ofstream nullStream("dev/null");
    
    if (silent){
      std::cout.rdbuf(nullStream.rdbuf());
    }
    
    problem.setup();
    problem.solve();
    
    if (silent){
      std::cout.rdbuf(originalBuffer);
    }
    
    errors_L2.push_back(problem.compute_error(VectorTools::L2_norm));
    errors_H1.push_back(problem.compute_error(VectorTools::H1_norm));
  }
  
  // Print the errors and estimate the convergence order.
  if (mpi_rank == 0)
  {
    std::cout << "==============================================="
              << std::endl;
    
    
    std::ofstream convergence_file("convergence.csv");
    convergence_file << "h,eL2,eH1" << std::endl;
    
    for (unsigned int i = 0; i < N_vector.size(); ++i)
    {
      const double h        = 1.0 / (N_vector[i]);
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
                std::log(h / (1.0 / (N_vector[i-1])));
        
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
                std::log(h / (1.0 / (N_vector[i-1])));
        
        std::cout << " (" << std::fixed << std::setprecision(2)
                  << std::setw(4) << p << ")";
      }
      else
        std::cout << " (  - )";
      
      std::cout << "\n";
    }
  }
}




void single_execution(const bool silent,
                      const unsigned int r,
                      const double T,
                      const double theta,
                      const double deltat,
                      const unsigned int N)
{
  
  SolverClass problem(N, r, T, deltat);
  
  std::streambuf* originalBuffer = std::cout.rdbuf();
  std::ofstream nullStream("dev/null");
  
  if (silent){
    std::cout.rdbuf(nullStream.rdbuf());
  }
  
  problem.setup();
  problem.solve();
  
  if (silent){
    std::cout.rdbuf(originalBuffer);
  }
  
}




// Main function.
int
main(int argc, char * argv[])
{
  
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  
  
  
  const unsigned int r = 1;
  const double T      = 1.0;
  const double theta  = 0.5;
  
  const bool silent = true;
  
  const unsigned int N = 100;
  
  const std::vector<double> deltat_vector = {0.2, 0.1, 0.05, 0.025};
  time_function(silent, r,T,theta, N, deltat_vector, mpi_rank);
  
  const double deltat = 0.01;
  constexpr int N0 = 4;
  const std::vector<unsigned int> N_values = {N0, N0*2, N0*4, N0*8};
  mesh_function(silent, r, T, theta, deltat, N_values, mpi_rank);
  std::cout << "dt = " << std::scientific << deltat << "\n";
  
  system("rm -r *.vtu *.pvtu *.vtk");
  
  //single_execution(silent, r, T, theta, deltat, N);
  
  return 0;
}



