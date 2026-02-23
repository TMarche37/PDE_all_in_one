#include "Heat.hpp"


#include <fstream>
#include <iostream>
#include <deal.II/base/convergence_table.h>
#include <vector>



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
    Heat problem(N, r, T, deltat, theta);

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
    Heat problem(N, r, T, deltat, theta);


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
  
  Heat problem(N, r, T, deltat, theta);
  
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

  const unsigned int N = 99+1;

  // system("rm *vt[uk]");

  const bool silent = true;
  const double dt = 0.05;
  const std::vector<double> deltat_vector = {dt, dt/2, dt/4, dt/8, dt/16};
  time_function(silent, r, T, theta, N, deltat_vector, mpi_rank);

  const double deltat = 0.0001;
  const std::vector<unsigned int> N_values = {4, 8, 16, 32};
  mesh_function(silent, r, deltat*50, theta, deltat, N_values, mpi_rank);
  
  single_execution(silent, r, T, theta, 0.1, N);
  
  return 0;
}



