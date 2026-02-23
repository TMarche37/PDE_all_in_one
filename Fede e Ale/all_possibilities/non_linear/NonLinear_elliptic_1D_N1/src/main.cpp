#include "NonLinearElliptic1D.hpp"

#include <fstream>
#include <iostream>
#include <vector>

using SolverClass = NonLinearElliptic1D;

void mesh_file_function(const bool silent,
                        const unsigned int r,
                        const std::vector<unsigned int> meshes,
                        const std::vector<double> h_vals,
                        const unsigned int mpi_rank)
{
  std::vector<double> errors_L2;
  std::vector<double> errors_H1;

  for (const auto &mesh : meshes)
  {
    SolverClass problem(mesh, r);

    std::streambuf* originalBuffer = std::cout.rdbuf();
    std::ofstream nullStream("dev/null");

    if (silent){
      std::cout.rdbuf(nullStream.rdbuf());
    }

    problem.setup();
    problem.solve_newton();

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

    for (unsigned int i = 0; i < meshes.size(); ++i)
    {
      const double h = h_vals[i];
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
                std::log(h / h_vals[i-1]);

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
                std::log(h / h_vals[i-1]);

        std::cout << " (" << std::fixed << std::setprecision(2)
                  << std::setw(4) << p << ")";
      }
      else
        std::cout << " (  - )";

      std::cout << "\n";
    }
  }
}

// */
void single_execution_file(const bool silent,
                           const unsigned int r,
                           const unsigned int mesh)
{
  
  SolverClass problem(mesh, r);
  
  std::streambuf* originalBuffer = std::cout.rdbuf();
  std::ofstream nullStream("dev/null");
  
  if (silent){
    std::cout.rdbuf(nullStream.rdbuf());
  }
  
  problem.setup();
  problem.solve_newton();
  
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
  const unsigned int mesh = 20;

  const bool silent = true;

  const std::vector<double> h_vals = {1.0 / 10.0,
                                      1.0 / 20.0,
                                      1.0 / 40.0,
                                      1.0 / 80.0};

  const std::vector<unsigned int> meshes = {10, 20, 40, 80};
  mesh_file_function(silent, r, meshes, h_vals,  mpi_rank);
  
  system("rm -r *.vtu *.pvtu *.vtk");
  
  single_execution_file(silent, r, mesh);

  return 0;
}