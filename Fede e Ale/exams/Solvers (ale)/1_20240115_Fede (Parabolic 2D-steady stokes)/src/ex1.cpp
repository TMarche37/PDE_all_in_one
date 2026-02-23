#include <fstream>

#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int               mpi_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  const unsigned int degree = 2;

  const double T     = 0.02;
  const double theta = 0.5;
  //const double deltat = 0.5;

  // const std::vector<double> deltat_vector = {
  //         0.1, 0.05, 0.05/2, 0.05/4};

  //const std::vector<double> deltat_vector = {
  //       0.0625};

  const std::vector<std::string> deltat_vector = {
          "../mesh/mesh-square-h0.100000.msh",
          "../mesh/mesh-square-h0.050000.msh",
          "../mesh/mesh-square-h0.025000.msh",
          "../mesh/mesh-square-h0.012500.msh"
  };
  const std::vector<double> h_vector = {
          0.100000,
          0.050000,
          0.025000,
          0.012500
  };

  //const std::vector<std::string> mesh_vector = {
  //        "../mesh/mesh-square-5.msh",
  //        "../mesh/mesh-square-10.msh",
  //        "../mesh/mesh-square-20.msh",
  //        "../mesh/mesh-square-40.msh"
  //};

  //const std::vector<double>      h_vals = {1.0/5.0, 0.1/10.0, 1.0/20.0, 1.0/40.0};

  std::vector<double> errors_L2;
  std::vector<double> errors_H1;

  for (const auto &deltat : deltat_vector)
    {
      // Heat problem("../mesh/mesh-square-h0.100000.msh", degree, T, deltat, theta);
      Heat problem(deltat, degree, T, 0.001, theta);

      problem.setup();
      problem.solve();

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

      auto& deltat_vector = h_vector;
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

  return 0;
}