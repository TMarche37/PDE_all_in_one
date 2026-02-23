#include <deal.II/base/convergence_table.h>
#include <fstream>

#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int               mpi_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  const unsigned int degree = 1;

  const double T     = 1.0;
  const double theta = 1;

  const int N = 10; // h = 0.1
  const double deltat = 0.01;

  Heat problem(N, degree, T, deltat, theta);

  problem.setup();
  problem.solve();

  return 0;
}