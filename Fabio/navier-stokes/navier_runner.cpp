#include "navier.hpp"

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // Example settings:
  // - mesh: reuse the Stokes mesh (adjust as needed)
  // - FE: u degree 2, p degree 1
  // - dt = 0.1, T = 1
  const std::string mesh_file = "../stokestimedependet/mesh-square-h0.100000.msh";

  NavierStokes problem(mesh_file,
                       /*degree_velocity=*/2,
                       /*degree_pressure=*/1,
                       /*dt=*/0.1,
                       /*T=*/1.0);

  problem.run();

  return 0;
}
