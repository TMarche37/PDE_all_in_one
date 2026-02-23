#include "Stokes.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name  = "../mesh/mesh-pipe.msh";
  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;
  const double T = 1;
  const double deltat = T/20;
  const double theta = 0.5;
  
  const double silent = true;
  
  
  std::streambuf* originalBuffer = std::cout.rdbuf();
  std::ofstream nullStream("dev/null");
  
  Stokes problem(mesh_file_name, degree_velocity, degree_pressure, T, deltat, theta);
  
  if (silent){
    std::cout.rdbuf(nullStream.rdbuf());
  }

  problem.setup();
  problem.solve();
  
  if (silent){
    std::cout.rdbuf(originalBuffer);
  }
  
  
  
  return 0;
}