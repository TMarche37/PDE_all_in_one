#include "Stokes.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name  = "../mesh/mesh-pipe.msh";
  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;
  const double silent = true;

  Stokes problem(mesh_file_name, degree_velocity, degree_pressure);
  
  
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
  
  
  


  return 0;
}