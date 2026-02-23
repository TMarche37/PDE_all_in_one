#include "LinearElasticity.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name = "../mesh/mesh-cube-10.msh";
  const unsigned int degree         = 1;
  
  const double silent = true;

  LinearElasticity problem(mesh_file_name, degree);
  
  std::streambuf* originalBuffer = std::cout.rdbuf();
  std::ofstream nullStream("dev/null");
  
  if (silent){
    std::cout.rdbuf(nullStream.rdbuf());
  }

  problem.setup();
  problem.assemble_system();
  problem.solve_system();
  problem.output();
  
  if (silent){
    std::cout.rdbuf(originalBuffer);
  }

  return 0;
}