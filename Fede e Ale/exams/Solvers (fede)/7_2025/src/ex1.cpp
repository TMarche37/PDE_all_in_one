#include <iostream>
#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <vector>


#include "Poisson1D.hpp"



// Main function.
int
main(int argc, char * argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int r = 1;
  const double T      = 1.0;
  const double theta  = 1;
  const unsigned int N = 9;
  const double deltat = 0.01;



  Poisson1D problem(N, r, T, deltat, theta);

  problem.setup();
  problem.solve();



  return 0;
}




/*


std::ofstream convergence_file("convergence.csv");
convergence_file << "h,eL2,eH1" << std::endl;

for (const unsigned int &N : N_values)
{
Poisson1D problem(N, r, T, deltat, theta);


problem.setup();
problem.solve();

const double h        = 1.0 / (N + 1.0);
const double error_L2 = problem.compute_error(VectorTools::L2_norm);
const double error_H1 = problem.compute_error(VectorTools::H1_norm);

table.add_value("h", h);
table.add_value("L2", error_L2);
table.add_value("H1", error_H1);

convergence_file << h << "," << error_L2 << "," << error_H1 << std::endl;
}

table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);

table.set_scientific("L2", true);
table.set_scientific("H1", true);

table.write_text(std::cout);



 */