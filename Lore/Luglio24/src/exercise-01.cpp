#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  int N_el = 40;
  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };
  const auto f  = [](const Point<dim>  &p, const double  &t) {
    return M_PI/2.0 * std::sin(M_PI / 2.0 * p[0]) * std::cos(M_PI / 2.0 * t)
           + (M_PI * M_PI / 4.0 - 1.0)* std::sin(M_PI / 2.0 * p[0]) * std::sin(M_PI / 2.0 * t)
           + (M_PI / 2.0) * std::cos(M_PI / 2.0 * p[0]) * std::sin(M_PI / 2.0 * t);
      
  };

  Heat problem(/*mesh_filename = */ N_el,
               /* degree = */ 2,
               /* T = */ 1.0,
               /* theta = */ 0.5,
               /* delta_t = */ 0.1,
               mu,
               f);

  problem.run();

  return 0;
}