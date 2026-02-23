#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;
  

  const unsigned int N_el = 20;
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const auto kappa = [](const Point<dim> & /*p*/) { return 6.0; };
  const auto f  = [](const Point<dim>  &/*p*/, const double  &t) {
    return 0.0;
  };
  // Knowing this, analyze the spatial convergence of the solver, considering meshes with N = 5, 10, 20, 40 elements and time steps ∆t = 0.05 and ∆t = 0.005.
  // vector of N_el
  const std::vector<unsigned int> N_el_values = {5,10, 20, 40};
  const std::vector<double> delta_t_values = {0.05, 0.005};


  for (const unsigned int &N_el : N_el_values)
    {
      for (const double &delta_t : delta_t_values)
        {
            Heat problem(/*number of elements = */ N_el,
                            /* degree = */ 1,
                            /* T = */ 1.0,
                            /* theta = */ 1.0,
                            /* delta_t = */ delta_t,
                            kappa,
                            f);

                problem.run();
              
        }
    }
  /*
  Heat problem(/*number of elements =  N_el,
               /* degree =  1,
               /* T =  1.0,
               /* theta =  1.0,
               /* delta_t =  0.05,
               kappa,
               f);

  problem.run();*/

  return 0;
}