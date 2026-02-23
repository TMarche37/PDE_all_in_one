#include <iostream>

#include "Poisson1D.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  constexpr unsigned int dim = Poisson1D::dim;

  const unsigned int N_el = 10;
  const unsigned int r    = 1;
  const auto         mu   = [](const Point<dim>           &/*p*/) { return 1.0; };
  const auto         f    = [](const Point<dim> &/*p*/) {
      return 0.0;
  };

  Poisson1D problem(N_el, r, mu, f);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}
