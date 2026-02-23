#include <iostream>

#include "Poisson2D.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  constexpr unsigned int dim = Poisson2D::dim;

  const std::string  mesh_file_name = "../mesh-square-h0.100000.msh";
  const unsigned int r              = 2;

  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };
  const auto sigma = [](const Point<dim> & /*p*/) { return 1.0; };
  const auto f  = [](const Point<dim>  &p) { return -2.0 + 1.0*(p[0] + p[1]); };
  const auto phi = [=](const Point<dim> &p)
{
  const double x = p[0];
  const double y = p[1];

  // lato x = 1
  if (std::abs(x - 1.0) < 1e-5)
    return 1.0 + 1.0 * (1.0 + y);

  // lato y = 1
  if (std::abs(y - 1.0) < 1e-5)
    return 1.0 + 1.0 * (x + 1.0);

  // Se viene chiamata fuori da Gamma_N, è un errore logico
  return 0.0;
};

  Poisson2D problem(mesh_file_name, r, mu, sigma, f, phi);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}