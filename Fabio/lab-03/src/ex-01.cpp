#include <deal.II/base/convergence_table.h>

#include <iostream>

#include "DiffusionReactionmine.hpp"

static constexpr unsigned int dim = DiffusionReactionmine::dim;

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  

  const std::string  mesh_file_name = "../mesh/mesh-cube-40.msh";
  const unsigned int r              = 1;

  const auto mu = [](const Point<dim> &p) {
    if(p[0]<0.5){
        return 100.0;
    }else if(p[0]>=0.5){
        return 1.0;
    }
  };
  const auto f  = [](const Point<dim>  &/*p*/) { return 1.0; };
  const auto sigma  = [](const Point<dim> &/*p*/) { return 1.0; };

  DiffusionReactionmine problem(mesh_file_name, r, mu, sigma, f);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}