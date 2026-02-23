#include <deal.II/base/convergence_table.h>

#include <iostream>

#include "DiffusionReaction.hpp"

static constexpr unsigned int dim = DiffusionReaction::dim;
/*
// Exact solution.
class ExactSolution : public Function<dim>
{
public:
  // Constructor.
  ExactSolution()
  {}

  // Evaluation.
  virtual double
  value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    (void)component;
    return std::sin(2.0 * M_PI * p[0]) * std::sin(4.0 * M_PI * p[1]);
  }

  // Gradient evaluation.
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int component = 0) const override
  {
    (void)component;
    Tensor<1, dim> result;

    result[0] =
      2.0 * M_PI * std::cos(2.0 * M_PI * p[0]) * std::sin(4.0 * M_PI * p[1]);
    result[1] =
      4.0 * M_PI * std::sin(2.0 * M_PI * p[0]) * std::cos(4.0 * M_PI * p[1]);

    return result;
  }

  static constexpr double A = -4.0 / 15.0 * std::pow(0.5, 2.5);
};
*/
// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  const unsigned int              r          = 2;

  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };

  const auto sigma = [](const Point<dim> & /*p*/) { return 1.0; };

  const auto f = [](const Point<dim> &p) {
    (void)p;
    return 0.0;
  };


    const std::string mesh_file_name ="../mesh/mesh-square-h0.100000.msh";

    DiffusionReaction problem(mesh_file_name, r, mu, sigma, f);

    problem.setup();
    problem.assemble();
    problem.solve();
    problem.output();

  return 0;
}