#include <deal.II/base/convergence_table.h>

#include "Heat.hpp"

static constexpr unsigned int dim = Heat::dim;

// Manufactured exact solution for convergence study:
// u(x,t) = sin(2*pi*x) sin(2*pi*y) sin(2*pi*z) * sin(pi*t)
// with mu = constant.
// Forcing f = u_t - mu * Laplacian(u)
class ExactSolution : public Function<dim>
{
public:
  ExactSolution() : Function<dim>(), mu(0.1) {}

  void
  set_mu(const double mu_) { mu = mu_; }

  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {
    const double S = cos(2.0*M_PI*p[0]) * cos(2.0*M_PI*p[1]) * cos(2.0*M_PI*p[2]);
    return sin(M_PI * this->get_time()) * S;

  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
                          const unsigned int /*component*/) const
  {
    const double t = this->get_time();

    Tensor<1, dim> grad;

    const double factor = -2.0 * numbers::PI * std::sin(numbers::PI * t);

    grad[0] =
      factor *
      std::sin(2.0 * numbers::PI * p[0]) *
      std::cos(2.0 * numbers::PI * p[1]) *
      std::cos(2.0 * numbers::PI * p[2]);

    grad[1] =
      factor *
      std::cos(2.0 * numbers::PI * p[0]) *
      std::sin(2.0 * numbers::PI * p[1]) *
      std::cos(2.0 * numbers::PI * p[2]);

    grad[2] =
      factor *
      std::cos(2.0 * numbers::PI * p[0]) *
      std::cos(2.0 * numbers::PI * p[1]) *
      std::sin(2.0 * numbers::PI * p[2]);

    return grad;
  }


private:
  double mu;
};

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  ConvergenceTable table;
  const std::vector<unsigned int> N_el_values = {5, 10, 20, 40};
  const unsigned int              r           = 2;

  // Physical parameters
  const double mu_val = 0.1;

  // Time parameters: choose final time and small delta_t so spatial error
  // dominates.
  const double T       = 1;
  const double theta   = 1;   // Backward Euler
  const double delta_t = 0.01;

  std::ofstream convergence_file("convergence.csv");
  convergence_file << "h,eL2,eH1" << std::endl;

  for (const auto &N_el : N_el_values)
    {
      const std::string mesh_file_name =
        "../mesh/mesh-cube-" + std::to_string(N_el) + ".msh";

      // Exact solution object.
      ExactSolution exact_solution;

      // mu lambda
      const auto mu = [mu_val](const Point<dim> & /*p*/) { return mu_val; };

      // Forcing term derived from the manufactured solution:
      // u(x,t) = cos(2πx) cos(2πy) cos(2πz) sin(πt)
      // f = u_t - mu * Laplacian(u)
      const auto f = [mu_val](const Point<dim> &p, const double &t) {
        const double S =
          std::cos(2.0 * M_PI * p[0]) *
          std::cos(2.0 * M_PI * p[1]) *
          std::cos(2.0 * M_PI * p[2]);

        const double result =
          M_PI * std::cos(M_PI * t) * S
          + 12.0 * M_PI * M_PI * mu_val * std::sin(M_PI * t) * S;

        return result;
      };

      Heat problem(mesh_file_name, r, T, theta, delta_t, mu, f);

      problem.run();

      // set exact time to final time and compute errors
      exact_solution.set_time(T);

      const double h = 1.0 / N_el;

      // compute L2 and H1 error
      double error_L2 = problem.compute_error(VectorTools::L2_norm, exact_solution);
      double error_H1 = problem.compute_error(VectorTools::H1_norm, exact_solution);

      table.add_value("h", h);
      table.add_value("L2", error_L2);
      table.add_value("H1", error_H1);

      convergence_file << h << "," << error_L2 << "," << error_H1 << std::endl;
    }

  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
  table.set_scientific("L2", true);
  table.set_scientific("H1", true);
  table.write_text(std::cout);

  return 0;
}