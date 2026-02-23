#include <deal.II/base/convergence_table.h>

#include "Heat.hpp"




static constexpr unsigned int dim = Heat::dim;
// Exact solution.
template <int dim>
class ExactSolution : public Function<dim>
{
public:
  // Constructor.
  ExactSolution()
  {}

  // Evaluation.
  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {

    return std::sin(2.0 * M_PI * this->get_time()) * std::sin( M_PI * p[0]) * std::sin(2.0 * M_PI * p[1]);

  }

  
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> result;
 
  
    result[0] = M_PI * std::sin(2.0 * M_PI * this->get_time()) * std::cos(M_PI * p[0]) * std::sin(2.0 * M_PI * p[1]);
    result[1] = 2.0 * M_PI * std::sin(2.0 * M_PI * this->get_time()) * std::sin(M_PI * p[0]) * std::cos(2.0 * M_PI * p[1]);


    return result;
  }


};


// Main function.
int
main(int argc, char *argv[])
{

  ConvergenceTable table;

  ExactSolution<Heat::dim> exact_solution;
  
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const auto mu = [](const Point<dim> & /*p*/) { return 1; };
  const auto f  = [](const Point<dim>  &p, const double  &t) {
    return 2.0 * M_PI * std::cos(2.0*M_PI*t) * std::sin(M_PI*p[0]) * std::sin(2.0*M_PI*p[1]) +
       5.0 * M_PI * M_PI * std::sin(2.0*M_PI*t) * std::sin(M_PI*p[0]) * std::sin(2.0*M_PI*p[1]) +
       0.1 * M_PI * std::sin(2.0*M_PI*t) * std::cos(M_PI*p[0]) * std::sin(2.0*M_PI*p[1]) +
       0.4 * M_PI * std::sin(2.0*M_PI*t) * std::sin(M_PI*p[0]) * std::cos(2.0*M_PI*p[1]);
  };
  
  std::vector<double> delta_ts = {0.05, 0.025, 0.0125, 0.00625};

  std::ofstream convergence_file("convergence.csv");
  convergence_file << "delta_t,eL2" << std::endl;

  
  for(const double& delta_t : delta_ts){
    
    Heat problem(/*mesh_filename = */ "../mesh/mesh-square-h0.100000.msh",
                /* degree = */ 2,
                /* T = */ 1.0,
                /* theta = */ 0.5,
                /* delta_t = */ delta_t,
                mu,
                f);
    
    problem.run();

    exact_solution.set_time(1.0);
    const double error_L2 = problem.compute_error(VectorTools::L2_norm, exact_solution);

    table.add_value("dt", delta_t);
    table.add_value("L2", error_L2);
    convergence_file << delta_t << "," << error_L2 << "\n";
  }

  table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
  table.write_text(std::cout);


  return 0;
}