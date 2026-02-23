#include <deal.II/base/convergence_table.h>

#include "Heat.hpp"




static constexpr unsigned int dim = Heat::dim;
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
        const unsigned int /*component*/ = 0) const override
  {
    double t = 0.25; // Final time
    // Points 3 and 4.
    return std::sin(2.0 * M_PI * t) * std::sin( M_PI * p[0]) * std::sin(2.0 * M_PI * p[1]);
    // Point 5.
    // if (p[0] < 0.5)
    //   return A * p[0];
    // else
    //   return A * p[0] + 4.0 / 15.0 * std::pow(p[0] - 0.5, 2.5);
  }

  
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> result;
    const double t = 0.25; // Final time
    // Points 3 and 4.
    //derivative in p[0] direction and p[1] direction
    result[0] = M_PI * std::sin(2.0 * M_PI * t) * std::cos(M_PI * p[0]) * std::sin(2.0 * M_PI * p[1]);
    result[1] = 2.0 * M_PI * std::sin(2.0 * M_PI * t) * std::sin(M_PI * p[0]) * std::cos(2.0 * M_PI * p[1]);
    
    // Point 5.
    // if (p[0] < 0.5)
    //   result[0] = A;
    // else
    //   result[0] = A + 2.0 / 3.0 * std::pow(p[0] - 0.5, 1.5);

    return result;
  }

  static constexpr double A = -4.0 / 15.0 * std::pow(0.5, 2.5);
};


// Main function.
int
main(int argc, char *argv[])
{

  ConvergenceTable table;



  ExactSolution exact_solution;
  
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const auto mu = [](const Point<dim> & /*p*/) { return 1; };
  const auto f  = [](const Point<dim>  &p, const double  &t) {
    return 2 * M_PI * std::cos(2*M_PI*t) * std::sin(M_PI*p[0]) * std::sin(2*M_PI*p[1])+
           M_PI * std::sin(2*M_PI*t) * (5*M_PI*std::sin(M_PI*p[0]) * std::sin(2*M_PI*p[1])+
           0.1 * std::cos(M_PI*p[0]) * std::sin(2*M_PI*p[1])+
           0.4 * std::sin(M_PI*p[0]) * std::cos(2*M_PI*p[1]));
  };
  //to provide a convergence study having only a mesh i can only vary delta_t
  //vector of delta_t values: 0.1, 0.05, 0.025, 0.0125
  std::vector<double> delta_ts = {0.05, 0.025, 0.0125, 0.00625};



  std::ofstream convergence_file("convergence.csv");
  convergence_file << "delta_t,eL2" << std::endl;

  
  for(const double& delta_t : delta_ts){
    
    Heat problem(/*mesh_filename = */ "../mesh-square-h0.100000.msh",
                /* degree = */ 1,
                /* T = */ 1.0,
                /* theta = */ 0.5,
                /* delta_t = */ delta_t,
                mu,
                f);
    
    problem.run();


    const double error_L2 = problem.compute_error(VectorTools::L2_norm, exact_solution);

    table.add_value("dt", delta_t);
    table.add_value("L2", error_L2);
    convergence_file << delta_t << "," << error_L2 << "\n";
  }

  table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
  table.write_text(std::cout);


  return 0;
}