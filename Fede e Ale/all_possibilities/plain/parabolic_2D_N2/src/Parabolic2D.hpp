#ifndef PARABOLIC2D_HPP
#define PARABOLIC2D_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <math.h>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class Parabolic2D
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;
  
  static constexpr double b1 = -2.0;
  static constexpr double b2 = -3.0;
  static constexpr double mu = 3.0;
  static constexpr double sigma = 2.0;
  
  
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    DiffusionCoefficient(){}
    
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return mu;
    }
  };
  
  
  class AdvectionCoefficient: public Function<dim>
  {
  public:
    AdvectionCoefficient(){}
    virtual void
    vector_value(const Point<dim> & /*p*/,
                 Vector<double> &values) const override{
      values[0] = b1;
      values[1] = b2;
    }
  };
  
  
  
  class ReactionCoefficient : public Function<dim>
  {
  public:
    ReactionCoefficient(){}
    
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return sigma;
    }
  };
  
  
  
  
  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    
    ForcingTerm(){}
    
    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override {
      /*return 2.0 * M_PI * cos(2.0*M_PI*get_time()) * sin(M_PI*p[0]) * sin(2.0*M_PI*p[1])
            + mu * M_PI * M_PI * sin(M_PI*p[0]) * sin(2.0*M_PI*get_time()) * sin(2.0*M_PI*p[1])
            + mu * 4.0 * M_PI * M_PI * sin(2.0*M_PI*p[1]) * sin(2.0*M_PI*get_time()) * sin(M_PI*p[0])
            + b1 * M_PI * cos(M_PI*p[0]) * sin(2.0*M_PI*get_time()) * sin(2.0*M_PI*p[1])
            + b2 * 2.0 * M_PI * cos(2.0*M_PI*p[1]) * sin(2.0*M_PI*get_time()) * sin(M_PI*p[0])
            + sigma * sin(2.0*M_PI*get_time()) * sin(M_PI*p[0]) * sin(2.0*M_PI*p[1]);
      
      */
      /*
      return 2.0 * M_PI *
                      cos(2.0 * M_PI * get_time()) *
              sin(M_PI * p[0]) *
              sin(2.0 * M_PI * p[1])
              +
              0.1 * M_PI *
                      sin(2.0 * M_PI * get_time()) *
              cos(M_PI * p[0]) *
              sin(2.0 * M_PI * p[1])
              +
              M_PI * sin(2.0 * M_PI * get_time()) * sin(M_PI * p[0]) *
              (
              5.0 * M_PI * sin(2 * M_PI * p[1]) +
              0.4 * cos(2 * M_PI * p[1])
              );
              
      */
      const double k  = 2.0*M_PI;
      const double t = get_time();
      const double S  = std::sin(k*(p[0]+p[1]));
      const double C  = std::cos(k*(p[0]+p[1]));
      const double T  = std::sin(k*t);
      const double Ct = std::cos(k*t);
      // derivate della u = S*T
      const double ut  = S * k * Ct;
      const double ux  = k * C * T;
      const double uy  = k * C * T;
      const double uxx = -k*k * S * T;
      const double uyy = -k*k * S * T;
      // parametri: mu=3, b=(-2,-3), sigma=2
      return ut - 3.0*(uxx+uyy) + (-2.0)*ux + (-3.0)*uy + 2.0*(S*T);
      
    }
  };
  
  
  // Dirichlet boundary conditions.
  class DirichletFunction : public Function<dim>
  {
  public:
    // Constructor.
    DirichletFunction(){}
    
    // Evaluation.
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      const double k = 2.0*M_PI, t = get_time();
      return std::sin(k*(p[0]+p[1])) * std::sin(k*t);
    }
  };
  
  
  
  
  // Dirichlet boundary conditions.
  class NeumannFunction : public Function<dim>
  {
  public:
    // Constructor.
    NeumannFunction(){}
    
    // Evaluation.
    virtual double
    value(const Point<dim> & p,
          const unsigned int component = 0) const override
    {
      /*ExactSolution exact_solution;
      exact_solution.set_time(get_time());
      return mu * exact_solution.gradient(p)[component] - (component==0 ? b1 : b2)*exact_solution.value(p);
       */
      return 2.0;
    }
  };
  
  
  
  class FunctionU0 : public Function<dim>
  {
  public:
    // Constructor.
    FunctionU0(){}
    
    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };
  
  
  // Exact solution.
  class ExactSolution : public Function<dim> {
  public:
    double value(const Point<dim>& p, unsigned) const override {
      const double k = 2.0*M_PI, t = get_time();
      return std::sin(k*(p[0]+p[1])) * std::sin(k*t);
    }
    Tensor<1,dim> gradient(const Point<dim>& p, unsigned) const override {
      const double k = 2.0*M_PI, t = get_time();
      const double C = std::cos(k*(p[0]+p[1]));
      const double T = std::sin(k*t);
      Tensor<1,dim> g;
      g[0] = k*C*T;          // u_x
      g[1] = k*C*T;          // u_y
      return g;
    }
  };
  
  
  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  Parabolic2D(const std::string  &mesh_file_name_,
              const unsigned int &r_,
              const double       &T_,
              const double       &deltat_,
              const double       &theta_)
          : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
          , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
          , pcout(std::cout, mpi_rank == 0)
          , T(T_)
          , mesh_file_name(mesh_file_name_)
          , r(r_)
          , deltat(deltat_)
          , theta(theta_)
          , mesh(MPI_COMM_WORLD)
  {}
  
  // Initialization.
  void
  setup();
  
  // Solve the problem.
  void
  solve();
  
  // Compute the error.
  double
  compute_error(const VectorTools::NormType &norm_type);

protected:
  // Assemble the mass and stiffness matrices.
  void
  assemble_matrices();
  
  // Assemble the right-hand side of the problem.
  void
  assemble_rhs(const double &time);
  
  // Solve the problem for one time step.
  void
  solve_time_step();
  
  // Output.
  void
  output(const unsigned int &time_step) const;
  
  // MPI parallel. /////////////////////////////////////////////////////////////
  
  // Number of MPI processes.
  const unsigned int mpi_size;
  
  // This MPI process.
  const unsigned int mpi_rank;
  
  // Parallel output stream.
  ConditionalOStream pcout;
  
  // Problem definition. ///////////////////////////////////////////////////////
  
  // mu coefficient.
  DiffusionCoefficient diffusion_coefficient;
  
  AdvectionCoefficient advection_coefficient;
  
  ReactionCoefficient reaction_coefficient;
  
  // Forcing term.
  ForcingTerm forcing_term;
  
  DirichletFunction dirichlet_func;
  
  NeumannFunction neumann_func;
  
  // Exact solution.
  ExactSolution exact_solution;
  
  
  
  FunctionU0 u_0;
  
  // Current time.
  double time;
  
  // Final time.
  const double T;
  
  // Discretization. ///////////////////////////////////////////////////////////
  
  // Mesh file name.
  const std::string mesh_file_name;
  
  // Polynomial degree.
  const unsigned int r;
  
  // Time step.
  const double deltat;
  
  // Theta parameter of the theta method.
  const double theta;
  
  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;
  
  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;
  
  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;
  
  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;
  
  
  // DoF handler.
  DoFHandler<dim> dof_handler;
  
  // DoFs owned by current process.
  IndexSet locally_owned_dofs;
  
  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;
  
  // Mass matrix M / deltat.
  TrilinosWrappers::SparseMatrix mass_matrix;
  
  // Stiffness matrix A.
  TrilinosWrappers::SparseMatrix stiffness_matrix;
  
  // Matrix on the left-hand side (M / deltat + theta A).
  TrilinosWrappers::SparseMatrix lhs_matrix;
  
  // Matrix on the right-hand side (M / deltat - (1 - theta) A).
  TrilinosWrappers::SparseMatrix rhs_matrix;
  
  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::Vector system_rhs;
  
  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;
  
  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;
};

#endif




