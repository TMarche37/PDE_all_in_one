#ifndef HEAT_HPP
#define HEAT_HPP

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

#include <deal.II/lac/solver_cg.h>
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
class Heat
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  // Function for the mu coefficient.
  class FunctionMu : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };


    class AdvectionCoefficient: public Function<dim>
    {
    public:
        AdvectionCoefficient(){}

        virtual void
        vector_value(const Point<dim> & /*p*/,
                     Vector<double> &values) const override{
          values[0] = 0.1;
          values[1] = 0.2;
        }

    };


  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return 2.0*M_PI*
              cos(2.0*M_PI*get_time())*
              sin(M_PI*p[0])*
              sin(2.0*M_PI*p[1])
              +
              0.1*M_PI*
              sin(2.0*M_PI*get_time())*
              cos(M_PI*p[0])*
              sin(2.0*M_PI*p[1])
              +
              M_PI*sin(2.0*M_PI*get_time())*sin(M_PI*p[0])*
              (
                      5.0*M_PI*sin(2*M_PI*p[1])+
                      0.4*cos(2*M_PI*p[1])
              )
      ;

      /*return 2*M_PI * cos(2*M_PI*get_time()) * sin(M_PI*p[0]) * sin(2*M_PI*p[1])
             + 5*M_PI*M_PI * sin(2*M_PI*get_time()) * sin(M_PI*p[0]) * sin(2*M_PI*p[1])
             + sin(2*M_PI*get_time())
               * ( 0.1*M_PI* cos(M_PI*p[0]) * sin(2*M_PI*p[1]) + 0.4*M_PI* sin(M_PI*p[0]) * cos(2*M_PI*p[1]));*/
    }
  };

  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {

      return (
              std::sin(2*M_PI*get_time())*
              std::sin(M_PI*p[0])*
              std::sin(2*M_PI*p[1])
      );
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;

      // duex / dx
      result[0] =  M_PI * std::sin(2 * M_PI * get_time()) * std::cos(M_PI * p[0]) *
                   std::sin(2 * M_PI * p[1]);

      // duex / dy
      result[1] = 2 * M_PI * std::sin(2 * M_PI * get_time()) * std::sin(M_PI * p[0]) *
                  std::cos(2 * M_PI * p[1]);
      return result;
    }
  };

  // Dirichlet boundary conditions.
  class ZeroFunction : public Function<dim>
  {
  public:
      // Constructor.
      ZeroFunction()
      {}

      // Evaluation.
      virtual double
      value(const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const override
      {
        return 0.0;
      }
  };


    class U0 : public Function<dim>
    {
    public:
        // Constructor.
        U0()
        {}

        // Evaluation.
        virtual double
        value(const Point<dim> & /*p*/,
              const unsigned int /*component*/ = 0) const override
        {
          return 0.0;
        }
    };

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  Heat(const std::string  &mesh_file_name_,
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
  FunctionMu mu;

  // Forcing term.
  ForcingTerm forcing_term;

  // Exact solution.
  ExactSolution exact_solution;

  AdvectionCoefficient advection_coefficient;

  ZeroFunction zero_function;

  U0 u0;

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