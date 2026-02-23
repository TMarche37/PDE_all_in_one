#ifndef NON_LINEAR_ELLIPTIC_HPP
#define NON_LINEAR_ELLIPTIC_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class NonLinearElliptic3D
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;
  
  static constexpr double mu0 = 5.0;
  static constexpr double mu1 = 10.0;
  static constexpr double b1 = 0.5;
  static constexpr double b2 = 0.5;
  static constexpr double b3 = 0.5;
  static constexpr double sigma = 3.0;
  static constexpr double dirichlet = 0.0;
  
  // Function for the mu_0 coefficient.
  class FunctionMu0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return mu0;
    }
  };
  
  // Function for the mu_1 coefficient.
  class FunctionMu1 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return mu1;
    }
  };
  
  // Function for the transport coefficient.
  class FunctionTransport : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      if(component == 0) return b1;
      if(component == 1) return b2;
      else return b3;
    }
    
    virtual void
    vector_value(const Point<dim> & p,
                 Vector<double> &values) const override{
      for (unsigned short i = 0; i < dim; i++)
      {
        values[i] = value(p, i);
      }
    }
  };
  
  // Function for the reaction coefficient.
  class FunctionReaction : public Function<dim>
  {
  public:
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
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override {
      
      /*
      const double pi = M_PI;
      double u = std::sin(pi * p[0]) * std::sin(pi * p[1]) * std::sin(pi * p[2]);
      
      double du_dx = pi * std::cos(pi * p[0]) * std::sin(pi * p[1]) * std::sin(pi * p[2]);
      double du_dy = pi * std::sin(pi * p[0]) * std::cos(pi * p[1]) * std::sin(pi * p[2]);
      double du_dz = pi * std::sin(pi * p[0]) * std::sin(pi * p[1]) * std::cos(pi * p[2]);
      
      double laplacian_u = -3 * pi * pi * u;
      double squared_gradient = du_dx * du_dx + du_dy * du_dy + du_dz * du_dz;
      
      double advection = b1*du_dx + b2*du_dy + b3*du_dz;
      
      return -((mu0 + mu1 * u * u) * laplacian_u - 2 * mu1 * u * squared_gradient) + advection + sigma*u;
       */
      
      
      
      double x = p[0], y = p[1], z = p[2];
      double u = x * (1 - x) * y * (1 - y) * z * (1 - z);
      
      double du_dx = (1 - 2 * x) * y * (1 - y) * z * (1 - z);
      double du_dy = x * (1 - x) * (1 - 2 * y) * z * (1 - z);
      double du_dz = x * (1 - x) * y * (1 - y) * (1 - 2 * z);
      
      double d2u_dx2 = -2 * y * (1 - y) * z * (1 - z);
      double d2u_dy2 = -2 * x * (1 - x) * z * (1 - z);
      double d2u_dz2 = -2 * x * (1 - x) * y * (1 - y);
      
      double laplacian_u = d2u_dx2 + d2u_dy2 + d2u_dz2;
      double squared_gradient = du_dx * du_dx + du_dy * du_dy + du_dz * du_dz;
      
      double advection = b1*du_dx + b2*du_dy + b3*du_dz;
      
      //return -((mu0 + mu1 * u * u) * laplacian_u + 2 * mu1 * u * squared_gradient);
      return -((mu0 + mu1 * u * u) * laplacian_u + 2 * mu1 * u * squared_gradient) + advection + sigma*u;
      
    }
  };
  
  
  class DirichletFunction : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return dirichlet;
    }
  };
  
  
  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    // Constructor.
    ExactSolution(){}
    
    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      double x = p[0], y = p[1], z = p[2];
      return x * (1 - x) * y * (1 - y) * z * (1 - z);
      
      //double pi = M_PI;
      //return std::sin(pi * p[0]) * std::sin(pi * p[1]) * std::sin(pi * p[2]);
      
      
      
    }
    
    // Gradient evaluation.
    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;
      
      
      double x = p[0], y = p[1], z = p[2];
      
      result[0] = (1 - 2 * x) * y * (1 - y) * z * (1 - z);
      result[1] = x * (1 - x) * (1 - 2 * y) * z * (1 - z);
      result[2] = x * (1 - x) * y * (1 - y) * (1 - 2 * z);
      
      /*
      double pi = M_PI;
      result[0] = pi * std::cos(pi * p[0]) * std::sin(pi * p[1]) * std::sin(pi * p[2]);
      result[1] = pi * std::sin(pi * p[0]) * std::cos(pi * p[1]) * std::sin(pi * p[2]);
      result[2] = pi * std::sin(pi * p[0]) * std::sin(pi * p[1]) * std::cos(pi * p[2]);
      */
      
      return result;
    }
  };
  
  

  
  
  // Constructor.
  NonLinearElliptic3D(const std::string &mesh_file_name_, const unsigned int &r_)
          : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
          , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
          , pcout(std::cout, mpi_rank == 0)
          , mesh_file_name(mesh_file_name_)
          , r(r_)
          , mesh(MPI_COMM_WORLD)
  {}
  
  // Initialization.
  void
  setup();
  
  // Solve the problem using Newton's method.
  void
  solve_newton();
  
  // Output.
  void
  output() const;
  
  double
  compute_error(const VectorTools::NormType &norm_type) const;

protected:
  // Assemble the tangent problem.
  void
  assemble_system();
  
  // Solve the tangent problem.
  void
  solve_system();
  
  // MPI parallel. /////////////////////////////////////////////////////////////
  
  // Number of MPI processes.
  const unsigned int mpi_size;
  
  // This MPI process.
  const unsigned int mpi_rank;
  
  // Parallel output stream.
  ConditionalOStream pcout;
  
  // Problem definition. ///////////////////////////////////////////////////////
  
  // mu_0 coefficient.
  FunctionMu0 mu_0;
  
  // mu_1 coefficient.
  FunctionMu1 mu_1;
  
  FunctionTransport transport_coeff;
  
  FunctionReaction reaction_coeff;
  
  // Forcing term.
  ForcingTerm forcing_term;
  
  ExactSolution exact_solution;
  
  DirichletFunction dirichlet_function;
  
  // Discretization. ///////////////////////////////////////////////////////////
  
  // Mesh file name.
  const std::string mesh_file_name;
  
  // Polynomial degree.
  const unsigned int r;
  
  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;
  
  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;
  
  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;
  
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;
  
  // DoF handler.
  DoFHandler<dim> dof_handler;
  
  // DoFs owned by current process.
  IndexSet locally_owned_dofs;
  
  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;
  
  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix jacobian_matrix;
  
  // Residual vector.
  TrilinosWrappers::MPI::Vector residual_vector;
  
  // Solution increment (without ghost elements).
  TrilinosWrappers::MPI::Vector delta_owned;
  
  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;
  
  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;
};

#endif