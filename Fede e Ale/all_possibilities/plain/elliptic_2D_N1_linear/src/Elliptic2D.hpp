#ifndef ELLIPTIC_2D_HPP
#define ELLIPTIC_2D_HPP

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;


class Elliptic2D
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;
  
  static constexpr double mu = 1;
  static constexpr double b1 = -1;
  static constexpr double b2 = -1;
  static constexpr double sigma = 1;


  class DiffusionCoefficient : public Function<dim>
  {
  public:
    DiffusionCoefficient(){}

    // Evaluation.
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

    virtual double 
    value(const Point<dim> &/*p*/,
          const unsigned int component) const override
    {
        if(component == 0) return b1;
        if(component == 1) return b2;
        else return 0;
    }
    
    virtual void
    vector_value(const Point<dim> &/*p*/,
                 Vector<double> &values) const override
    {
      values[0] = b1;
      values[1] = b2;
    }
  };

  
  // Reaction coefficient.
  class ReactionCoefficient : public Function<dim>
  {
  public:
    ReactionCoefficient(){}

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return sigma;
    }
  };


  
  // Forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    ForcingTerm(){}

    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      ExactSolution exact_solution;
      double exactValue = exact_solution.value(p);
      Tensor<1,dim> exactGrad = exact_solution.gradient(p);
      return -mu *
         (
             0
         )
         + b1*exactGrad[0] + b2*exactGrad[1] + sigma*exactValue;
    }
  };

  // Dirichlet boundary conditions.
  class DirichletFunction : public Function<dim>
  {
  public:
    DirichletFunction(){}

    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      ExactSolution exact_solution;
      return exact_solution.value(p);
    }
  };


  class NeumannFunction1 : public Function<dim>
  {
  public:
      NeumannFunction1(){}

      // Evaluation.
      virtual double
      value(const Point<dim> & p,
            const unsigned int /*component*/ = 0) const override
      {
        ExactSolution exact_solution;
        return mu * exact_solution.gradient(p)[0] - b1*exact_solution.value(p);
      }
  };
  class NeumannFunction3 : public Function<dim>
  {
  public:
      NeumannFunction3(){}

      // Evaluation.
      virtual double
      value(const Point<dim> & p,
            const unsigned int /*component*/ = 0) const override
      {
        ExactSolution exact_solution;
        return mu * exact_solution.gradient(p)[1] - b2*exact_solution.value(p);
      }
  };

  static constexpr double offset = 0;

  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(){}

    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return p[0] * p[1] + offset;
    }

    // Gradient evaluation.
    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;

      result[0] = p[1];

      result[1] = p[0];

      return result;
    }
  };

  // Constructor.
  Elliptic2D(const std::string &mesh_file_name_, const unsigned int &r_)
    : mesh_file_name(mesh_file_name_)
    , r(r_)
  {}

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output() const;

  // Compute the error.
  double
  compute_error(const VectorTools::NormType &norm_type) const;

protected:
  // Path to the mesh file.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Diffusion coefficient.
  DiffusionCoefficient diffusion_coefficient;

  // Reaction coefficient.
  ReactionCoefficient reaction_coefficient;
  
  AdvectionCoefficient advection_coefficient;

  // Forcing term.
  ForcingTerm forcing_term;

  DirichletFunction dirichlet_func;

  NeumannFunction1 neumann_func1;
  NeumannFunction3 neumann_func3;
  
  ExactSolution exact_solution;

  // Triangulation.
  Triangulation<dim> mesh;

  // Finite element space.
  // We use a unique_ptr here so that we can choose the type and degree of the
  // finite elements at runtime (the degree is a constructor parameter). The
  // class FiniteElement<dim> is an abstract class from which all types of
  // finite elements implemented by deal.ii inherit.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  // We use a unique_ptr here so that we can choose the type and order of the
  // quadrature formula at runtime (the order is a constructor parameter).
  std::unique_ptr<Quadrature<dim>> quadrature;

  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;


    // DoF handler.
  DoFHandler<dim> dof_handler;

  // Sparsity pattern.
  SparsityPattern sparsity_pattern;

  // System matrix.
  SparseMatrix<double> system_matrix;

  // System right-hand side.
  Vector<double> system_rhs;

  // System solution.
  Vector<double> solution;
};

#endif