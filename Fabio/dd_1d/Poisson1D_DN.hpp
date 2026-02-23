#ifndef POISSON_1D_DN_HPP
#define POISSON_1D_DN_HPP

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * 1D reaction-diffusion problem:
 *   -u'' + alpha*u = f  on (0,1)
 * with u(0)=u(1)=0.
 *
 * Domain decomposition (Dirichlet-Neumann) with interface x = gamma:
 *   Omega_1 = (0,gamma)  -> Dirichlet at x=gamma
 *   Omega_2 = (gamma,1)  -> Neumann  at x=gamma
 *
 * This class mirrors the structure of Poisson2D.{hpp,cpp} from the labs.
 */
class Poisson1D_DN
{
public:
  static constexpr unsigned int dim = 1;

  // Constructor: subdomain_id must be 0 (left) or 1 (right).
  Poisson1D_DN(const unsigned int &subdomain_id_)
    : subdomain_id(subdomain_id_)
  {}

  void setup();
  void assemble();
  void solve();
  void output(const unsigned int &iter) const;

  void apply_interface_dirichlet(const Poisson1D_DN &other);
  void apply_interface_neumann(Poisson1D_DN &other);

  const Vector<double> &get_solution() const { return solution; }

  void apply_relaxation(const Vector<double> &old_solution, const double &lambda);

protected:
  std::map<types::global_dof_index, types::global_dof_index>
  compute_interface_map(const Poisson1D_DN &other) const;

  const unsigned int subdomain_id;

  // Problem parameters required by the assignment.
  static constexpr double alpha = 1.0;
  static constexpr double gamma = 0.75;

  Triangulation<dim> mesh;

  std::map<types::global_dof_index, Point<dim>> support_points;

  std::unique_ptr<FiniteElement<dim>> fe;
  std::unique_ptr<Quadrature<dim>>    quadrature;

  DoFHandler<dim> dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       system_rhs;
  Vector<double>       solution;
};

#endif
