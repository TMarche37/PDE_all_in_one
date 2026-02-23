#ifndef STOKES_HPP
#define STOKES_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;


class Stokes
{
public:
  static constexpr unsigned int dim = 2;

  class BoundaryG
  {
  public:
    double value_on_boundary_id(const types::boundary_id id) const
    {
      if (id == 0) return 2.0;
      if (id == 1) return 1.0;
      return 0.0; // Default return to avoid warning
    }
  };

  // Triangular block preconditioner.
  class PreconditionBlockTriangular
  {
  public:
    void initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                    const TrilinosWrappers::SparseMatrix &pressure_mass_,
                    const TrilinosWrappers::SparseMatrix &B_)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;
      B                  = &B_;

      preconditioner_velocity.initialize(*velocity_stiffness);
      preconditioner_pressure.initialize(*pressure_mass);
    }

    void vmult(TrilinosWrappers::MPI::BlockVector       &dst,
               const TrilinosWrappers::MPI::BlockVector &src) const
    {
      // Solve A_u x_u = rhs_u
      SolverControl solver_control_velocity(1000, 1e-2 * src.block(0).l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(solver_control_velocity);

      solver_cg_velocity.solve(*velocity_stiffness,
                               dst.block(0),
                               src.block(0),
                               preconditioner_velocity);

      // rhs_p := src_p - B * x_u
      tmp.reinit(src.block(1));
      B->vmult(tmp, dst.block(0));
      tmp.sadd(-1.0, src.block(1));

      // Solve M_p x_p = rhs_p  (Schur approx.)
      SolverControl solver_control_pressure(1000, 1e-2 * src.block(1).l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(solver_control_pressure);

      solver_cg_pressure.solve(*pressure_mass,
                               dst.block(1),
                               tmp,
                               preconditioner_pressure);
    }

  private:
    const TrilinosWrappers::SparseMatrix *velocity_stiffness = nullptr;
    TrilinosWrappers::PreconditionILU     preconditioner_velocity;

    const TrilinosWrappers::SparseMatrix *pressure_mass = nullptr;
    TrilinosWrappers::PreconditionILU     preconditioner_pressure;

    const TrilinosWrappers::SparseMatrix *B = nullptr;

    mutable TrilinosWrappers::MPI::Vector tmp;
  };

  // Constructor: keep your Stokes args, but add dt and T like Heat.
  Stokes(const std::string  &mesh_file_name_,
         const unsigned int &degree_velocity_,
         const unsigned int &degree_pressure_,
         const double       &delta_t_,
         const double       &T_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , mu(1.0)
    , delta_t(delta_t_)
    , T(T_)
    , mesh_file_name(mesh_file_name_)
    , degree_velocity(degree_velocity_)
    , degree_pressure(degree_pressure_)
    , mesh(MPI_COMM_WORLD)
  {}

 // Setup system.
  void
  setup();

  // Assemble system. We also assemble the pressure mass matrix (needed for the
  // preconditioner).
  void
  assemble();

  // Solve system.
  void
  solve();


  // Output results.
  void output();

  // Run time-stepping loop.
  void run();

protected:
  // MPI
  const unsigned int mpi_size;
  const unsigned int mpi_rank;
  ConditionalOStream pcout;

  // Physical parameters
  const double mu = 1.0;
  const double alpha = 100;

  // Time parameters (like Heat)
  const double delta_t;
  const double T;
  double time = 0.0;
  unsigned int timestep_number = 0;
  

  // Boundary data g
  BoundaryG g_data;

  // Mesh + FE
  const std::string  mesh_file_name;
  const unsigned int degree_velocity;
  const unsigned int degree_pressure;
  double theta = 1.0;

  parallel::fullydistributed::Triangulation<dim> mesh;

  std::unique_ptr<FiniteElement<dim>> fe;
  std::unique_ptr<Quadrature<dim>> quadrature;
  std::unique_ptr<Quadrature<dim-1>> quadrature_face;

  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
  std::vector<IndexSet> block_owned_dofs;
  std::vector<IndexSet> block_relevant_dofs;

  AffineConstraints<double> constraints;

  TrilinosWrappers::BlockSparseMatrix system_matrix;
  TrilinosWrappers::BlockSparseMatrix pressure_mass;

  TrilinosWrappers::MPI::BlockVector system_rhs;

  // Current solution
  TrilinosWrappers::MPI::BlockVector solution_owned;
  TrilinosWrappers::MPI::BlockVector solution;

  // Old solution
  TrilinosWrappers::MPI::BlockVector solution_old_owned;
  TrilinosWrappers::MPI::BlockVector solution_old;
};

#endif


