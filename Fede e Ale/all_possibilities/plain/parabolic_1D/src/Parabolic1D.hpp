#ifndef PARABOLIC_1D_HPP
#define PARABOLIC_1D_HPP



#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>



#include <fstream>
#include <iostream>
#include <math.h>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class Parabolic1D
{
public:
    // Physical dimension (1D, 2D, 3D)
    static constexpr unsigned int dim = 1;

    static constexpr double mu = 8.0;
    static constexpr double b = 0.5;
    static constexpr double sigma = 2.0;


    class DiffusionCoefficient : public Function<dim>
    {
    public:
        DiffusionCoefficient() {}

        // Evaluation.
        virtual double
        value(const Point<dim> & /*p*/,
              const unsigned int /*component*/ = 0) const override
        {
          return mu;
        }
    };


    class AdvectionCoefficient : public Function<dim>
    {
    public:
        AdvectionCoefficient() {}

        // Evaluation.
        virtual double
        value(const Point<dim> & /*p*/,
              const unsigned int /*component*/ = 0) const override
        {
          return b;
        }
    };



    class ReactionCoefficient : public Function<dim>
    {
    public:
        ReactionCoefficient() {}

        // Evaluation.
        virtual double
        value(const Point<dim> & /*p*/,
              const unsigned int /*component*/ = 0) const override
        {
          return sigma;
        }
    };



    //Forcing term.
    class ForcingTerm : public Function<dim>
    {
    public:
        ForcingTerm(){}

        // Evaluation.
        virtual double
        value(const Point<dim> &p,
              const unsigned int /*component*/ = 0) const override
        {
          double exact = std::sin(2.0 * M_PI * p[0]) * get_time();
          double exactD1 = 2*M_PI*std::cos(2.0 * M_PI * p[0]) * get_time();
          double exactD2 = -4*M_PI*M_PI*std::sin(2.0 * M_PI * p[0]) * get_time();
          double exactDT = std::sin(2.0 * M_PI * p[0]);
          return exactDT  -8*exactD2 + 0.5*exactD1 + 2*exact;
        }
    };


    //Dirichlet boundary function
    class DirichletFunc : public Function<dim>
    {
    public:
        DirichletFunc() {}

        // Evaluation.
        virtual double
        value(const Point<dim> & p,
              const unsigned int /*component*/ = 0) const override
        {
          return 0;
        }
    };


    //Neumann boundary function
    class NeumannFunc : public Function<dim>
    {
    public:
        NeumannFunc(){}

        // Evaluation.
        virtual double
        value(const Point<dim> & p,
              const unsigned int /*component*/ = 0) const override
        {
          return 2*M_PI*mu*get_time();
        }
    };


    //Initial condition
    class FunctionU0 : public Function<dim>
    {
    public:
        virtual double
        value(const Point<dim> &p,
              const unsigned int /*component*/ = 0) const override
        {
          return 0.0;
        }
    };



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
          return std::sin(2.0 * M_PI * p[0]) * get_time();
        }

        virtual Tensor<1, dim>
        gradient(const Point<dim> &p,
                 const unsigned int /*component*/ = 0) const override
        {
          Tensor<1, dim> result;
          result[0] = 2*M_PI*std::cos(2.0 * M_PI * p[0]) * get_time();


          return result;
        }


    };

    // Constructor.
    Parabolic1D(const unsigned int &N_,
              const unsigned int &r_,
              const double &T_,
              const double &deltat_,
              const double &theta_)

            : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
            , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
            , pcout(std::cout, mpi_rank == 0)
            , mesh(MPI_COMM_WORLD)
            , N(N_)
            , r(r_)
            , deltat(deltat_)
            , theta(theta_)
            , T(T_)
    {}

    // Initialization.
    void
    setup();

    // Solve the problem.
    void
    solve();

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


    // Triangulation.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // N+1 is the number of elements.
    const unsigned int N;

    // Polynomial degree.
    const unsigned int r;

    // Time step.
    const double deltat;

    // Theta parameter of the theta method.
    const double theta;

    // Final time.
    const double T;


    DiffusionCoefficient diffusion_coefficient;

    AdvectionCoefficient advection_coefficient;

    ReactionCoefficient reaction_coefficient;

    DirichletFunc dirichlet_func;

    NeumannFunc neumann_func;

    FunctionU0 u_0;

    ExactSolution exact_solution;

    double time;

    // Forcing term.
    ForcingTerm forcing_term;

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

    // DoFs owned by current process.
    IndexSet locally_owned_dofs;

    // DoFs relevant to the current process (including ghost DoFs).
    IndexSet locally_relevant_dofs;


    // Sparsity pattern.
    SparsityPattern sparsity_pattern;

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