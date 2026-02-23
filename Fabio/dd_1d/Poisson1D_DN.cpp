#include "Poisson1D_DN.hpp"

void
Poisson1D_DN::setup()
{
  // Create the mesh.
  {
    const unsigned int N_el = 60; // reasonable default; change if needed.

    const double a = (subdomain_id == 0 ? 0.0 : gamma);
    const double b = (subdomain_id == 0 ? gamma : 1.0);

    GridGenerator::subdivided_hyper_cube(mesh, N_el, a, b, /*colorize=*/true);

    const std::string mesh_file_name =
      "mesh-1d-sub" + std::to_string(subdomain_id) + ".vtk";
    GridOut       grid_out;
    std::ofstream grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh, grid_out_file);
  }

  // FE space and quadrature.
  {
    fe         = std::make_unique<FE_SimplexP<dim>>(1);
    quadrature = std::make_unique<QGaussSimplex<dim>>(2);
  }

  // DoF handler + support points.
  {
    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    FE_SimplexP<dim> fe_linear(1);
    MappingFE        mapping(fe_linear);
    support_points = DoFTools::map_dofs_to_support_points(mapping, dof_handler);
  }

  // Linear system.
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
  }
}

void
Poisson1D_DN::assemble()
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Assignment doesn’t specify f explicitly in the statement shown;
          // we take the common non-trivial choice f=1 (constant) so that the
          // solution is not identically zero.
          const double f_loc = 1.0;

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // a(u,v) = int u'v' + alpha int u v
                  cell_matrix(i, j) += (fe_values.shape_grad(i, q) *
                                        fe_values.shape_grad(j, q) +
                                        alpha * fe_values.shape_value(i, q) *
                                          fe_values.shape_value(j, q)) *
                                       fe_values.JxW(q);
                }

              cell_rhs(i) += f_loc * fe_values.shape_value(i, q) *
                             fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);
      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  // Apply ONLY the external Dirichlet boundary condition u=0.
  {
    std::map<types::global_dof_index, double>           boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    Functions::ZeroFunction<dim> bc;

    // With colorize=true on an interval:
    //   boundary id 0 = left endpoint, boundary id 1 = right endpoint.
    // Subdomain 0: external boundary is x=0 (id 0)
    // Subdomain 1: external boundary is x=1 (id 1)
    const types::boundary_id ext_id = (subdomain_id == 0 ? 0 : 1);
    boundary_functions[ext_id]      = &bc;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, false);
  }
}

void
Poisson1D_DN::solve()
{
  SolverControl            solver_control(1000, 1e-12 * (system_rhs.l2_norm() + 1.0));
  SolverCG<Vector<double>> solver(solver_control);

  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}

void
Poisson1D_DN::output(const unsigned int &iter) const
{
  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution, "solution");
  data_out.build_patches();

  const std::string output_file_name =
    "output-1d-sub" + std::to_string(subdomain_id) + "-" +
    std::to_string(iter) + ".vtk";
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);
}

void
Poisson1D_DN::apply_interface_dirichlet(const Poisson1D_DN &other)
{
  const auto interface_map = compute_interface_map(other);

  std::map<types::global_dof_index, double> boundary_values;
  for (const auto &dof : interface_map)
    boundary_values[dof.first] = other.solution[dof.second];

  MatrixTools::apply_boundary_values(
    boundary_values, system_matrix, solution, system_rhs, false);
}

void
Poisson1D_DN::apply_interface_neumann(Poisson1D_DN &other)
{
  const auto interface_map = compute_interface_map(other);

  // Residual trick (same idea as Poisson2D): build weak normal derivative
  // from the residual of the other subproblem, excluding interface conditions.
  Vector<double> interface_residual;
  other.assemble();
  interface_residual = other.system_rhs;
  interface_residual *= -1.0;
  other.system_matrix.vmult_add(interface_residual, other.solution);

  for (const auto &dof : interface_map)
    system_rhs[dof.first] -= interface_residual[dof.second];
}

std::map<types::global_dof_index, types::global_dof_index>
Poisson1D_DN::compute_interface_map(const Poisson1D_DN &other) const
{
  IndexSet current_interface_dofs;
  IndexSet other_interface_dofs;

  // Subdomain 0 interface is the RIGHT endpoint -> boundary id 1
  // Subdomain 1 interface is the LEFT  endpoint -> boundary id 0
  if (subdomain_id == 0)
    {
      current_interface_dofs =
        DoFTools::extract_boundary_dofs(dof_handler, ComponentMask(), {1});
      other_interface_dofs = DoFTools::extract_boundary_dofs(other.dof_handler,
                                                             ComponentMask(),
                                                             {0});
    }
  else
    {
      current_interface_dofs =
        DoFTools::extract_boundary_dofs(dof_handler, ComponentMask(), {0});
      other_interface_dofs = DoFTools::extract_boundary_dofs(other.dof_handler,
                                                             ComponentMask(),
                                                             {1});
    }

  std::map<types::global_dof_index, types::global_dof_index> interface_map;

  for (const auto &dof_current : current_interface_dofs)
    {
      const Point<dim> &p = support_points.at(dof_current);

      types::global_dof_index nearest = *other_interface_dofs.begin();
      for (const auto &dof_other : other_interface_dofs)
        {
          if (p.distance_square(other.support_points.at(dof_other)) <
              p.distance_square(other.support_points.at(nearest)))
            nearest = dof_other;
        }

      interface_map[dof_current] = nearest;
    }

  return interface_map;
}

void
Poisson1D_DN::apply_relaxation(const Vector<double> &old_solution,
                               const double         &lambda)
{
  solution *= lambda;
  solution.add(1.0 - lambda, old_solution);
}
