#include "NonLinearParabolic1D.hpp"

void
NonLinearParabolic1D::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;
    GridGenerator::subdivided_hyper_cube(mesh, N, 0.0, 1.0, true);
    pcout << "  Number of elements = " << mesh.n_active_cells()
          << std::endl;
    
    // Write the mesh to file.
    const std::string mesh_file_name = "../mesh/mesh-" + std::to_string(N) + ".vtk";
    GridOut           grid_out;
    std::ofstream     grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh, grid_out_file);
    pcout << "  Mesh saved to " << mesh_file_name << std::endl;
  }
  
  pcout << "-----------------------------------------------" << std::endl;
  
  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;
    
    fe = std::make_unique < FE_Q < dim >> (r);
    
    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;
    
    quadrature = std::make_unique < QGauss < dim >> (r + 1);
    
    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
    
    quadrature_boundary = std::make_unique < QGauss < dim - 1 >> (r + 1);
    
    std::cout << "  Quadrature points per boundary cell = " << quadrature_boundary->size() << std::endl;
  }
  
  pcout << "-----------------------------------------------" << std::endl;
  
  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;
    
    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);
    
    // We retrieve the set of locally owned DoFs, which will be useful when
    // initializing linear algebra classes.
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    
    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }
  
  pcout << "-----------------------------------------------" << std::endl;
  
  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;
    
    pcout << "  Initializing the sparsity pattern" << std::endl;
    
    // To initialize the sparsity pattern, we use Trilinos' class, that manages
    // some of the inter-process communication.
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    
    // After initialization, we need to call compress, so that all process
    // retrieve the information they need for the rows they own (i.e. the rows
    // corresponding to locally owned DoFs).
    sparsity.compress();
    
    // Then, we use the sparsity pattern to initialize the system matrix. Since
    // the sparsity pattern is partitioned by row, so will the matrix.
    pcout << "  Initializing the system matrix" << std::endl;
    jacobian_matrix.reinit(sparsity);
    
    // Finally, we initialize the right-hand side and solution vectors.
    pcout << "  Initializing the system right-hand side" << std::endl;
    residual_vector.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    delta_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }
}

void
NonLinearParabolic1D::assemble_system()
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);
  
  
  FEFaceValues<dim> fe_values_boundary(*fe,
                                       *quadrature_boundary,
                                       update_values |
                                       update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_residual(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian_matrix = 0.0;
  residual_vector = 0.0;

  // Value and gradient of the solution on current cell.
  std::vector<double>         solution_loc(n_q);
  std::vector<Tensor<1, dim>> solution_gradient_loc(n_q);

  // Value of the solution at previous timestep (un) on current cell.
  std::vector<double> solution_old_loc(n_q);

  forcing_term.set_time(time);
  dirichlet_func.set_time(time);
  neumann_function.set_time(time);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix   = 0.0;
      cell_residual = 0.0;

      fe_values.get_function_values(solution, solution_loc);
      fe_values.get_function_gradients(solution, solution_gradient_loc);
      fe_values.get_function_values(solution_old, solution_old_loc);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Evaluate coefficients on this quadrature node.
          
          
          Tensor<1, dim> b_loc;
          for (unsigned short i = 0; i < dim; i++)
          {
            b_loc[i] = transport_coeff.value(fe_values.quadrature_point(q), i);
          }
          
          const double mu_0_loc = mu_0.value(fe_values.quadrature_point(q));
          const double mu_1_loc = mu_1.value(fe_values.quadrature_point(q));
          const double sigma_loc = reaction_coeff.value(fe_values.quadrature_point(q));
          const double f_loc = forcing_term.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Mass matrix.
                  cell_matrix(i, j) += fe_values.shape_value(i, q)
                                       * fe_values.shape_value(j, q)
                                       / deltat
                                       * fe_values.JxW(q);
                  
                  // Non-linear stiffness matrix, first term.
                  cell_matrix(i, j) += (2.0 * mu_1_loc * fe_values.shape_value(j, q) * solution_loc[q])
                                       * scalar_product(solution_gradient_loc[q], fe_values.shape_grad(i, q))
                                       * fe_values.JxW(q);

                  // Non-linear stiffness matrix, second term.
                  cell_matrix(i, j) += (mu_0_loc + mu_1_loc * solution_loc[q] * solution_loc[q])
                                       * scalar_product(fe_values.shape_grad(j, q), fe_values.shape_grad(i, q))
                                       * fe_values.JxW(q);
                  
                  
                  // Linear terms
                  cell_matrix(i, j) += scalar_product(b_loc, fe_values.shape_grad(j,q))
                                       * fe_values.shape_value(i,q)
                                       * fe_values.JxW(q);
                  
                  cell_matrix(i, j) += sigma_loc * fe_values.shape_value(j,q)
                                       * fe_values.shape_value(i,q)
                                       * fe_values.JxW(q);
                  
                }

              // Assemble the residual vector (with changed sign).
              
              // Time derivative term.
              cell_residual(i) -= (solution_loc[q] - solution_old_loc[q])
                                  / deltat
                                  * fe_values.shape_value(i, q)
                                  * fe_values.JxW(q);

              // Diffusion term.
              cell_residual(i) -= (mu_0_loc + mu_1_loc * solution_loc[q] * solution_loc[q])
                                  * scalar_product(solution_gradient_loc[q], fe_values.shape_grad(i, q))
                                  * fe_values.JxW(q);
              
              
              cell_residual(i) -= scalar_product(b_loc, solution_gradient_loc[q])
                                  * fe_values.shape_value(i,q)
                                  * fe_values.JxW(q);
              
              cell_residual(i) -= sigma_loc * solution_loc[q]
                                  * fe_values.shape_value(i,q)
                                  * fe_values.JxW(q);
              
              // Forcing term.
              cell_residual(i) += f_loc
                                  * fe_values.shape_value(i, q)
                                  * fe_values.JxW(q);
              
            }
        }
      
      neumann_function.set_time(time);
      
      // Neumann Boundary conditions
      if (cell->at_boundary())
      {
        
        for (unsigned int face_number = 0; face_number < cell->n_faces();
             ++face_number)
        {
          
          if (cell->face(face_number)->at_boundary() &&
              (cell->face(face_number)->boundary_id() == 1))
          {
            fe_values_boundary.reinit(cell, face_number);
            
            for (unsigned int q = 0; q < quadrature_boundary->size(); ++q)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_residual(i) += neumann_function.value(fe_values_boundary.quadrature_point(q))
                               * fe_values_boundary.shape_value(i, q)
                               * fe_values_boundary.JxW(q);
            
          }
        }
      }


      cell->get_dof_indices(dof_indices);

      jacobian_matrix.add(dof_indices, cell_matrix);
      residual_vector.add(dof_indices, cell_residual);
    }

  jacobian_matrix.compress(VectorOperation::add);
  residual_vector.compress(VectorOperation::add);

  // We apply Dirichlet boundary conditions.
  // The linear system solution is delta, which is the difference between
  // u_{n+1}^{(k+1)} and u_{n+1}^{(k)}. Both must satisfy the same Dirichlet
  // boundary conditions: therefore, on the boundary, delta = u_{n+1}^{(k+1)} -
  // u_{n+1}^{(k+1)} = 0. We impose homogeneous Dirichlet BCs.
  {
    std::map<types::global_dof_index, double> boundary_values;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    dirichlet_func.set_time(time);

    boundary_functions[0] = &dirichlet_func;
    //boundary_functions[1] = &dirichlet_func;
    
    
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, jacobian_matrix, delta_owned, residual_vector, false);
  }
}

void
NonLinearParabolic1D::solve_linear_system()
{
  SolverControl solver_control(10000, 1e-6);

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(jacobian_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
  pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;
}

void
NonLinearParabolic1D::solve_newton()
{
  const unsigned int n_max_iters        = 1000;
  const double       residual_tolerance = 1e-6;

  unsigned int n_iter        = 0;
  double       residual_norm = residual_tolerance + 1;

  // We apply the boundary conditions to the initial guess (which is stored in
  // solution_owned and solution).
  {
    IndexSet dirichlet_dofs = DoFTools::extract_boundary_dofs(dof_handler);
    dirichlet_dofs          = dirichlet_dofs & dof_handler.locally_owned_dofs();
    
    dirichlet_func.set_time(time);

    TrilinosWrappers::MPI::Vector vector_dirichlet(solution_owned);
    VectorTools::interpolate(dof_handler, dirichlet_func, vector_dirichlet);

    for (const auto &idx : dirichlet_dofs)
      solution_owned[idx] = vector_dirichlet[idx];

    solution_owned.compress(VectorOperation::insert);
    solution = solution_owned;
  }

  while (n_iter < n_max_iters && residual_norm > residual_tolerance)
    {
      assemble_system();
      residual_norm = residual_vector.l2_norm();

      pcout << "  Newton iteration " << n_iter << "/" << n_max_iters
            << " - ||r|| = " << std::scientific << std::setprecision(6)
            << residual_norm << std::flush;

      // We actually solve the system only if the residual is larger than the
      // tolerance.
      if (residual_norm > residual_tolerance)
        {
          solve_linear_system();

          solution_owned += delta_owned;
          solution = solution_owned;
        }
      else
        {
          pcout << " < tolerance" << std::endl;
        }

      ++n_iter;
    }
}

void
NonLinearParabolic1D::output(const unsigned int &time_step) const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record("./", "output", time_step, MPI_COMM_WORLD, 3);
}

void
NonLinearParabolic1D::solve()
{
  pcout << "===============================================" << std::endl;

  time = 0.0;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    output(0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T - 0.5 * deltat)
    {
      time += deltat;
      ++time_step;

      // Store the old solution, so that it is available for assembly.
      solution_old = solution;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << std::fixed << time << std::endl;

      // At every time step, we invoke Newton's method to solve the non-linear
      // problem.
      solve_newton();

      output(time_step);

      pcout << std::endl;
    }
}

double
NonLinearParabolic1D::compute_error(const VectorTools::NormType &norm_type)
{
  FE_Q<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);
  
  
  const QGauss<dim> quadrature_error(r + 2);
  
  exact_solution.set_time(time);
  
  
  // First we compute the norm on each element, and store it in a vector.
  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);
  
  // Then, we add out all the cells.
  const double error = VectorTools::compute_global_error(mesh, error_per_cell, norm_type);
  
  return error;
}