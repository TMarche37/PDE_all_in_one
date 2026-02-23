#include "Stokes.hpp"

void Stokes::setup()
{
  // -----------------------------
  // 1) Mesh (same as your Stokes)
  // -----------------------------
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data =
      TriangulationDescription::Utilities::create_description_from_triangulation(mesh_serial,
                                                                                MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // -----------------------------
  // 2) FE space (FESystem u+p)
  // -----------------------------
  {
    pcout << "Initializing the finite element space" << std::endl;

    const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
    const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);

    fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity, dim,
                                         fe_scalar_pressure, 1);

    pcout << "  Velocity degree = " << fe_scalar_velocity.degree << std::endl;
    pcout << "  Pressure degree = " << fe_scalar_pressure.degree << std::endl;
    pcout << "  DoFs per cell    = " << fe->dofs_per_cell << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);
    quadrature_face = std::make_unique<QGaussSimplex<dim-1>>(fe->degree + 1);

    pcout << "  Q points/cell = " << quadrature->size() << std::endl;
    pcout << "  Q points/face = " << quadrature_face->size() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // -----------------------------
  // 3) DoFs + block ordering
  // -----------------------------
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // reorder: velocity first, then pressure (same as your Stokes)
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    const auto dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    block_owned_dofs.resize(2);
    block_relevant_dofs.resize(2);

    block_owned_dofs[0] = locally_owned_dofs.get_view(0, n_u);
    block_owned_dofs[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);

    block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
    block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    pcout << "  DoFs: velocity = " << n_u
          << ", pressure = " << n_p
          << ", total = " << (n_u + n_p) << std::endl;
  }
  pcout << "-----------------------------------------------" << std::endl;

  // -----------------------------
  // 5) Matrices + vectors (same style)
  // -----------------------------
  {
    pcout << "Initializing the linear system" << std::endl;

    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        coupling[c][d] = (c == dim && d == dim) ? DoFTools::none : DoFTools::always;

    TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity, constraints, false);
    sparsity.compress();

    // pressure mass sparsity
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        coupling[c][d] = (c == dim && d == dim) ? DoFTools::always : DoFTools::none;

    TrilinosWrappers::BlockSparsityPattern sparsity_pm(block_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity_pm, constraints, false);
    sparsity_pm.compress();

    system_matrix.reinit(sparsity);
    pressure_mass.reinit(sparsity_pm);

    system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);

    solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);

    solution_old_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
    solution_old.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
  }
}

void Stokes::assemble()
{
  pcout << "Assembling system at time t=" << time << std::endl;

  system_matrix = 0.0;
  system_rhs    = 0.0;
  pressure_mass = 0.0;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();
  const unsigned int n_q_face      = quadrature_face->size();

  FEValues<dim> fe_values(*fe, *quadrature,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(*fe, *quadrature_face,
                                   update_values | update_normal_vectors |
                                   update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_mass(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<1, dim>> u_old_q(n_q);
  // Aggiorna ghost values
  solution_old.update_ghost_values();

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_matrix        = 0.0;
    cell_pressure_mass = 0.0;
    cell_rhs           = 0.0;

    fe_values[velocity].get_function_values(solution_old, u_old_q);

    // Matrici locali per M e A
    FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_stiffness(dofs_per_cell, dofs_per_cell);
    cell_mass = 0.0;
    cell_stiffness = 0.0;

    // Termini noti locali f^{n+1} e f^n (qui solo f^n, f^{n+1} va aggiunto se noto)
    Vector<double> cell_rhs_f_n(dofs_per_cell);
    Vector<double> cell_rhs_f_np1(dofs_per_cell);
    cell_rhs_f_n = 0.0;
    cell_rhs_f_np1 = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          // Matrice di massa M
          cell_mass(i,j) += (fe_values[velocity].value(i, q) * fe_values[velocity].value(j, q)) * fe_values.JxW(q);

          // Matrice di stiffness A (tutti i termini tranne la massa)
          // mu(∇u, ∇v)
          cell_stiffness(i,j) += mu * scalar_product(fe_values[velocity].gradient(i, q), fe_values[velocity].gradient(j, q)) * fe_values.JxW(q);
          //alpha 
          cell_stiffness(i,j) += alpha * (fe_values[velocity].value(i, q) * fe_values[velocity].value(j, q)) * fe_values.JxW(q);
          // - (p, div v)
          cell_stiffness(i,j) -= fe_values[velocity].divergence(i, q) * fe_values[pressure].value(j, q) * fe_values.JxW(q);
          // - (q, div u)
          cell_stiffness(i,j) -= fe_values[velocity].divergence(j, q) * fe_values[pressure].value(i, q) * fe_values.JxW(q);

          // pressure mass for preconditioner
          cell_pressure_mass(i,j) += (fe_values[pressure].value(i, q) * fe_values[pressure].value(j, q)) / mu * fe_values.JxW(q);
        }
      }
    }

    // Assemblaggio matrice e rhs secondo lo schema theta
    // Matrice: (M/dt + theta*A)
    cell_matrix = 0.0;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        cell_matrix(i,j) = (1.0/delta_t) * cell_mass(i,j) + theta * cell_stiffness(i,j);

    // Vettore rhs: (M/dt - (1-theta)A)u^n + theta f^{n+1} + (1-theta) f^n
    // Per ora solo la parte con u^n (solution_old)
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      double val = 0.0;
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        val += ((1.0/delta_t) * cell_mass(i,j) - (1.0-theta) * cell_stiffness(i,j)) * solution_old(j);
      cell_rhs(i) = val;
    }
    // Se hai f^n e f^{n+1}, aggiungi qui:
    // cell_rhs.add(theta, cell_rhs_f_np1);
    // cell_rhs.add(1.0-theta, cell_rhs_f_n);
// Boundary integral for Neumann BCs.
      if (cell->at_boundary())
        {
          for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
              if (cell->face(f)->at_boundary() &&
                  (cell->face(f)->boundary_id() == 0 || cell->face(f)->boundary_id() == 1))
                {
                  fe_face_values.reinit(cell, f);

                  for (unsigned int q = 0; q < n_q_face; ++q)
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          cell_rhs(i) +=
                            -BoundaryG().value_on_boundary_id(cell->face(f)->boundary_id()) *
                            fe_face_values[velocity].value(i, q) *
                            fe_face_values.normal_vector(q) *
                            fe_face_values.JxW(q);
                        }
                    }
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
      pressure_mass.add(dof_indices, cell_pressure_mass);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
  pressure_mass.compress(VectorOperation::add);

  // Dirichlet boundary conditions.
  {
    std::map<types::global_dof_index, double>           boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    ComponentMask mask_velocity(dim + 1, true);
    mask_velocity.set(dim, false);
    Functions::ZeroFunction<dim> zero_function(dim + 1);
    // We interpolate first the inlet velocity condition alone, then the wall
    // condition alone, so that the latter "win" over the former where the two
    // boundaries touch.
    boundary_functions[2] = &zero_function;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             mask_velocity);

    boundary_functions.clear();
    
    boundary_functions[3] = &zero_function;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             mask_velocity);

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution_owned, system_rhs, false);
  }
  
}

void Stokes::solve()
{
  SolverControl solver_control(2000, 1e-6 * system_rhs.l2_norm());
  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

  PreconditionBlockTriangular preconditioner;
  preconditioner.initialize(system_matrix.block(0,0),
                            pressure_mass.block(1,1),
                            system_matrix.block(1,0));

  solution_owned = 0.0;
  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);

  pcout << "  " << solver_control.last_step() << "GMRES iterations" << std::endl;

  // Enforce constraints on the final solution vector
  constraints.distribute(solution_owned);

  // Update ghosted vector
  solution = solution_owned;
  
}

void Stokes::output()
{
  DataOut<dim> data_out;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  std::vector<std::string> names(dim, "velocity");
  names.push_back("pressure");

  data_out.add_data_vector(dof_handler, solution, names, interpretation);

  // parallel partition (like your Stokes)
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::string basename = "output-stokes-td";
  data_out.write_vtu_with_pvtu_record("./", basename, timestep_number, MPI_COMM_WORLD);

  pcout << "Output written, step=" << timestep_number
        << ", time=" << time << std::endl;
}

void Stokes::run()
{
  // Heat-style initialization block
  {
    setup();

    // Exam: u0 = 0, so just set everything to 0.
    solution_owned = 0.0;
    solution       = solution_owned;

    solution_old_owned = 0.0;
    solution_old       = solution_old_owned;

    time = 0.0;
    timestep_number = 0;

    output();
  }

  // Heat-style time stepping loop
  while (time < T - 0.5 * delta_t)
  {
    time += delta_t;
    ++timestep_number;

    pcout << "Timestep " << std::setw(3) << timestep_number
          << ", time = " << std::setw(5) << std::fixed << std::setprecision(2)
          << time << " : " << std::endl;

    assemble();
    solve();

    // Update old solution for next step (like Heat’s “solution is old” idea)
    solution_old_owned = solution_owned;
    solution_old       = solution;

    output();
  }
}