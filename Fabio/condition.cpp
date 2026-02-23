// condition.cpp
// Guide to boundary conditions and lifting across nmpde-labs-aa-25-26.
// Documentation-only file (not part of any build).
// Paths below are relative to nmpde-labs-aa-25-26/.
//
// =====================================================================
// 0) Common building blocks (deal.II patterns used in this repo)
// =====================================================================
// Dirichlet (strong imposition, scalar problems):
//   std::map<types::global_dof_index, double> boundary_values;
//   std::map<types::boundary_id, const Function<dim> *> boundary_functions;
//   boundary_functions[0] = &bc_function;
//   boundary_functions[1] = &bc_function;
//   VectorTools::interpolate_boundary_values(dof_handler,
//                                            boundary_functions,
//                                            boundary_values);
//   MatrixTools::apply_boundary_values(boundary_values,
//                                      system_matrix,
//                                      solution,
//                                      system_rhs,
//                                      true);
//   Examples: lab-01/src/Poisson1D.cpp, lab-02/src/Poisson2D.cpp,
//             exams/gen24/Heat.cpp
//
// Dirichlet with AffineConstraints (useful in parallel or block systems):
//   AffineConstraints<double> constraints;
//   constraints.clear();
//   DoFTools::make_hanging_node_constraints(dof_handler, constraints);
//   VectorTools::interpolate_boundary_values(dof_handler,
//                                            boundary_functions,
//                                            constraints);
//   constraints.close();
//   constraints.distribute_local_to_global(cell_matrix, cell_rhs,
//                                          dof_indices,
//                                          system_matrix, system_rhs);
//   Pattern for constraints usage is visible in stokestimedependet/Stokes.cpp
//   (pressure nullspace); the same mechanism applies to Dirichlet data.
//
// Neumann (natural BC, add boundary integral to RHS):
//   FEFaceValues<dim> fe_face_values(*fe, *quadrature_face,
//                                    update_values |
//                                    update_normal_vectors |
//                                    update_JxW_values);
//   if (cell->face(f)->at_boundary() && is_neumann_id)
//   {
//     fe_face_values.reinit(cell, f);
//     for (unsigned int q = 0; q < n_q_face; ++q)
//       for (unsigned int i = 0; i < dofs_per_cell; ++i)
//         cell_rhs(i) += h(fe_face_values.quadrature_point(q)) *
//                        fe_face_values.shape_value(i, q) *
//                        fe_face_values.JxW(q);
//   }
//   Example: lab-02/src/Poisson2D.cpp, exams/june24/Heat.cpp
//
// Pure Neumann (no Dirichlet anywhere):
//   The linear system is singular. Fix the nullspace by enforcing
//   zero-mean value (or pinning a DoF). For Stokes, the pressure nullspace
//   is fixed by constraints in stokestimedependet/Stokes.cpp.
//
// Lifting idea (Dirichlet):
//   Choose u = v + u_lift with v|Gamma_D = 0.
//   1) Build u_lift in the FE space (interpolation of g, or a separate solve).
//   2) Assemble A for homogeneous Dirichlet.
//   3) Modify RHS: rhs <- rhs - A * u_lift.
//   4) Solve for v, then set u = v + u_lift.
//   This is equivalent to strong imposition if u_lift is chosen as the FE
//   interpolation of g, but it makes the algebra explicit.
//
// Lifting idea (Neumann):
//   Neumann data is already a natural term. If you want to view it as lifting,
//   choose w with grad(w) dot n = g on Gamma_N and solve for v with homogeneous
//   Neumann. In practice, this is exactly the boundary integral added to RHS.
//   For pure Neumann, still fix the mean value.
//
// =====================================================================
// 1) Poisson (scalar elliptic)
//    -div(mu grad u) = f in Omega
//    u = g on Gamma_D
//    mu grad u dot n = h on Gamma_N
// =====================================================================
// 1D (dim=1)
//   - Dirichlet: lab-01/src/Poisson1D.cpp
//   - Mixed DN (domain decomposition): dd_1d/Poisson1D_DN.cpp
//   - Neumann in 1D: add the boundary integral on the endpoint faces
//     (dim-1 = 0, faces are points). The FEFaceValues pattern above applies.
//   - Boundary ids in 1D are usually 0 (left) and 1 (right) if mesh is
//     colorized; see dd_1d/Poisson1D_DN.cpp.
//
// 2D (dim=2)
//   - Dirichlet + Neumann: lab-02/src/Poisson2D.cpp
//     * Dirichlet on boundary ids 0,1 via apply_boundary_values
//     * Neumann on boundary ids 2,3 via FEFaceValues boundary integral
//   - Another 2D reference: exams/feb24/Poisson2D.cpp
//
// 3D (dim=3)
//   - No explicit Poisson3D file in this year, but the code is dimension-agnostic:
//     set dim=3, use QGaussSimplex<dim-1> for faces, and keep the same BC code.
//   - 3D FEValues usage can be seen in lab-03/src/DiffusionReaction.cpp (dim=3)
//     and lab-04/src/Heat.cpp (dim=3).
//
// Lifting for Poisson:
//   - Dirichlet: use u = v + u_lift and modify RHS with -A*u_lift.
//   - Neumann: boundary integral is the lifting term; fix mean if pure Neumann.
//
// =====================================================================
// 2) Heat (scalar parabolic)
//    u_t - div(mu grad u) = f, time discretized by theta method
// =====================================================================
// 1D (dim=1)
//   - exams/june24/Heat.cpp
//     * Dirichlet on boundary id 0 (time-dependent bc_function.set_time(time))
//     * Neumann on boundary id 1 via FEFaceValues boundary integral
//
// 2D (dim=2)
//   - exams/gen24/Heat.cpp
//     * Dirichlet on boundary ids 1..4 via apply_boundary_values
//
// 3D (dim=3)
//   - lab-04/src/Heat.cpp and Heat_convergence/src/Heat.cpp
//     * homogeneous Neumann (do nothing)
//     * for non-homogeneous Neumann, add the boundary integral like Poisson2D
//
// Lifting for Heat:
//   - Dirichlet: u^n = v^n + u_lift^n each time step.
//     Modify RHS with (1/dt) u_lift^n and the theta diffusion terms.
//     If g depends on time, update u_lift^n at every step.
//   - Neumann: add the boundary integral at time t^n (or theta-weighted with
//     t^{n-1} if the scheme needs it). For pure Neumann, fix the mean value.
//
// =====================================================================
// 3) Stokes (steady)
//    -nu laplace u + grad p = f,  div u = 0
// =====================================================================
// Dirichlet for velocity (strong, vector-valued):
//   lab-07/src/Stokes.cpp
//   - use a ComponentMask to select velocity components only
//   - apply boundary values in two passes so walls override inlet at corners
//   Example pattern:
//     ComponentMask mask_velocity(dim + 1, true);
//     mask_velocity.set(dim, false); // pressure component
//     VectorTools::interpolate_boundary_values(dof_handler,
//                                              boundary_functions,
//                                              boundary_values,
//                                              mask_velocity);
//     MatrixTools::apply_boundary_values(boundary_values, system_matrix,
//                                        solution_owned, system_rhs, false);
//
// Neumann (traction) on boundary:
//   lab-07/src/Stokes.cpp
//   - add boundary integral with normal vector:
//     cell_rhs(i) += -p_out * (n dot v_i) * JxW;
//
// Pressure nullspace:
//   - If velocity BCs are all Neumann/traction, pressure is defined up to a
//     constant. Fix with a zero-mean constraint or pin one pressure DoF.
//
// Lifting for Stokes:
//   - Dirichlet: u = v + u_lift, v|Gamma_D = 0, RHS <- RHS - A*u_lift.
//   - Neumann: the traction integral is the lifting term on RHS.
//
// =====================================================================
// 4) Stokes time-dependent
// =====================================================================
// stokestimedependet/Stokes.hpp, stokestimedependet/Stokes.cpp
//   - pure traction BC: (du/dn - p n) = phi with phi = -g n
//   - boundary integral assembled in RHS using FEFaceValues
//   - zero-mean pressure constraint in AffineConstraints (pressure nullspace)
//
// If you need Dirichlet velocity in the time-dependent code:
//   - add VectorTools::interpolate_boundary_values into constraints
//   - apply constraints during assembly with distribute_local_to_global
//   - distribute constraints after the linear solve
//
// =====================================================================
// 5) Quick file map (examples by dimension and BC type)
// =====================================================================
// Poisson:
//   1D Dirichlet:            lab-01/src/Poisson1D.cpp
//   1D Dirichlet-Neumann:    dd_1d/Poisson1D_DN.cpp
//   2D Dirichlet + Neumann:  lab-02/src/Poisson2D.cpp
//   2D exam variant:         exams/feb24/Poisson2D.cpp
//
// Heat:
//   1D Dirichlet + Neumann:  exams/june24/Heat.cpp
//   2D Dirichlet:            exams/gen24/Heat.cpp
//   3D Neumann:              lab-04/src/Heat.cpp
//
// Stokes:
//   steady Dirichlet/Neumann: lab-07/src/Stokes.cpp
//   time-dependent traction:  stokestimedependet/Stokes.cpp
