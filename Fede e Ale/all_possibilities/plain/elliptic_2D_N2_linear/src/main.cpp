#include "Elliptic2D.hpp"


#include <fstream>
#include <iostream>
#include <deal.II/base/convergence_table.h>
#include <vector>

//----------------------------------------------
//---------------------2D-----------------------
//----------------------------------------------


void mesh_file_function(const bool silent,
                        const unsigned int r,
                        const std::vector<std::string> meshes,
                        const std::vector<double> h_vals)


{
  ConvergenceTable table;
  
  std::vector<double> errors_L2;
  std::vector<double> errors_H1;
  
  for (const auto &mesh : meshes)
  {
    Elliptic2D problem(mesh, r);
    
    std::streambuf* originalBuffer = std::cout.rdbuf();
    std::ofstream nullStream("dev/null");
    
    if (silent){
      std::cout.rdbuf(nullStream.rdbuf());
    }

    problem.setup();
    problem.assemble();
    problem.solve();
    problem.output();
    
    if (silent){
      std::cout.rdbuf(originalBuffer); 
    }
    errors_L2.push_back(problem.compute_error(VectorTools::L2_norm));
    errors_H1.push_back(problem.compute_error(VectorTools::H1_norm));
  }
  
  // Print the errors and estimate the convergence order.
  
  std::cout << "==============================================="
            << std::endl;
  
  
  std::ofstream convergence_file("convergence.csv");
  convergence_file << "h,eL2,eH1" << std::endl;
  
  for (unsigned int i = 0; i < meshes.size(); ++i)
  {
    const double h = h_vals[i];
    convergence_file << h << "," << errors_L2[i] << ","
                     << errors_H1[i] << std::endl;
    
    std::cout << std::scientific << "h = " << std::setw(5)
              << std::setprecision(2) << h;
    
    std::cout << std::scientific << " | eL2 = " << errors_L2[i];
    
    // Estimate the convergence order.
    if (i > 0)
    {
      const double p =
              std::log(errors_L2[i] / errors_L2[i - 1]) /
              std::log(h / h_vals[i-1]);
      
      std::cout << " (" << std::fixed << std::setprecision(2)
                << std::setw(5) << p << ")";
    }
    else
      std::cout << " (  -  )";
    
    std::cout << std::scientific << " | eH1 = " << errors_H1[i];
    
    // Estimate the convergence order.
    if (i > 0)
    {
      const double p =
              std::log(errors_H1[i] / errors_H1[i - 1]) /
              std::log(h / h_vals[i-1]);
      
      std::cout << " (" << std::fixed << std::setprecision(2)
                << std::setw(5) << p << ")";
    }
    else
      std::cout << " (  -  )";
    
    std::cout << " | mesh = " << meshes[i] << "\n";
  }
}


void single_execution_file(const bool silent,
                           const unsigned int r,
                           const std::string mesh)


{
  
  Elliptic2D problem(mesh, r);
  
  std::streambuf* originalBuffer = std::cout.rdbuf();
  std::ofstream nullStream("dev/null");
  
  if (silent){
    std::cout.rdbuf(nullStream.rdbuf());
  }

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();
  
  if (silent){
    std::cout.rdbuf(originalBuffer);
  }
  
}



// Main function.
int
main()
{
  
  const unsigned int r = 2;

  
  const std::vector<std::string> meshes = {
          "../mesh/mesh-square-h0.100000.msh",
          "../mesh/mesh-square-h0.050000.msh",
          "../mesh/mesh-square-h0.025000.msh",
          "../mesh/mesh-square-h0.012500.msh"
  };
  
  const bool silent = false;
  
  
  const std::vector<double> h_vals = {0.1, 0.05, 0.025, 0.0125};
  //mesh_file_function(silent, r,  meshes, h_vals);
  
  // system("rm -rf *.vtu *.pvtu *.vtk");
  
  const std::string mesh = "../mesh/mesh-square-h0.025000.msh";
  
  single_execution_file(silent, r, mesh);

  return 0;
}








