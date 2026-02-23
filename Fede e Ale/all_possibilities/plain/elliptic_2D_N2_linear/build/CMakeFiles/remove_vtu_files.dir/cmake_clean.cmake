file(REMOVE_RECURSE
  "*.pvtu"
  "*.vtk"
  "*.vtu"
  "CMakeFiles/remove_vtu_files"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/remove_vtu_files.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
