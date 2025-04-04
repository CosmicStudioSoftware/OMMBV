project('OMMBV', 'c',
  version : '1.1.0',
  license: 'BSD-3',
  meson_version: '>=0.64.0',
  default_options : [
    'warning_level=2',
    'c_args=-Wno-unused-parameter -Wno-cast-function-type -Wno-missing-field-initializers',
    'fortran_args=-Wno-line-truncation -Wno-conversion -Wno-unused-variable -Wno-maybe-uninitialized -Wno-unused-dummy-argument -Wno-compare-reals',
    'fortran_std=legacy'],
)

add_languages('fortran', native: false)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

# Unlike distutils, meson doesn't yet include some of the f2py stuff
fortranobject_c = incdir_f2py / 'fortranobject.c'

fortranobject_lib = static_library('_fortranobject',
  fortranobject_c,
  dependencies: py_dep,
  include_directories: [incdir_numpy, incdir_f2py])

fortranobject_dep = declare_dependency(
  link_with: fortranobject_lib,
  include_directories: [incdir_numpy, incdir_f2py])


igrf_source = custom_target('igrfmodule.c',
  input : ['ommbvfortran/igrf14.f'],  # .f so no F90 wrappers
  output : ['igrfmodule.c', 'igrf-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'igrf', '--lower']
)
py.extension_module('igrf',
  [
    'ommbvfortran/igrf14.f',
    igrf_source,
    fortranobject_c
  ],
  include_directories: inc_np,
  link_with: fortranobject_lib,
  dependencies : [py_dep, fortranobject_dep],
  subdir: 'OMMBV',
  install : true
)


sources_source = custom_target('sourcesmodule.c',
  input : ['ommbvfortran/sources.f', 'OMMBV/igrf14.f'],  # .f so no F90 wrappers
  output : ['sourcesmodule.c', 'sources-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'sources', '--lower']
)
py.extension_module('sources',
  [
   'ommbvfortran/sources.f',
   'ommbvfortran/igrf14.f',
   sources_source,
   fortranobject_c
 ],
  include_directories: inc_np,
  link_with: fortranobject_lib,
  dependencies : [py_dep, fortranobject_dep],
  subdir: 'OMMBV',
  install : true
)


fcoords_source = custom_target('fortran_coordsmodule.c',
  input : ['ommbvfortran/_coords.f'],  # .f so no F90 wrappers
  output : ['fortran_coordsmodule.c', 'fortran_coords-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'fortran_coords', '--lower']
)
py.extension_module('fortran_coords',
  [
    'ommbvfortran/_coords.f',
    fcoords_source,
    fortranobject_c
  ],
  include_directories: inc_np,
  link_with: fortranobject_lib,
  dependencies : [py_dep, fortranobject_dep],
  subdir: 'OMMBV',
  install : true
)

py.install_sources(
  'OMMBV/_core.py',
  'OMMBV/heritage.py',
  'OMMBV/satellite.py',
  'OMMBV/trace.py',
  'OMMBV/trans.py',
  'OMMBV/utils.py',
  'OMMBV/vector.py',
  'OMMBV/__init__.py',
  pure: false,
  subdir: 'OMMBV'
)