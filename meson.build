project('OMMBV', 'c',
  version : '1.0.2',
  license: 'BSD-3',
  meson_version: '>=0.64.0',
  default_options : [
    'warning_level=2',
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


igrf_source = custom_target('igrfmodule.c',
  input : ['OMMBV/igrf13.f'],  # .f so no F90 wrappers
  output : ['igrfmodule.c', 'igrf-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'igrf', '--lower']
)
py.extension_module('igrf',
  [
    'OMMBV/igrf13.f',
    igrf_source
  ],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  install : true
)

sources_source = custom_target('sourcesmodule.c',
  input : ['OMMBV/sources.f', 'OMMBV/igrf13.f'],  # .f so no F90 wrappers
  output : ['sourcesmodule.c', 'sources-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'igrf', '--lower']
)
py.extension_module('sources',
  [
    'OMMBV/sources.f',
    'OMMBV/igrf13.f',
    sources_source
  ],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  install : true
)


fcoords_source = custom_target('_coordsmodule.c',
  input : ['OMMBV/_coords.f'],  # .f so no F90 wrappers
  output : ['_coordsmodule.c', '_coords-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'igrf', '--lower']
)
py.extension_module('fortran_coords',
  [
    'OMMBV/_coords.f'
  ],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  install : true
)
