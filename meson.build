project('OMMBV', 'c',
  version : '1.0.2',
  license: 'BSD-3',
  meson_version: '>=0.64.0',
  default_options : [
    'warning_level=2',
    'fortran_args=-Wno-line-truncation -Wno-conversion -Wno-unused-variable -Wno-maybe-uninitialized -Wno-unused-dummy-argument',
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

py.extension_module('OMMBV.igrf',
  [
    'OMMBV/igrf13.f'
  ],
  incdir_f2py / 'fortranobject.c',
  f_args : f_flags,
  include_directories: inc_np,
  dependencies : py_dep,
  install : true
)

py.extension_module('OMMBV.sources',
  [
    'OMMBV/sources.f',
    'OMMBV/igrf13.f'
  ],
  incdir_f2py / 'fortranobject.c',
  f_args : f_flags,
  include_directories: inc_np,
  dependencies : py_dep,
  install : true
)

py.extension_module('OMMBV.fortran_coords',
  [
    'OMMBV/_coords.f'
  ],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  install : true
)
