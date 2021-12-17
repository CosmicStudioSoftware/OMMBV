"""Provide support for adding OMMBV to NASA Ionospheric Connections Explorer."""

import OMMBV


def add_mag_drift_unit_vectors_ecef(inst, lat_label='latitude',
                                    long_label='longitude',
                                    alt_label='altitude', **kwargs):
    """Add geomagnetic basis vectors expressed in ECEF coordinates.

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument object that will get unit vectors
    lat_label : str
        Label used within `inst` to identify latitude information.
        (default='latitude')
    long_label : str
        Label used within `inst` to identify longitude information.
        (default='longitude')
    alt_label : str
        Label used within `inst` to identify altitude information.
        (default='altitude')
    **kwargs
        Passed along to calculate_mag_drift_unit_vectors_ecef

    Returns
    -------
    None
        unit vectors are added to the passed Instrument object with a naming
        scheme:
            'unit_zon_ecef_*' : unit zonal vector, component along ECEF-(X,Y,Z)
            'unit_fa_ecef_*' : unit field-aligned vector
            'unit_mer_ecef_*' : unit meridional vector

    """

    # add unit vectors for magnetic drifts in ecef coordinates
    (zvx, zvy, zvz,
     bx, by, bz,
     mx, my, mz) = OMMBV.calculate_mag_drift_unit_vectors_ecef(inst[lat_label],
                                                               inst[long_label],
                                                               inst[alt_label],
                                                               inst.index,
                                                               **kwargs)
    # Add data to pysat.Instrument
    inst['unit_zon_ecef_x'] = zvx
    inst['unit_zon_ecef_y'] = zvy
    inst['unit_zon_ecef_z'] = zvz

    inst['unit_fa_ecef_x'] = bx
    inst['unit_fa_ecef_y'] = by
    inst['unit_fa_ecef_z'] = bz

    inst['unit_mer_ecef_x'] = mx
    inst['unit_mer_ecef_y'] = my
    inst['unit_mer_ecef_z'] = mz

    # Add metadata to pysat.Instrument
    info = {'long_name': 'Zonal unit vector along ECEF-x',
            'desc': 'Zonal unit vector along ECEF-x',
            'label': 'Zonal unit vector along ECEF-x',
            'notes': ('Magnetic zonal unit vector expressed using '
                      'Earth Centered Earth Fixed (ECEF) basis. '
                      'Vector system is calculated by determining the '
                      'vector direction that, '
                      'when mapped to the apex, is purely horizontal. '
                      'This component is along the ECEF-x direction.'),
            'axis': 'Zonal unit vector along ECEF-x',
            'value_min': -1.,
            'value_max': 1.,
            }
    inst.meta['unit_zon_ecef_x'] = info

    info = {'long_name': 'Zonal unit vector along ECEF-y',
            'desc': 'Zonal unit vector along ECEF-y',
            'label': 'Zonal unit vector along ECEF-y',
            'notes': ('Magnetic zonal unit vector expressed using '
                      'Earth Centered Earth Fixed (ECEF) basis. '
                      'Vector system is calculated by determining the '
                      'vector direction that, '
                      'when mapped to the apex, is purely horizontal. '
                      'This component is along the ECEF-y direction.'),
            'axis': 'Zonal unit vector along ECEF-y',
            'value_min': -1.,
            'value_max': 1.,
            }
    inst.meta['unit_zon_ecef_y'] = info

    info = {'long_name': 'Zonal unit vector along ECEF-z',
            'desc': 'Zonal unit vector along ECEF-z',
            'label': 'Zonal unit vector along ECEF-z',
            'notes': ('Magnetic zonal unit vector expressed using '
                      'Earth Centered Earth Fixed (ECEF) basis. '
                      'Vector system is calculated by determining the '
                      'vector direction that, '
                      'when mapped to the apex, is purely horizontal. '
                      'This component is along the ECEF-z direction.'),
            'axis': 'Zonal unit vector along ECEF-z',
            'value_min': -1.,
            'value_max': 1.,
            }
    inst.meta['unit_zon_ecef_z'] = info

    info = {'long_name': 'Field-aligned unit vector along ECEF-x',
            'desc': 'Field-aligned unit vector along ECEF-x',
            'label': 'Field-aligned unit vector along ECEF-x',
            'notes': ('Field-aligned unit vector expressed using '
                      'Earth Centered Earth Fixed (ECEF) basis. '
                      'This component is along the ECEF-x direction.'
                      ),
            'axis': 'Field-aligned unit vector along ECEF-x',
            'value_min': -1.,
            'value_max': 1.,
            }
    inst.meta['unit_fa_ecef_x'] = info

    info = {'long_name': 'Field-aligned unit vector along ECEF-y',
            'desc': 'Field-aligned unit vector along ECEF-y',
            'label': 'Field-aligned unit vector along ECEF-y',
            'notes': ('Field-aligned unit vector expressed using '
                      'Earth Centered Earth Fixed (ECEF) basis. '
                      'This component is along the ECEF-y direction.'
                      ),
            'axis': 'Field-aligned unit vector along ECEF-y',
            'value_min': -1.,
            'value_max': 1.,
            }
    inst.meta['unit_fa_ecef_y'] = info

    info = {'long_name': 'Field-aligned unit vector along ECEF-z',
            'desc': 'Field-aligned unit vector along ECEF-z',
            'label': 'Field-aligned unit vector along ECEF-z',
            'notes': ('Field-aligned unit vector expressed using '
                      'Earth Centered Earth Fixed (ECEF) basis. '
                      'This component is along the ECEF-z direction.'
                      ),
            'value_min': -1.,
            'value_max': 1.,
            }
    inst.meta['unit_fa_ecef_z'] = info

    info = {'long_name': 'Meridional unit vector along ECEF-x',
            'desc': 'Meridional unit vector along ECEF-x',
            'label': 'Meridional unit vector along ECEF-x',
            'notes': ('Magnetic meridional unit vector expressed using '
                      'Earth Centered Earth Fixed (ECEF) basis. '
                      'Vector system is calculated by determining the '
                      'magnetic zonal vector direction that, '
                      'when mapped to the apex, is purely horizontal. '
                      'The meridional vector is perpendicular to the zonal '
                      'and field-aligned directions. '
                      'This component is along the ECEF-x direction.'),
            'axis': 'Meridional unit vector along ECEF-x',
            'value_min': -1.,
            'value_max': 1.,
            }
    inst.meta['unit_mer_ecef_x'] = info

    info = {'long_name': 'Meridional unit vector along ECEF-y',
            'desc': 'Meridional unit vector along ECEF-y',
            'label': 'Meridional unit vector along ECEF-y',
            'notes': ('Magnetic meridional unit vector expressed using '
                      'Earth Centered Earth Fixed (ECEF) basis. '
                      'Vector system is calculated by determining the '
                      'magnetic zonal vector direction that, '
                      'when mapped to the apex, is purely horizontal. '
                      'The meridional vector is perpendicular to the zonal '
                      'and field-aligned directions. '
                      'This component is along the ECEF-y direction.'),
            'axis': 'Meridional unit vector along ECEF-y',
            'value_min': -1.,
            'value_max': 1.,
            }
    inst.meta['unit_mer_ecef_y'] = info

    info = {'long_name': 'Meridional unit vector along ECEF-z',
            'desc': 'Meridional unit vector along ECEF-z',
            'label': 'Meridional unit vector along ECEF-z',
            'notes': ('Magnetic meridional unit vector expressed using '
                      'Earth Centered Earth Fixed (ECEF) basis. '
                      'Vector system is calculated by determining the '
                      'magnetic zonal vector direction that, '
                      'when mapped to the apex, is purely horizontal. '
                      'The meridional vector is perpendicular to the zonal '
                      'and field-aligned directions. '
                      'This component is along the ECEF-z direction.'),
            'axis': 'Meridional unit vector along ECEF-z',
            'value_min': -1.,
            'value_max': 1.,
            }
    inst.meta['unit_mer_ecef_z'] = info

    return


def add_mag_drift_unit_vectors(inst, lat_label='latitude',
                               long_label='longitude',
                               alt_label='altitude', **kwargs):
    """Add geomagnetic basis vectors expressed in S/C coordinates.

    Internally, routine calls `add_mag_drift_unit_vectors_ecef`, which
    requires the orientation of the S/C basis vectors in ECEF using the
    following naming, 'sc_*hat_*' where *hat (*=x,y,z) is the S/C basis
    vector and _* (*=x,y,z) is the ECEF direction.

    Parameters
    ----------
    inst : pysat.Instrument object
        Instrument object to be modified
    lat_label : str
        Label used within `inst` to identify latitude information.
        (default='latitude')
    long_label : str
        Label used within `inst` to identify longitude information.
        (default='longitude')
    alt_label : str
        Label used within `inst` to identify altitude information.
        (default='altitude')
    **kwargs
        Passed along to `calculate_mag_drift_unit_vectors_ecef`

    Returns
    -------
    None
        Modifies instrument object in place. Adds 'unit_zon_*' where * = x,y,z
        'unit_fa_*' and 'unit_mer_*' for zonal, field aligned, and meridional
        directions. Note that vector components are expressed in the S/C basis.

    """

    # Vectors are returned in geo/ecef coordinate system
    add_mag_drift_unit_vectors_ecef(inst, lat_label=lat_label,
                                    long_label=long_label,
                                    alt_label=alt_label, **kwargs)

    # Convert them to S/C using transformation supplied by OA
    (inst['unit_zon_x'],
     inst['unit_zon_y'],
     inst['unit_zon_z']) = OMMBV.vector.project_onto_basis(
        inst['unit_zon_ecef_x'], inst['unit_zon_ecef_y'],
        inst['unit_zon_ecef_z'],
        inst['sc_xhat_x'], inst['sc_xhat_y'], inst['sc_xhat_z'],
        inst['sc_yhat_x'], inst['sc_yhat_y'], inst['sc_yhat_z'],
        inst['sc_zhat_x'], inst['sc_zhat_y'], inst['sc_zhat_z'])

    (inst['unit_fa_x'],
     inst['unit_fa_y'],
     inst['unit_fa_z']) = OMMBV.vector.project_onto_basis(
        inst['unit_fa_ecef_x'], inst['unit_fa_ecef_y'],
        inst['unit_fa_ecef_z'],
        inst['sc_xhat_x'], inst['sc_xhat_y'], inst['sc_xhat_z'],
        inst['sc_yhat_x'], inst['sc_yhat_y'], inst['sc_yhat_z'],
        inst['sc_zhat_x'], inst['sc_zhat_y'], inst['sc_zhat_z'])

    (inst['unit_mer_x'],
     inst['unit_mer_y'],
     inst['unit_mer_z']) = OMMBV.vector.project_onto_basis(
        inst['unit_mer_ecef_x'], inst['unit_mer_ecef_y'],
        inst['unit_mer_ecef_z'],
        inst['sc_xhat_x'], inst['sc_xhat_y'], inst['sc_xhat_z'],
        inst['sc_yhat_x'], inst['sc_yhat_y'], inst['sc_yhat_z'],
        inst['sc_zhat_x'], inst['sc_zhat_y'], inst['sc_zhat_z'])

    # Set metadata
    info = {'long_name': 'Zonal direction along IVM-x',
            'desc': 'Unit vector for the zonal geomagnetic direction.',
            'label': 'Zonal Unit Vector: IVM-X component',
            'axis': 'Zonal Unit Vector: IVM-X component',
            'notes': ('The zonal vector is perpendicular to the '
                      'local magnetic field and the magnetic meridional plane. '
                      'The zonal vector maps to purely horizontal '
                      'at the magnetic equator, with positive values '
                      'pointed generally eastward. This vector '
                      'is expressed here in the IVM instrument frame.'
                      'The IVM-x direction points along the instrument '
                      'boresight, which is pointed into ram for '
                      'standard operations.'
                      'Calculated using the corresponding unit vector '
                      'in ECEF and the orientation '
                      'of the IVM also expressed in ECEF (sc_*hat_*).'),
            'scale': 'linear',
            'units': '',
            'value_min': -1.,
            'value_max': 1}
    inst.meta['unit_zon_x'] = info

    info = {'long_name': 'Zonal direction along IVM-y',
            'desc': 'Unit vector for the zonal geomagnetic direction.',
            'label': 'Zonal Unit Vector: IVM-Y component',
            'axis': 'Zonal Unit Vector: IVM-Y component',
            'notes': ('The zonal vector is perpendicular to the '
                      'local magnetic field and the magnetic meridional plane. '
                      'The zonal vector maps to purely horizontal '
                      'at the magnetic equator, with positive values '
                      'pointed generally eastward. '
                      'The unit vector is expressed here in the IVM coordinate '
                      'system, where Y = Z x X, nominally southward when '
                      'in standard pointing, X along ram. '
                      'Calculated using the corresponding unit vector '
                      'in ECEF and the orientation '
                      'of the IVM also expressed in ECEF (sc_*hat_*).'),
            'scale': 'linear',
            'units': '',
            'value_min': -1.,
            'value_max': 1}
    inst.meta['unit_zon_y'] = info

    info = {'long_name': 'Zonal direction along IVM-z',
            'desc': 'Unit vector for the zonal geomagnetic direction.',
            'label': 'Zonal Unit Vector: IVM-Z component',
            'axis': 'Zonal Unit Vector: IVM-Z component',
            'notes': ('The zonal vector is perpendicular to the '
                      'local magnetic field and the magnetic meridional plane. '
                      'The zonal vector maps to purely horizontal '
                      'at the magnetic equator, with positive values '
                      'pointed generally eastward. This vector '
                      'is expressed here in the IVM instrument frame.'
                      'The IVM-Z direction points towards nadir '
                      'when IVM-X is pointed into ram for '
                      'standard operations.'
                      'Calculated using the corresponding unit vector '
                      'in ECEF and the orientation '
                      'of the IVM also expressed in ECEF (sc_*hat_*).'),
            'scale': 'linear',
            'units': '',
            'value_min': -1.,
            'value_max': 1}
    inst.meta['unit_zon_z'] = info

    info = {'long_name': 'Field-aligned direction along IVM-x',
            'desc': 'Unit vector for the geomagnetic field line direction.',
            'label': 'Field Aligned Unit Vector: IVM-X component',
            'axis': 'Field Aligned Unit Vector: IVM-X component',
            'notes': ('The field-aligned vector points along the '
                      'geomagnetic field, with positive values '
                      'along the field direction, and is '
                      'expressed here in the IVM instrument frame. '
                      'The IVM-x direction points along the instrument '
                      'boresight, which is pointed into ram for '
                      'standard operations.'
                      'Calculated using the corresponding unit vector '
                      'in ECEF and the orientation '
                      'of the IVM also expressed in ECEF (sc_*hat_*).'),
            'scale': 'linear',
            'units': '',
            'value_min': -1.,
            'value_max': 1}
    inst.meta['unit_fa_x'] = info

    info = {'long_name': 'Field-aligned direction along IVM-y',
            'desc': 'Unit vector for the geomagnetic field line direction.',
            'label': 'Field Aligned Unit Vector: IVM-Y component',
            'axis': 'Field Aligned Unit Vector: IVM-Y component',
            'notes': ('The field-aligned vector points along the '
                      'geomagnetic field, with positive values '
                      'along the field direction. '
                      'The unit vector is expressed here in the IVM coordinate '
                      'system, where Y = Z x X, nominally southward when '
                      'in standard pointing, X along ram. '
                      'Calculated using the corresponding unit vector '
                      'in ECEF and the orientation '
                      'of the IVM also expressed in ECEF (sc_*hat_*).'),
            'scale': 'linear',
            'units': '',
            'value_min': -1.,
            'value_max': 1}
    inst.meta['unit_fa_y'] = info

    info = {'long_name': 'Field-aligned direction along IVM-z',
            'desc': 'Unit vector for the geomagnetic field line direction.',
            'label': 'Field Aligned Unit Vector: IVM-Z component',
            'axis': 'Field Aligned Unit Vector: IVM-Z component',
            'notes': ('The field-aligned vector points along the '
                      'geomagnetic field, with positive values '
                      'along the field direction, and is '
                      'expressed here in the IVM instrument frame. '
                      'The IVM-Z direction points towards nadir '
                      'when IVM-X is pointed into ram for '
                      'standard operations.'
                      'Calculated using the corresponding unit vector '
                      'in ECEF and the orientation '
                      'of the IVM also expressed in ECEF (sc_*hat_*).'),
            'scale': 'linear',
            'units': '',
            'value_min': -1.,
            'value_max': 1}
    inst.meta['unit_fa_z'] = info

    info = {'long_name': 'Meridional direction along IVM-x',
            'desc': 'Unit vector for the geomagnetic meridional direction.',
            'label': 'Meridional Unit Vector: IVM-X component',
            'axis': 'Meridional Unit Vector: IVM-X component',
            'notes': ('The meridional unit vector is perpendicular to the '
                      'geomagnetic field and maps along magnetic field lines '
                      'to vertical at the magnetic equator, where positive is '
                      'up. The unit vector is expressed here in the IVM '
                      'coordinate system, where x is along the IVM boresight, '
                      'nominally along ram when in standard pointing. '
                      'Calculated using the corresponding unit vector in '
                      'ECEF and the orientation of the IVM also expressed in '
                      'ECEF (sc_*hat_*).'),
            'scale': 'linear',
            'units': '',
            'value_min': -1.,
            'value_max': 1}
    inst.meta['unit_mer_x'] = info

    info = {'long_name': 'Meridional direction along IVM-y',
            'desc': 'Unit vector for the geomagnetic meridional direction.',
            'label': 'Meridional Unit Vector: IVM-Y component',
            'axis': 'Meridional Unit Vector: IVM-Y component',
            'notes': ('The meridional unit vector is perpendicular to the '
                      'geomagnetic field and maps along magnetic field lines '
                      'to vertical at the magnetic equator, where positive '
                      'is up. The unit vector is expressed here in the IVM '
                      'coordinate system, where Y = Z x X, nominally '
                      'southward when in standard pointing, X along ram. '
                      'Calculated using the corresponding unit vector in ECEF '
                      'and the orientation of the IVM also expressed in '
                      'ECEF (sc_*hat_*).'),
            'scale': 'linear',
            'units': '',
            'value_min': -1.,
            'value_max': 1}
    inst.meta['unit_mer_y'] = info

    info = {'long_name': 'Meridional direction along IVM-z',
            'desc': 'Unit vector for the geomagnetic meridional direction.',
            'label': 'Meridional Unit Vector: IVM-Z component',
            'axis': 'Meridional Unit Vector: IVM-Z component',
            'notes': ('The meridional unit vector is perpendicular to the '
                      'geomagnetic field and maps along magnetic field lines '
                      'to vertical at the magnetic equator, where positive '
                      'is up. The unit vector is expressed here in the IVM '
                      'coordinate system, where Z is nadir pointing '
                      '(towards Earth), when in standard pointing, X along '
                      'ram. Calculated using the corresponding unit vector '
                      'in ECEF and the orientation of the IVM also expressed '
                      'in ECEF (sc_*hat_*).'),
            'scale': 'linear',
            'units': '',
            'value_min': -1.,
            'value_max': 1}
    inst.meta['unit_mer_z'] = info

    return


def add_mag_drifts(inst):
    """Add ion drifts expressed along geomagnetic basis vectors.

    Note
    ----
        Requires ion drifts under labels 'iv_*' where * = (x,y,z) along with
        unit vectors labels 'unit_zonal_*', 'unit_fa_*', and 'unit_mer_*',
        where the unit vectors are expressed in S/C coordinates. These
        vectors are calculated by `add_mag_drift_unit_vectors`.

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument object will be modified to include new ion drift magnitudes

    Returns
    -------
    None
        Instrument object modified in place

    """

    info = {'data': inst['unit_zon_x'] * inst['iv_x']
                    + inst['unit_zon_y'] * inst['iv_y']
                    + inst['unit_zon_z'] * inst['iv_z'],
            'units': 'm/s',
            'long_name': 'Zonal ion velocity',
            'notes': ('Ion velocity relative to co-rotation along the magnetic '
                      'zonal direction, normal to local magnetic meridional '
                      'plane and the geomagnetic field (positive east). '
                      'The local zonal vector maps to purely horizontal at the '
                      'magnetic equator. Velocity obtained using ion '
                      'velocities relative to co-rotation in the instrument '
                      'frame along with the corresponding unit vectors '
                      'expressed in the instrument frame to express the '
                      'observed vector along a geomagnetic basis. '),
            'label': 'Zonal Ion Velocity',
            'axis': 'Zonal Ion Velocity',
            'desc': 'Zonal ion velocity',
            'scale': 'Linear',
            'value_min': -500.,
            'value_max': 500.}
    inst['iv_zon'] = info

    info = {'data': inst['unit_fa_x'] * inst['iv_x']
                     + inst['unit_fa_y'] * inst['iv_y']
                     + inst['unit_fa_z'] * inst['iv_z'],
            'units': 'm/s',
            'long_name': 'Field-Aligned ion velocity',
            'notes': ('Ion velocity relative to co-rotation along geomagnetic '
                      'field lines. Positive along the main field vector. ',
                      'Velocity obtained using ion velocities relative '
                      'to co-rotation in the instrument frame along '
                      'with the corresponding unit vectors expressed in '
                      'the instrument frame to express the observed vector '
                      'along a geomagnetic basis. '),
            'label': 'Field-Aligned Ion Velocity',
            'axis': 'Field-Aligned Ion Velocity',
            'desc': 'Field-Aligned Ion Velocity',
            'scale': 'Linear',
            'value_min': -500.,
            'value_max': 500.}
    inst['iv_fa'] = info

    info = {'data': (inst['unit_mer_x'] * inst['iv_x']
                     + inst['unit_mer_y'] * inst['iv_y']
                     + inst['unit_mer_z'] * inst['iv_z'][0]),
            'units': 'm/s',
            'long_name': 'Meridional ion velocity',
            'notes': ('Ion velocity along local magnetic meridional direction, '
                      'perpendicular to geomagnetic field and within local '
                      'magnetic meridional plane. The local meridional vector '
                      'maps to vertical at the magnetic equator, positive is '
                      'up. Velocity obtained using ion velocities relative '
                      'to co-rotation in the instrument frame along '
                      'with the corresponding unit vectors expressed in '
                      'the instrument frame to express the observed vector '
                      'along a geomagnetic basis. '),
            'label': 'Meridional Ion Velocity',
            'axis': 'Meridional Ion Velocity',
            'desc': 'Meridional Ion Velocity',
            'scale': 'Linear',
            'value_min': -500.,
            'value_max': 500.}
    inst['iv_mer'] = info

    return


def add_footpoint_and_equatorial_drifts(inst,
                                        equ_mer_scalar='equ_mer_drifts_scalar',
                                        equ_zonal_scalar=''.join(['equ_zon_',
                                                                  'drifts_',
                                                                  'scalar']),
                                        north_mer_scalar=''.join(['north_foot',
                                                                  'point_mer_',
                                                                  'drifts_',
                                                                  'scalar']),
                                        north_zon_scalar=''.join(['north_foot',
                                                                  'point_zon_',
                                                                  'drifts_',
                                                                  'scalar']),
                                        south_mer_scalar=''.join(['south_foot',
                                                                  'point_mer_',
                                                                  'drifts_',
                                                                  'scalar']),
                                        south_zon_scalar=''.join(['south_foot',
                                                                  'point_zon_',
                                                                  'drifts_',
                                                                  'scalar']),
                                        mer_drift='iv_mer',
                                        zon_drift='iv_zon'):
    """Translate ion velocities to those at footpoints and magnetic equator.

    Note
    ----
        Presumes scalar values for mapping ion velocities are already in the
        `inst`, labeled by 'north_footpoint_zon_drifts_scalar',
        'north_footpoint_mer_drifts_scalar',
        'equ_mer_drifts_scalar', 'equ_zon_drifts_scalar'.

        Also presumes that ion motions in the geomagnetic system are present
        and labeled as 'iv_mer' and 'iv_zon' for meridional and zonal
        ion motions.

        This naming scheme is used by the other pysat oriented routines
        in this package.

    Parameters
    ----------
    inst : pysat.Instrument
    equ_mer_scalar : str
        Label used to identify equatorial scalar for meridional ion drift.
        (default='equ_mer_drifts_scalar')
    equ_zonal_scalar : str
        Label used to identify equatorial scalar for zonal ion drift.
        (default='equ_zon_drifts_scalar')
    north_mer_scalar : str
        Label used to identify northern footpoint scalar for meridional ion
        drift. (default='north_footpoint_mer_drifts_scalar')
    north_zon_scalar : str
        Label used to identify northern footpoint scalar for zonal ion drift.
        (default='north_footpoint_zon_drifts_scalar')
    south_mer_scalar : str
        Label used to identify northern footpoint scalar for meridional ion
        drift. (default='south_footpoint_mer_drifts_scalar')
    south_zon_scalar : str
        Label used to identify southern footpoint scalar for zonal ion drift.
        (default='south_footpoint_zon_drifts_scalar')
    mer_drift : str
        Label used to identify meridional ion drifts within `inst`.
        (default='iv_mer')
    zon_drift : str
        Label used to identify zonal ion drifts within `inst`.
        (default='iv_zon')

    Returns
    -------
    None
        Modifies pysat.Instrument object in place. Drifts mapped to the
        magnetic equator are labeled 'equ_mer_drift' and 'equ_zon_drift'.
        Mappings to the northern and southern footpoints are labeled
        'south_footpoint_mer_drift' and 'south_footpoint_zon_drift'.
        Similarly for the northern hemisphere.

    """
    info = {'data': inst[equ_mer_scalar] * inst[mer_drift],
            'units': 'm/s',
            'long_name': 'Equatorial meridional ion velocity',
            'notes': ('Velocity along local magnetic meridional direction, '
                      'perpendicular to geomagnetic field and within local '
                      'magnetic meridional plane, field-line mapped to '
                      'apex/magnetic equator. The meridional vector is purely '
                      'vertical at the magnetic equator, positive up. '
                      'Velocity obtained using ion velocities relative to '
                      'co-rotation in the instrument frame along with the '
                      'corresponding unit vectors expressed in the instrument '
                      'frame. Field-line mapping and the assumption of '
                      'equi-potential field lines is used to translate the '
                      'locally measured ion motion to the magnetic equator. '
                      'The mapping is used to determine the change in magnetic '
                      'field line distance, which, under assumption of '
                      'equipotential field lines, in turn alters the electric '
                      'field at that location (E=V/d). '),
            'label': 'Equatorial Meridional Ion Velocity',
            'axis': 'Equatorial Meridional Ion Velocity',
            'desc': 'Equatorial Meridional Ion Velocity',
            'scale': 'Linear',
            'value_min': -500.,
            'value_max': 500.}
    inst['equ_mer_drift'] = info

    info = {'data': inst[equ_zonal_scalar] * inst[zon_drift],
            'units': 'm/s',
            'long_name': 'Equatorial zonal ion velocity',
            'notes': ('Velocity along local magnetic zonal direction, '
                      'perpendicular to geomagnetic field and the local '
                      'magnetic meridional plane, field-line mapped to '
                      'apex/magnetic equator. The zonal vector is purely '
                      'horizontal when mapped to the magnetic equator, '
                      'positive is generally eastward. Velocity obtained '
                      'using ion velocities relative to co-rotation in the '
                      'instrument frame along with the corresponding unit '
                      'vectors expressed in the instrument frame. '
                      'Field-line mapping and the assumption of equi-potential '
                      'field lines is used to translate the locally '
                      'measured ion motion to the magnetic equator. The '
                      'mapping is used to determine the change in magnetic '
                      'field line distance, which, under assumption of '
                      'equipotential field lines, in turn alters '
                      'the electric field at that location (E=V/d). '),
            'label': 'Equatorial Zonal Ion Velocity',
            'axis': 'Equatorial Zonal Ion Velocity',
            'desc': 'Equatorial Zonal Ion Velocity',
            'scale': 'Linear',
            'value_min': -500.,
            'value_max': 500.}

    inst['equ_zon_drift'] = info

    notes = ('Velocity along local magnetic meridional direction, '
             'perpendicular to geomagnetic field and within local magnetic '
             'meridional plane, field-line mapped to southern footpoint. '
             'The meridional vector is purely vertical at the magnetic equator,'
             ' positive up. Velocity obtained using ion velocities relative '
             'to co-rotation in the instrument frame along with the '
             'corresponding unit vectors expressed in the instrument frame. '
             'Field-line mapping and the assumption of equi-potential field '
             'lines is used to translate the locally measured ion motion '
             'to the magnetic footpoint. The mapping is used to determine '
             'the change in magnetic field line distance, which, under '
             'assumption of equipotential field lines, in turn alters '
             'the electric field at that location (E=V/d). ')
    info = {'data': inst[south_mer_scalar] * inst[mer_drift],
            'units': 'm/s',
            'long_name': 'Southern meridional ion velocity',
            'notes': notes,
            'label': 'Southern Meridional Ion Velocity',
            'axis': 'Southern Meridional Ion Velocity',
            'desc': 'Southern Meridional Ion Velocity',
            'scale': 'Linear',
            'value_min': -500.,
            'value_max': 500.}
    inst['south_footpoint_mer_drift'] = info

    info = {'data': inst[south_zon_scalar] * inst[zon_drift],
            'units': 'm/s',
            'long_name': 'Southern zonal ion velocity',
            'notes': ('Velocity along local magnetic zonal direction, '
                      'perpendicular to geomagnetic field and the local '
                      'magnetic meridional plane, field-line mapped to '
                      'southern footpoint. The zonal vector is purely '
                      'horizontal when mapped to the magnetic equator, '
                      'positive is generally eastward. '
                      'Velocity obtained using ion velocities relative '
                      'to co-rotation in the instrument frame along '
                      'with the corresponding unit vectors expressed in '
                      'the instrument frame. Field-line mapping and '
                      'the assumption of equi-potential field lines '
                      'is used to translate the locally measured ion '
                      'motion to the southern footpoint. The mapping '
                      'is used to determine the change in magnetic '
                      'field line distance, which, under assumption of '
                      'equipotential field lines, in turn alters '
                      'the electric field at that location (E=V/d). '),
            'label': 'Southern Zonal Ion Velocity',
            'axis': 'Southern Zonal Ion Velocity',
            'desc': 'Southern Zonal Ion Velocity',
            'scale': 'Linear',
            'value_min': -500.,
            'value_max': 500.}
    inst['south_footpoint_zon_drift'] = info

    info = {'data': inst[north_mer_scalar] * inst[mer_drift],
            'units': 'm/s',
            'long_name': 'Northern meridional ion velocity',
            'notes': ('Velocity along local magnetic meridional direction, '
                      'perpendicular to geomagnetic field and within local '
                      'magnetic meridional plane, field-line mapped to '
                      'northern footpoint. '
                      'The meridional vector is purely vertical at '
                      'the magnetic equator, positive up. '
                      'Velocity obtained using ion velocities relative '
                      'to co-rotation in the instrument frame along '
                      'with the corresponding unit vectors expressed in '
                      'the instrument frame. Field-line mapping and '
                      'the assumption of equi-potential field lines '
                      'is used to translate the locally measured ion '
                      'motion to the magnetic footpoint. The mapping '
                      'is used to determine the change in magnetic '
                      'field line distance, which, under assumption of '
                      'equipotential field lines, in turn alters '
                      'the electric field at that location (E=V/d). '),
            'label': 'Northern Meridional Ion Velocity',
            'axis': 'Northern Meridional Ion Velocity',
            'desc': 'Northern Meridional Ion Velocity',
            'scale': 'Linear',
            'value_min': -500.,
            'value_max': 500.}
    inst['north_footpoint_mer_drift'] = info

    info = {'data': inst[north_zon_scalar] * inst[zon_drift],
            'units': 'm/s',
            'long_name': 'Northern zonal ion velocity',
            'notes': ('Velocity along local magnetic zonal direction, '
                      'perpendicular to geomagnetic field and the local '
                      'magnetic meridional plane, field-line mapped to '
                      'northern footpoint. '
                      'The zonal vector is purely horizontal when '
                      'mapped to the magnetic equator, '
                      'positive is generally eastward. '
                      'Velocity obtained using ion velocities relative '
                      'to co-rotation in the instrument frame along '
                      'with the corresponding unit vectors expressed in '
                      'the instrument frame. Field-line mapping and '
                      'the assumption of equi-potential field lines '
                      'is used to translate the locally measured ion '
                      'motion to the northern footpoint. The mapping '
                      'is used to determine the change in magnetic '
                      'field line distance, which, under assumption of '
                      'equipotential field lines, in turn alters '
                      'the electric field at that location (E=V/d). '),
            'label': 'Northern Zonal Ion Velocity',
            'axis': 'Northern Zonal Ion Velocity',
            'desc': 'Northern Zonal Ion Velocity',
            'scale': 'Linear',
            'value_min': -500.,
            'value_max': 500.}
    inst['north_footpoint_zon_drift'] = info

    return
