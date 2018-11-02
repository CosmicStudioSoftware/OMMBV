from . import *

def add_mag_drift_unit_vectors_ecef(inst, steps=None, max_steps=40000, step_size=10.,
                                    ref_height=120.):
    """Adds unit vectors expressing the ion drift coordinate system
    organized by the geomagnetic field. Unit vectors are expressed
    in ECEF coordinates.
    
    Parameters
    ----------
    inst : pysat.Instrument
        Instrument object that will get unit vectors
    max_steps : int
        Maximum number of steps allowed for field line tracing
    step_size : float
        Maximum step size (km) allowed when field line tracing
    ref_height : float
        Altitude used as cutoff for labeling a field line location a footpoint
        
    Returns
    -------
    None
        unit vectors are added to the passed Instrument object with a naming 
        scheme:
            'unit_zon_ecef_*' : unit zonal vector, component along ECEF-(X,Y,or Z)
            'unit_fa_ecef_*' : unit field-aligned vector, component along ECEF-(X,Y,or Z)
            'unit_mer_ecef_*' : unit meridional vector, component along ECEF-(X,Y,or Z)
            
    """

    # add unit vectors for magnetic drifts in ecef coordinates
    zvx, zvy, zvz, bx, by, bz, mx, my, mz = calculate_mag_drift_unit_vectors_ecef(inst['latitude'], 
                                                            inst['longitude'], inst['altitude'], inst.data.index,
                                                            steps=steps, max_steps=max_steps, step_size=step_size, ref_height=ref_height)
    
    inst['unit_zon_ecef_x'] = zvx
    inst['unit_zon_ecef_y'] = zvy
    inst['unit_zon_ecef_z'] = zvz

    inst['unit_fa_ecef_x'] = bx
    inst['unit_fa_ecef_y'] = by
    inst['unit_fa_ecef_z'] = bz

    inst['unit_mer_ecef_x'] = mx
    inst['unit_mer_ecef_y'] = my
    inst['unit_mer_ecef_z'] = mz

    inst.meta['unit_zon_ecef_x'] = {'long_name': 'Zonal unit vector along ECEF-x',
                                    'desc': 'Zonal unit vector along ECEF-x',
                                    'label': 'Zonal unit vector along ECEF-x',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Zonal unit vector along ECEF-x',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_zon_ecef_y'] = {'long_name': 'Zonal unit vector along ECEF-y',
                                    'desc': 'Zonal unit vector along ECEF-y',
                                    'label': 'Zonal unit vector along ECEF-y',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Zonal unit vector along ECEF-y',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_zon_ecef_z'] = {'long_name': 'Zonal unit vector along ECEF-z',
                                    'desc': 'Zonal unit vector along ECEF-z',
                                    'label': 'Zonal unit vector along ECEF-z',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Zonal unit vector along ECEF-z',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }

    inst.meta['unit_fa_ecef_x'] = {'long_name': 'Field-aligned unit vector along ECEF-x',
                                    'desc': 'Field-aligned unit vector along ECEF-x',
                                    'label': 'Field-aligned unit vector along ECEF-x',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Field-aligned unit vector along ECEF-x',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_fa_ecef_y'] = {'long_name': 'Field-aligned unit vector along ECEF-y',
                                    'desc': 'Field-aligned unit vector along ECEF-y',
                                    'label': 'Field-aligned unit vector along ECEF-y',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Field-aligned unit vector along ECEF-y',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_fa_ecef_z'] = {'long_name': 'Field-aligned unit vector along ECEF-z',
                                    'desc': 'Field-aligned unit vector along ECEF-z',
                                    'label': 'Field-aligned unit vector along ECEF-z',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Field-aligned unit vector along ECEF-z',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }

    inst.meta['unit_mer_ecef_x'] = {'long_name': 'Meridional unit vector along ECEF-x',
                                    'desc': 'Meridional unit vector along ECEF-x',
                                    'label': 'Meridional unit vector along ECEF-x',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Meridional unit vector along ECEF-x',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_mer_ecef_y'] = {'long_name': 'Meridional unit vector along ECEF-y',
                                    'desc': 'Meridional unit vector along ECEF-y',
                                    'label': 'Meridional unit vector along ECEF-y',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Meridional unit vector along ECEF-y',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_mer_ecef_z'] = {'long_name': 'Meridional unit vector along ECEF-z',
                                    'desc': 'Meridional unit vector along ECEF-z',
                                    'label': 'Meridional unit vector along ECEF-z',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Meridional unit vector along ECEF-z',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }

    return


def add_mag_drift_unit_vectors(inst, max_steps=40000, step_size=10.):
    """Add unit vectors expressing the ion drift coordinate system
    organized by the geomagnetic field. Unit vectors are expressed
    in S/C coordinates.
    
    Interally, routine calls add_mag_drift_unit_vectors_ecef. 
    See function for input parameter description.
    Requires the orientation of the S/C basis vectors in ECEF using naming,
    'sc_xhat_x' where *hat (*=x,y,z) is the S/C basis vector and _* (*=x,y,z)
    is the ECEF direction. 
    
    Parameters
    ----------
    inst : pysat.Instrument object
        Instrument object to be modified
    max_steps : int
        Maximum number of steps taken for field line integration
    step_size : float
        Maximum step size (km) allowed for field line tracer
    
    Returns
    -------
    None
        Modifies instrument object in place. Adds 'unit_zon_*' where * = x,y,z
        'unit_fa_*' and 'unit_mer_*' for zonal, field aligned, and meridional
        directions. Note that vector components are expressed in the S/C basis.
        
    """

    # vectors are returned in geo/ecef coordinate system
    add_mag_drift_unit_vectors_ecef(inst, max_steps=max_steps, step_size=step_size)
    # convert them to S/C using transformation supplied by OA
    inst['unit_zon_x'], inst['unit_zon_y'], inst['unit_zon_z'] = project_ecef_vector_onto_basis(inst['unit_zon_ecef_x'], inst['unit_zon_ecef_y'], inst['unit_zon_ecef_z'],
                                                                                                inst['sc_xhat_x'], inst['sc_xhat_y'], inst['sc_xhat_z'],
                                                                                                inst['sc_yhat_x'], inst['sc_yhat_y'], inst['sc_yhat_z'],
                                                                                                inst['sc_zhat_x'], inst['sc_zhat_y'], inst['sc_zhat_z'])
    inst['unit_fa_x'], inst['unit_fa_y'], inst['unit_fa_z'] = project_ecef_vector_onto_basis(inst['unit_fa_ecef_x'], inst['unit_fa_ecef_y'], inst['unit_fa_ecef_z'],
                                                                                                inst['sc_xhat_x'], inst['sc_xhat_y'], inst['sc_xhat_z'],
                                                                                                inst['sc_yhat_x'], inst['sc_yhat_y'], inst['sc_yhat_z'],
                                                                                                inst['sc_zhat_x'], inst['sc_zhat_y'], inst['sc_zhat_z'])
    inst['unit_mer_x'], inst['unit_mer_y'], inst['unit_mer_z'] = project_ecef_vector_onto_basis(inst['unit_mer_ecef_x'], inst['unit_mer_ecef_y'], inst['unit_mer_ecef_z'],
                                                                                                inst['sc_xhat_x'], inst['sc_xhat_y'], inst['sc_xhat_z'],
                                                                                                inst['sc_yhat_x'], inst['sc_yhat_y'], inst['sc_yhat_z'],
                                                                                                inst['sc_zhat_x'], inst['sc_zhat_y'], inst['sc_zhat_z'])

    inst.meta['unit_zon_x'] = { 'long_name':'Zonal direction along IVM-x',
                                'desc': 'Unit vector for the zonal geomagnetic direction.',
                                'label': 'Zonal Unit Vector: IVM-X component',
                                'axis': 'Zonal Unit Vector: IVM-X component',
                                'notes': ('Positive towards the east. Zonal vector is normal to magnetic meridian plane. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_zon_y'] = {'long_name':'Zonal direction along IVM-y',
                                'desc': 'Unit vector for the zonal geomagnetic direction.',
                                'label': 'Zonal Unit Vector: IVM-Y component',
                                'axis': 'Zonal Unit Vector: IVM-Y component',
                                'notes': ('Positive towards the east. Zonal vector is normal to magnetic meridian plane. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_zon_z'] = {'long_name':'Zonal direction along IVM-z',
                                'desc': 'Unit vector for the zonal geomagnetic direction.',
                                'label': 'Zonal Unit Vector: IVM-Z component',
                                'axis': 'Zonal Unit Vector: IVM-Z component',
                                'notes': ('Positive towards the east. Zonal vector is normal to magnetic meridian plane. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}

    inst.meta['unit_fa_x'] = {'long_name':'Field-aligned direction along IVM-x',
                                'desc': 'Unit vector for the geomagnetic field line direction.',
                                'label': 'Field Aligned Unit Vector: IVM-X component',
                                'axis': 'Field Aligned Unit Vector: IVM-X component',
                                'notes': ('Positive along the field, generally northward. Unit vector is along the geomagnetic field. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_fa_y'] = {'long_name':'Field-aligned direction along IVM-y',
                                'desc': 'Unit vector for the geomagnetic field line direction.',
                                'label': 'Field Aligned Unit Vector: IVM-Y component',
                                'axis': 'Field Aligned Unit Vector: IVM-Y component',
                                'notes': ('Positive along the field, generally northward. Unit vector is along the geomagnetic field. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_fa_z'] = {'long_name':'Field-aligned direction along IVM-z',
                                'desc': 'Unit vector for the geomagnetic field line direction.',
                                'label': 'Field Aligned Unit Vector: IVM-Z component',
                                'axis': 'Field Aligned Unit Vector: IVM-Z component',
                                'notes': ('Positive along the field, generally northward. Unit vector is along the geomagnetic field. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}

    inst.meta['unit_mer_x'] = {'long_name':'Meridional direction along IVM-x',
                                'desc': 'Unit vector for the geomagnetic meridional direction.',
                                'label': 'Meridional Unit Vector: IVM-X component',
                                'axis': 'Meridional Unit Vector: IVM-X component',
                                'notes': ('Positive is aligned with vertical at '
                                          'geomagnetic equator. Unit vector is perpendicular to the geomagnetic field '
                                          'and in the plane of the meridian.'
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_mer_y'] = {'long_name':'Meridional direction along IVM-y',
                                'desc': 'Unit vector for the geomagnetic meridional direction.',
                                'label': 'Meridional Unit Vector: IVM-Y component',
                                'axis': 'Meridional Unit Vector: IVM-Y component',
                                'notes': ('Positive is aligned with vertical at '
                                          'geomagnetic equator. Unit vector is perpendicular to the geomagnetic field '
                                          'and in the plane of the meridian.'
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_mer_z'] = {'long_name':'Meridional direction along IVM-z',
                                'desc': 'Unit vector for the geomagnetic meridional direction.',
                                'label': 'Meridional Unit Vector: IVM-Z component',
                                'axis': 'Meridional Unit Vector: IVM-Z component',
                                'notes': ('Positive is aligned with vertical at '
                                          'geomagnetic equator. Unit vector is perpendicular to the geomagnetic field '
                                          'and in the plane of the meridian.'
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}

    return


def add_mag_drifts(inst):
    """Adds ion drifts in magnetic coordinates using ion drifts in S/C coordinates
    along with pre-calculated unit vectors for magnetic coordinates.
    
    Note
    ----
        Requires ion drifts under labels 'iv_*' where * = (x,y,z) along with
        unit vectors labels 'unit_zonal_*', 'unit_fa_*', and 'unit_mer_*',
        where the unit vectors are expressed in S/C coordinates. These
        vectors are calculated by add_mag_drift_unit_vectors.
    
    Parameters
    ----------
    inst : pysat.Instrument
        Instrument object will be modified to include new ion drift magnitudes
        
    Returns
    -------
    None
        Instrument object modified in place
    
    """
    
    inst['iv_zon'] = {'data':inst['unit_zon_x'] * inst['iv_x'] + inst['unit_zon_y']*inst['iv_y'] + inst['unit_zon_z']*inst['iv_z'],
                      'units':'m/s',
                      'long_name':'Zonal ion velocity',
                      'notes':('Ion velocity relative to co-rotation along zonal '
                               'direction, normal to meridional plane. Positive east. '
                               'Velocity obtained using ion velocities relative '
                               'to co-rotation in the instrument frame along '
                               'with the corresponding unit vectors expressed in '
                               'the instrument frame. '),
                      'label': 'Zonal Ion Velocity',
                      'axis': 'Zonal Ion Velocity',
                      'desc': 'Zonal ion velocity',
                      'scale': 'Linear',
                      'value_min':-500., 
                      'value_max':500.}
                      
    inst['iv_fa'] = {'data':inst['unit_fa_x'] * inst['iv_x'] + inst['unit_fa_y'] * inst['iv_y'] + inst['unit_fa_z'] * inst['iv_z'],
                      'units':'m/s',
                      'long_name':'Field-Aligned ion velocity',
                      'notes':('Ion velocity relative to co-rotation along magnetic field line. Positive along the field. ',
                               'Velocity obtained using ion velocities relative '
                               'to co-rotation in the instrument frame along '
                               'with the corresponding unit vectors expressed in '
                               'the instrument frame. '),
                      'label':'Field-Aligned Ion Velocity',
                      'axis':'Field-Aligned Ion Velocity',
                      'desc':'Field-Aligned Ion Velocity',
                      'scale':'Linear',
                      'value_min':-500., 
                      'value_max':500.}

    inst['iv_mer'] = {'data':inst['unit_mer_x'] * inst['iv_x'] + inst['unit_mer_y']*inst['iv_y'] + inst['unit_mer_z']*inst['iv_z'],
                      'units':'m/s',
                      'long_name':'Meridional ion velocity',
                      'notes':('Velocity along meridional direction, perpendicular '
                               'to field and within meridional plane. Positive is up at magnetic equator. ',
                               'Velocity obtained using ion velocities relative '
                               'to co-rotation in the instrument frame along '
                               'with the corresponding unit vectors expressed in '
                               'the instrument frame. '),
                      'label':'Meridional Ion Velocity',
                      'axis':'Meridional Ion Velocity',
                      'desc':'Meridional Ion Velocity',
                      'scale':'Linear',
                      'value_min':-500., 
                      'value_max':500.}
    
    return


def add_footpoint_and_equatorial_drifts(inst, equ_mer_scalar='equ_mer_drifts_scalar',
                                              equ_zonal_scalar='equ_zon_drifts_scalar',
                                              north_mer_scalar='north_footpoint_mer_drifts_scalar',
                                              north_zon_scalar='north_footpoint_zon_drifts_scalar',
                                              south_mer_scalar='south_footpoint_mer_drifts_scalar',
                                              south_zon_scalar='south_footpoint_zon_drifts_scalar',
                                              mer_drift='iv_mer',
                                              zon_drift='iv_zon'):
    """Translates geomagnetic ion velocities to those at footpoints and magnetic equator.
    Note
    ----
        Presumes scalar values for mapping ion velocities are already in the inst, labeled
        by north_footpoint_zon_drifts_scalar, north_footpoint_mer_drifts_scalar,
        equ_mer_drifts_scalar, equ_zon_drifts_scalar.
    
        Also presumes that ion motions in the geomagnetic system are present and labeled
        as 'iv_mer' and 'iv_zon' for meridional and zonal ion motions.
        
        This naming scheme is used by the other pysat oriented routines
        in this package.
    
    Parameters
    ----------
    inst : pysat.Instrument
    equ_mer_scalar : string
        Label used to identify equatorial scalar for meridional ion drift
    equ_zon_scalar : string
        Label used to identify equatorial scalar for zonal ion drift
    north_mer_scalar : string
        Label used to identify northern footpoint scalar for meridional ion drift
    north_zon_scalar : string
        Label used to identify northern footpoint scalar for zonal ion drift
    south_mer_scalar : string
        Label used to identify northern footpoint scalar for meridional ion drift
    south_zon_scalar : string
        Label used to identify southern footpoint scalar for zonal ion drift
    mer_drift : string
        Label used to identify meridional ion drifts within inst
    zon_drift : string
        Label used to identify zonal ion drifts within inst
        
    Returns
    -------
    None
        Modifies pysat.Instrument object in place. Drifts mapped to the magnetic equator
        are labeled 'equ_mer_drift' and 'equ_zon_drift'. Mappings to the northern
        and southern footpoints are labeled 'south_footpoint_mer_drift' and
        'south_footpoint_zon_drift'. Similarly for the northern hemisphere.
    """

    inst['equ_mer_drift'] = {'data' : inst[equ_mer_scalar]*inst[mer_drift],
                            'units':'m/s',
                            'long_name':'Equatorial meridional ion velocity',
                            'notes':('Velocity along meridional direction, perpendicular '
                                    'to field and within meridional plane, scaled to '
                                    'magnetic equator. Positive is up at magnetic equator. '
                                    'Velocity obtained using ion velocities relative '
                                    'to co-rotation in the instrument frame along '
                                    'with the corresponding unit vectors expressed in '
                                    'the instrument frame. Field-line mapping and '
                                    'the assumption of equi-potential field lines '
                                    'is used to translate the locally measured ion '
                                    'motion to the magnetic equator. The mapping '
                                    'is used to determine the change in magnetic '
                                    'field line distance, which, under assumption of '
                                    'equipotential field lines, in turn alters '
                                    'the electric field at that location (E=V/d). '),
                            'label':'Equatorial Meridional Ion Velocity',
                            'axis':'Equatorial Meridional Ion Velocity',
                            'desc':'Equatorial Meridional Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['equ_zon_drift'] = {'data' : inst[equ_zonal_scalar]*inst[zon_drift],
                            'units':'m/s',
                            'long_name':'Equatorial zonal ion velocity',
                            'notes':('Velocity along zonal direction, perpendicular '
                                    'to field and the meridional plane, scaled to '
                                    'magnetic equator. Positive is generally eastward. '
                                    'Velocity obtained using ion velocities relative '
                                    'to co-rotation in the instrument frame along '
                                    'with the corresponding unit vectors expressed in '
                                    'the instrument frame. Field-line mapping and '
                                    'the assumption of equi-potential field lines '
                                    'is used to translate the locally measured ion '
                                    'motion to the magnetic equator. The mapping '
                                    'is used to determine the change in magnetic '
                                    'field line distance, which, under assumption of '
                                    'equipotential field lines, in turn alters '
                                    'the electric field at that location (E=V/d). '),
                            'label':'Equatorial Zonal Ion Velocity',
                            'axis':'Equatorial Zonal Ion Velocity',
                            'desc':'Equatorial Zonal Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['south_footpoint_mer_drift'] = {'data' : inst[south_mer_scalar]*inst[mer_drift],
                            'units':'m/s',
                            'long_name':'Southern meridional ion velocity',
                            'notes':('Velocity along meridional direction, perpendicular '
                                    'to field and within meridional plane, scaled to '
                                    'southern footpoint. Positive is up at magnetic equator. '
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
                            'label':'Southern Meridional Ion Velocity',
                            'axis':'Southern Meridional Ion Velocity',
                            'desc':'Southern Meridional Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['south_footpoint_zon_drift'] = {'data':inst[south_zon_scalar]*inst[zon_drift],
                            'units':'m/s',
                            'long_name':'Southern zonal ion velocity',
                            'notes':('Velocity along zonal direction, perpendicular '
                                    'to field and the meridional plane, scaled to '
                                    'southern footpoint. Positive is generally eastward. '
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
                            'label':'Southern Zonal Ion Velocity',
                            'axis':'Southern Zonal Ion Velocity',
                            'desc':'Southern Zonal Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['north_footpoint_mer_drift'] = {'data':inst[north_mer_scalar]*inst[mer_drift],
                            'units':'m/s',
                            'long_name':'Northern meridional ion velocity',
                            'notes':('Velocity along meridional direction, perpendicular '
                                    'to field and within meridional plane, scaled to '
                                    'northern footpoint. Positive is up at magnetic equator. '
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
                            'label':'Northern Meridional Ion Velocity',
                            'axis':'Northern Meridional Ion Velocity',
                            'desc':'Northern Meridional Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['north_footpoint_zon_drift'] = {'data':inst[north_zon_scalar]*inst[zon_drift],
                            'units':'m/s',
                            'long_name':'Northern zonal ion velocity',
                            'notes':('Velocity along zonal direction, perpendicular '
                                    'to field and the meridional plane, scaled to '
                                    'northern footpoint. Positive is generally eastward. '
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
                            'label':'Northern Zonal Ion Velocity',
                            'axis':'Northern Zonal Ion Velocity',
                            'desc':'Northern Zonal Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}
