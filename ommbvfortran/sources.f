

! Transformation functions
      subroutine ecef_to_geodetic(pos, latitude, lon, h)
      ! converts from ECEF coordinates to Geodetic
      ! pos is the 3 element array of positions in km
      ! latitude returned in Radians
      ! longitude returned in Radians
      ! h is the geodetic altitude, returned in km

      ! pos is (x,y,z) in ECEF coordinates
      real*8, dimension(3) :: pos
      real*8 a,b,ellip,e2,p,e_prime,theta,latitude,r_n,h
      real*8 lon
      real*8 pi

cf2py intent(in) pos
Cf2py intent(out) latitude, lon, h

      pi = 4.D0*DATAN(1.D0)
      ! interpret height as geodetic limit
      ! take current position in lat,lon, and final height
      ! convert to ECEF and use as a limit
      ! equivalent geocentric height
      ! closed form solution
      b = 6356.75231424518d0
      a = 6378.1370d0
      ellip = dsqrt(1.d0-b**2/a**2)
      ! first eccentricity squared
      e2 = ellip**2 !6.6943799901377997E-3
      ! radial position from z, think rho and cylindrical coords
      p = dsqrt(pos(1)**2+pos(2)**2)
      e_prime = dsqrt((a**2-b**2)/b**2)
      theta = datan(pos(3)*a/(p*b))
      latitude = datan( (pos(3)+e_prime**2*b*dsin(theta)**3)/
     &(p-ellip**2*a*dcos(theta)**3))

      r_n = a/dsqrt(1.d0-ellip**2*dsin(latitude)**2)
      ! geodetic height
      h = p/dcos(latitude) - r_n

      lon = datan(pos(2)/pos(1))
      ! Based on values from atan2() from Python
      if (pos(1).lt.0) then
         if (pos(2).lt.0) lon=lon-pi
         if (pos(2).ge.0) lon=lon+pi
      end if
      if (pos(1).eq.0) then
         if (pos(2).gt.0) lon=lon+pi/2.D0
         if (pos(2).lt.0) lon=lon-pi/2.D0
      end if

      return
      end


      subroutine end_vector_to_ecef(be,bn,bd,colat,elong,bx,by,bz)
      ! inputs are east, north, and down components
      ! colatitude (radians, 0 at northern pole)
      ! longitude (radians)
      ! mag field values
      real*8 colat,elong,bn,be,bd,lat
      ! position values of point
      real*8 bx,by,bz
      real*8 pi

cf2py intent(in) be, bn, bd, colat, elong
Cf2py intent(out)  bx, by, bz

      pi = 4.D0*DATAN(1.D0)
      lat = pi/2.D0 - colat
      ! convert the mag field in spherical unit vectors to ECEF
      !bx=-bd*dsin(colat)*dcos(elong)-bn*dcos(colat)*dcos(elong)
      bx=-bd*dcos(lat)*dcos(elong)-bn*dsin(lat)*dcos(elong)
      bx=bx-be*dsin(elong)
      by=-bd*dcos(lat)*dsin(elong)-bn*dsin(lat)*dsin(elong)
      by=by+be*dcos(elong)
      bz=-bd*dsin(lat)+bn*dcos(lat)

      return
      end


      subroutine ecef_to_colat_long_r(pos, colat, elong, r)
      ! pos is (x,y,z) in ECEF coordinates
      real*8, dimension(3) :: pos
      ! position values of point
      real*8 r, colat, elong
      ! height to stop integration at
      real*8 pi
      pi = 4.D0*DATAN(1.D0)

cf2py intent(in) pos
Cf2py intent(out) colat, elong, r

      ! calculate longitude, colatitude, and altitude from ECEF position
      ! altitude should be radial distance from center of earth
      r = dsqrt(pos(1)**2+pos(2)**2+pos(3)**2)
      colat = dacos(pos(3)/r)
      elong = datan(pos(2)/pos(1))

      ! Based on values from atan2() from Python
      ! Based on values from atan2() from Python
      if (pos(1).lt.0) then
         if (pos(2).lt.0) elong=elong-pi
         if (pos(2).ge.0) elong=elong+pi
      end if
      if (pos(1).eq.0) then
         if (pos(2).gt.0) elong=elong+pi/2.D0
         if (pos(2).lt.0) elong=elong-pi/2.D0
      end if

      return
      end

      subroutine ecef_to_end_vector(x,y,z,colat,elong,vn,ve,vd)
      ! Vector expressed along ECEF returned along END directions
      !
      ! inputs are x, y, and z ECEF components
      ! colatitude (radians, 0 at northern pole)
      ! longitude (radians)
      ! outputs are vn, ve, vd components

      real*8 x,y,z,colat,elong,ve,vn,vd,pi,lat
      pi = 4.D0*DATAN(1.D0)
      lat = pi/2.D0 - colat

      ve = -x*dsin(elong) + y*dcos(elong)
      vn = -x*dcos(elong)*dsin(lat)-y*dsin(elong)*dsin(lat)+z*dcos(lat)
      vd = -x*dcos(elong)*dcos(lat)-y*dsin(elong)*dcos(lat)-z*dsin(lat)

      return
      end

      subroutine colat_long_r_to_ecef(pos, colat, elong, r)
      ! Convert position in geocentric to ECEF
      ! pos is (x,y,z) in ECEF coordinates
      real*8, dimension(3) :: pos
      ! position values of point
      real*8 r, colat, elong, x, y, z

cf2py intent(out) pos
Cf2py intent(in) colat, elong, r

      x = r * dsin(colat) * dcos(elong)
      y = r * dsin(colat) * dsin(elong)
      z = r * dcos(colat)
      pos = (/x, y, z/)

      return
      end

      subroutine dipole_field(pos, offs, m, b)
      ! returns magnetic field from dipole
      ! pos, location where field is returned, is (x, y, z) ECEF in km
      ! offs, offset location of dipole from origin, (x, y, z) ECEF in km
      ! m is (mx, my, mz)
      ! b, calculated magnetic field, (bx, by, bz)

      real*8, dimension(3) :: pos, offs, m, b, r, rhat
      real*8 mu_r3, r3, rmag

      ! eqn is mu/4(pi)*( 3n(n dot m) - m)/r**3

      ! construct vector to dipole
      ! put in terms in meters
      do i=1,3
        r(i) = (pos(i) - offs(i)) * 1000.D0
      end do

      ! position unit vector (n)
      rmag = dsqrt(r(1)**2 + r(2)**2 + r(3)**2)
      do i=1,3
        rhat(i) = r(i) / rmag
      end do

      ! n dot m term
      rm = rhat(1) * m(1) + rhat(2) * m(2) + rhat(3) * m(3)
      ! r**3 term
      r3 = rmag**3
      ! mu / 4 pi / |r**3|
      mu_r3 = 1.D-7 / r3

      ! calculate magnetic field
      do i=1,3
        b(i) = mu_r3 * (3.D0 * rm * rhat(i) - m(i))
      end do

      return
      end


      subroutine linear_quadrupole(pos, offs, m, step, b)
      ! returns magnetic field from linear quadrupole
      ! pos, location where field is returned, (x, y, z) ECEF in km
      ! offs, offset location of quadrupole from origin, (x, y, z) ECEF in km
      ! sep, distance from quadrupole origin for underlying dipoles (km)
      ! m is magnetic moment of primary underlying dipoles (vector)
      ! b, calculated magnetic field, (bx, by, bz)
      real*8, dimension(3) :: pos, offs, b, temp_b, temp_m, temp_o, m
      real*8, dimension(3) :: um
      real*8 :: step, mag, scalar

      ! Calculate step vector. Get unit mag vector then step along.
      mag = dsqrt(m(1) * m(1) + m(2) * m(2) + m(3) * m(3))
      scalar = step / mag
      do i=1,3
        um(i) = m(i) * scalar
      enddo

      ! Take step, from offs, along magnetic moment direction
      do i=1,3
        temp_o(i) = offs(i) + um(i)
      enddo

      call dipole_field(pos, temp_o, m, temp_b)
      b(1) = temp_b(1)
      b(2) = temp_b(2)
      b(3) = temp_b(3)

      ! Underlying dipoles arranged along moment direction
      ! Take opposite step, from offs, as first dipole.
      do i=1,3
        temp_o(i) = offs(i) - um(i)
      enddo
      ! Reversed moment.
      temp_m = (/ -1.D0 * m(1), -1.D0 * m(2), -1.D0 * m(3) /)

      call dipole_field(pos, temp_o, temp_m, temp_b)
      b(1) = b(1) + temp_b(1)
      b(2) = b(2) + temp_b(2)
      b(3) = b(3) + temp_b(3)

      return
      end

      subroutine poles (flag,isv,date,itype,alt,colat,elong,x,y,z,f)
      ! Generate magnetic field for testing purposes
      ! Function signature needs to mostly match that of igrfsyn
      ! flag indicates which magnetic field source to produce.
      ! 0 : Dipole only field
      ! 1 : Dipole + Linear Quadrupole along ECEF +Z
      ! 2 : Dipole + Linear Quadrupole along ECEF +X
      ! 3 : Dipole + Normal Quadrupole, equatorial plane
      ! 4 : Dipole + Normal Octupole, equatorial plane
      ! isv, date not used
      real*8, dimension(3) :: bdip, bq, m, offs, pos, tbq
      real*8 x, y, z, f ! field along east, north, down, magnitude
      real*8 vx, vy, vz ! field along ECEF x, y, z
      real*8 alt, colat, elong, date, mag
      integer isv, itype, flag

C following added by RStoneback
Cf2py intent(in) flag,isv,date,itype,alt,colat,elong
Cf2py intent(out) x,y,z,f

      ! Convert inputs to ECEF
      if (itype.eq.2) then
        call colat_long_r_to_ecef(pos, colat, elong, alt)
      end if

      ! Calculate dipole magnetic field
      offs = (/0.0D0, 0.0D0, 0.0D0/)
      m = (/0.D0, 0.D0, -8.D22/)
      call dipole_field(pos, offs, m, bdip)

      ! Initialize memory for quadrupole
      bq = (/ 0.D0, 0.D0, 0.D0 /)

      if (flag.eq.1) then
          ! assymetrical northern hemisphere via linear quadrupole
          m = (/0.D0, 0.D0, -2.37D22/)
          call linear_quadrupole(pos, offs, m, 1000.D0, tbq)
          do i=1,3
           bq(i) = bq(i) + tbq(i)
          enddo
      elseif (flag.eq.2) then
          ! linear quad, equatorial plane
          m = (/2.37D21, 0.D0, 0.D0/)
          call linear_quadrupole(pos, offs, m, 1000.D0, tbq)
          do i=1,3
           bq(i) = bq(i) + tbq(i)
          enddo
      elseif (flag.eq.3) then
          ! normal quad, equatorial plane
          m = (/2.37D21, 0.D0, 0.D0/)
          call linear_quadrupole(pos, offs, m, 1000.D0, tbq)
          do i=1,3
           bq(i) = bq(i) + tbq(i)
          enddo

          m = (/0.D0, 2.37D21, 0.D0/)
          call linear_quadrupole(pos, offs, m, -1000.D0, tbq)
          do i=1,3
           bq(i) = bq(i) + tbq(i)
          enddo
      elseif (flag.eq.4) then
          ! normal octupole, equatorial plane
          mag = 1.D20
          m = (/mag, mag, 0.D0/)
          offs = (/0.0D0, 0.0D0, 1000.0D0/)
          call linear_quadrupole(pos, offs, m, -1000.D0, tbq)
          do i=1,3
           bq(i) = bq(i) + tbq(i)
          enddo

          m = (/-mag, mag, 0.D0/)
          call linear_quadrupole(pos, offs, m, 1000.D0, tbq)
          do i=1,3
           bq(i) = bq(i) + tbq(i)
          enddo

          m = (/mag, mag, 0.D0/)
          offs = (/0.0D0, 0.0D0, -1000.0D0/)
          call linear_quadrupole(pos, offs, m, -1000.D0, tbq)
          do i=1,3
           bq(i) = bq(i) + tbq(i)
          enddo

          m = (/-mag, mag, 0.D0/)
          call linear_quadrupole(pos, offs, m, 1000.D0, tbq)
          do i=1,3
           bq(i) = bq(i) + tbq(i)
          enddo
      end if

      ! Combine dipole and quadrupole contributions
      vx = bdip(1) + bq(1)
      vy = bdip(2) + bq(2)
      vz = bdip(3) + bq(3)
      f = dsqrt(vx**2 + vy**2 + vz**2)

      ! Rotate field into north, east, vertical (down) components
      call ecef_to_end_vector(vx,vy,vz,colat,elong,x,y,z)

      return
      end


      subroutine gen_step(mflag,gflag,out,pos,t,date,step,dir,height)
      ! Generalized integrand field-line step function for testing

      ! IGRF call
      external :: igrf14syn
      ! pos is (x,y,z) in ECEF coordinates
      real*8, dimension(3) :: pos
      ! t is total time supplied by integration routine, ignored
      ! date is year.fractional_doy
      real*8 t, date
      ! step size along field (km)
      real*8 step
      ! Flag for integration direction
      ! +1 Field line direction, -1 anti-field-aligned
      real*8 dir
      ! Height to stop integration at
      real*8 height
      ! Flag for type of magnetic field. -1 for IGRF.
      ! See `poles` for fields returned for 1 and higher.
      integer mflag
      ! Flag for geodetic(0) or geocentric(1)
      integer gflag
      ! output position info
      real*8, dimension(3) :: out

      ! internal function values
      ! mag field values
      real*8 bx,by,bz,bn,be,bd,bm
      ! position values
      real*8 r, colat, elong, h, latitude

cf2py intent(in) mflag,gflag,pos,t,date,step,dir,height
Cf2py intent(out) out

      ! Calculate longitude, colatitude, and altitude from ECEF position
      ! Altitude should be radial distance from center of earth
      call ecef_to_colat_long_r(pos,colat,elong,r)

      ! Calculate magnetic field value
      if (mflag.eq.-1) then
        ! Use IGRF. Supported to ensure reults same as operational
        ! IGRF function.
        call igrf14syn(0,date,2,r,colat,elong,bn,be,bd,bm)
      else
        ! Use a physically specified dipole + higher order poles
        call poles(mflag,0,date,2,r,colat,elong,bn,be,bd,bm)
      end if

      ! Convert the mag field in spherical unit vectors to ECEF vectors
      call end_vector_to_ecef(be,bn,bd,colat,elong,bx,by,bz)

      ! Get updated position, we need to know when to terminate
      if (gflag.eq.0) then
        ! Geodetic Earth
        call ecef_to_geodetic(pos, latitude, elong, h)
      else
        ! Geocentric Earth
        h = r - 6371.D0
      end if

      ! Stop moving position if we go below height
      if (h.le.(height)) then
        step = 0
      else if (h.le.(height+step)) then
        step = step*(1. - ((height+step - h)/step)**2)
      end if

      ! Take step
      bm = 1.D0 / bm
      out(1) = dir * step * bx * bm
      out(2) = dir * step * by * bm
      out(3) = dir * step * bz * bm

      return
      end


      subroutine igrf_step(out,pos,t,date,scalar,dir,height)

      ! IGRF call
      external :: igrf14syn

      real*8, dimension(3) :: pos
      real*8 t, date

      ! pos is (x,y,z) in ECEF coordinates
      ! t is time supplied by integration routine

      ! mag field values
      real*8 bx,by,bz,bn,be,bd,bm
      ! position values of point
      real*8 r, colat, elong
      ! output position info
      real*8, dimension(3) :: out
      ! scalar for how big of a step size (km)
      real*8 scalar
      ! scalar for integration direction (+/- 1)
      real*8 dir
      ! height to stop integration at
      real*8 height
      real*8 pi
      ! parameters for geodetic to ecef calc
      real*8 a,b,ellip,e2,p,e_prime,theta,latitude,r_n,h
      pi = 4.D0*DATAN(1.D0)

cf2py intent(in) pos,t,date,scalar,dir,height
Cf2py intent(out) out

      ! calculate longitude, colatitude, and altitude from ECEF position
      ! altitude should be radial distance from center of earth
      call ecef_to_colat_long_r(pos,colat,elong,r)

      call igrf14syn(0,date,2,r,colat,elong,bn,be,bd,bm)

      ! convert the mag field in spherical unit vectors to ECEF vectors
      call end_vector_to_ecef(be,bn,bd,colat,elong,bx,by,bz)

      ! get updated geodetic position, we need to know
      ! when to terminate
      call ecef_to_geodetic(pos, latitude, elong, h)

      ! stop moving position if we go below height
      if (h.le.(height)) then
        scalar = 0
      else if (h.le.(height+scalar)) then
        scalar = scalar*(1. - ((height+scalar - h)/scalar)**2)
      end if

      out(1) = dir*scalar*bx/bm
      out(2) = dir*scalar*by/bm
      out(3) = dir*scalar*bz/bm

      return
      end
