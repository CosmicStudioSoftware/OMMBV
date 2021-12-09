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

      subroutine quadrupole_field(pos, offs, sep, m, b)
      ! returns magnetic field from quadrupole
      ! pos, location where field is returned, is (x, y, z) ECEF in km
      ! offs, offset location of quadrupole from origin, (x, y, z) ECEF in km
      ! sep, distance from quadrupole origin of underlying dipoles (km)
      ! m is magnetic moment of underlying dipoles (scalar)
      ! b, calculated magnetic field, (bx, by, bz)
      real*8, dimension(3) :: pos, offs, b, temp_b, temp_m, temp_o
      real*8 sep, m

      ! Underlying dipoles arranged along y-z plane. + y +z
      temp_m = (/ 0.D0, m * (-1.D0), m * (-1.D0) /)
      temp_o = (/ offs(1), offs(2) + sep, offs(3) + sep/)
      call dipole_field(pos, temp_o, temp_m, temp_b)
      b(1) = temp_b(1)
      b(2) = temp_b(2)
      b(3) = temp_b(3)

      ! Underlying dipoles arranged along y-z plane. + y -z
      ! Reversed moment.
      temp_m = (/ 0.D0, m * 1.D0, m * (-1.D0) /)
      temp_o = (/ offs(1), offs(2) + sep, offs(3) - sep /)
      call dipole_field(pos, temp_o, temp_m, temp_b)
      b(1) = b(1) + temp_b(1)
      b(2) = b(2) + temp_b(2)
      b(3) = b(3) + temp_b(3)

      ! Underlying dipoles arranged along y-z plane. - y +z
      ! Reversed moment.
      temp_m = (/ 0.D0, m * (-1.D0), m * 1.D0 /)
      temp_o = (/ offs(1), offs(2) - sep, offs(3) + sep /)
      call dipole_field(pos, temp_o, temp_m, temp_b)
      b(1) = b(1) + temp_b(1)
      b(2) = b(2) + temp_b(2)
      b(3) = b(3) + temp_b(3)

      ! Underlying dipoles arranged along y-z plane. - y -z
      temp_m = (/ 0.D0, m * 1.D0, m * 1.D0 /)
      temp_o = (/ offs(1), offs(2) - sep, offs(3) - sep /)
      call dipole_field(pos, temp_o, temp_m, temp_b)
      b(1) = b(1) + temp_b(1)
      b(2) = b(2) + temp_b(2)
      b(3) = b(3) + temp_b(3)

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


      subroutine igrf_step(out,pos,t,date,scalar,dir,height)

      real*8, dimension(3) :: pos
      real*8 t, date

      ! mag field values
      real*8 bx,by,bz,bn,be,bd,bm
      ! position values of point
      real*8 r, colat, elong
      ! output position info
      real*8, dimension(3) :: out
      ! scalar for how big of a step size, fractions of Re
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

      ! pos is (x,y,z) in ECEF coordinates
      ! t is time supplied by integration routine

      ! calculate longitude, colatitude, and altitude from ECEF position
      ! altitude should be radial distance from center of earth
      call ecef_to_colat_long_r(pos,colat,elong,r)

      call igrf13syn(0,date,2,r,colat,elong,bn,be,bd,bm)

      ! convert the mag field in spherical unit vectors to ECEF vectors
      call end_vector_to_ecef(be,bn,bd,colat,elong,bx,by,bz)

      ! get updated geodetic position, we need to know
      ! when to terminate
      call ecef_to_geodetic(pos, latitude, elong, h)

      ! Use geographic Earth instead of geodetic
!     h = r - 6371.D0

      ! stop moving position if we go below height
      if (h.le.(height)) then
        scalar = 0
      else if (h.le.(height+10.)) then
        !scalar=scalar*exp(r-(6371.+height))
        scalar = scalar*(1. - ((height+10. - h)/10.)**2)
      end if

      !if (scalar.lt.0) then
      !  scalar = 0.
      !end if
      out(1) = dir*scalar*bx/bm
      out(2) = dir*scalar*by/bm
      out(3) = dir*scalar*bz/bm

      !write(*,*) sqrt(bn**2+be**2+bd**2)/bm
      !write(*,*) sqrt(bx**2+by**2+bz**2)/bm

      return
      end


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
Cf2py intent(out)  latitude, lon, h

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


      subroutine ecef_to_end_vector(x,y,z,colat,elong,vn,ve,vd)
      ! inputs are x, y, and z ECEF components
      ! colatitude (radians, 0 at northern pole)
      ! longitude (radians)

      real*8 x,y,z,colat,elong,ve,vn,vd,pi,lat
      pi = 4.D0*DATAN(1.D0)
      lat = pi/2.D0 - colat

      ve = -x*dsin(elong) + y*dcos(elong)
      vn = -x*dcos(elong)*dsin(lat)-y*dsin(elong)*dsin(lat)+z*dcos(lat)
      vd = -x*dcos(elong)*dcos(lat)-y*dsin(elong)*dcos(lat)-z*dsin(lat)

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


      subroutine colat_long_r_to_ecef(pos, colat, elong, r)
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


      subroutine igrf13syntest (isv,date,itype,alt,colat,elong,x,y,z,f)
      ! Generate magnetic field for testing purposes
      ! isv, date, itype not supported
      real*8, dimension(3) :: bdip, bq, m, offs, pos, tbq
      real*8 x, y, z, f ! field along east, north, down, magnitude
      real*8 vx, vy, vz ! field along ECEF x, y, z
      real*8 alt, colat, elong, date, mag
      integer isv, itype

C following added by RStoneback
Cf2py intent(in) isv,date,itype,alt,colat,elong
Cf2py intent(out) x,y,z,f

      ! Convert inputs to ECEF
      if (itype.eq.2) then
        call colat_long_r_to_ecef(pos, colat, elong, alt)
      end if


      ! Calculate magnetic field
      ! This needs to be moved to the IGRF generation area
      offs = (/0.0D0, 0.0D0, 0.0D0/)
      m = (/0.D0, 0.D0, -8.D22/)
      bq = (/ 0.D0, 0.D0, 0.D0 /)
      call dipole_field(pos, offs, m, bdip)


!     ! assymetrical northern hemisphere
!     m = (/0.D0, 0.D0, -2.37D22/)
!     call linear_quadrupole(pos, offs, m, 1000.D0, tbq)
!     do i=1,3
!      bq(i) = bq(i) + tbq(i)
!     enddo


!     ! linear quad, equatorial plane
!!     m = (/2.37D21, 0.D0, 0.D0/)
!     call linear_quadrupole(pos, offs, m, 1000.D0, tbq)
!     do i=1,3
!      bq(i) = bq(i) + tbq(i)
!     enddo


!     ! normal quad, equatorial plane
!     m = (/2.37D21, 0.D0, 0.D0/)
!     call linear_quadrupole(pos, offs, m, 1000.D0, tbq)
!     do i=1,3
!      bq(i) = bq(i) + tbq(i)
!     enddo
!!     m = (/0.D0, 2.37D21, 0.D0/)
!     call linear_quadrupole(pos, offs, m, -1000.D0, tbq)
!     do i=1,3
!      bq(i) = bq(i) + tbq(i)
!     enddo


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

      vx = bdip(1) + bq(1)
      vy = bdip(2) + bq(2)
      vz = bdip(3) + bq(3)
      f = dsqrt(vx**2 + vy**2 + vz**2)

!      write(*,*) 'Vector in ECEF x, y, z and magnitude'
!      write(*,*) vx, vy, vz, f
!      write(*,*) 'Colatitude, longitude'
!      write(*,*) colat, elong

      ! Rotate field into north, east, vertical (down) components
      call ecef_to_end_vector(vx,vy,vz,colat,elong,x,y,z)
!      write(*,*) 'Vector along north, east, down'
!      write(*,*) x, y, z
      return
      end

      subroutine igrf13syn (isv,date,itype,alt,colat,elong,x,y,z,f)
c
c     This is a synthesis routine for the 13th generation IGRF as agreed
c     in December 2019 by IAGA Working Group V-MOD. It is valid 1900.0 to
c     2025.0 inclusive. Values for dates from 1945.0 to 2015.0 inclusive are
c     definitive, otherwise they are non-definitive.
c   INPUT
c     isv   = 0 if main-field values are required
c     isv   = 1 if secular variation values are required
c     date  = year A.D. Must be greater than or equal to 1900.0 and
c             less than or equal to 2030.0. Warning message is given
c             for dates greater than 2025.0. Must be double precision.
c     itype = 1 if geodetic (spheroid)
c     itype = 2 if geocentric (sphere)
c     alt   = height in km above sea level if itype = 1
c           = distance from centre of Earth in km if itype = 2 (>3485 km)
c     colat = colatitude (0-180) /changed by RS(radians)
c     elong = east-longitude (0-360) /changed by RS(radians)
c     alt, colat and elong must be double precision.
c   OUTPUT
c     x     = north component (nT) if isv = 0, nT/year if isv = 1
c     y     = east component (nT) if isv = 0, nT/year if isv = 1
c     z     = vertical component (nT) if isv = 0, nT/year if isv = 1
c     f     = total intensity (nT) if isv = 0, rubbish if isv = 1
c
c     To get the other geomagnetic elements (D, I, H and secular
c     variations dD, dH, dI and dF) use routines ptoc and ptocsv.
c
c     Adapted from 8th generation version to include new maximum degree for
c     main-field models for 2000.0 and onwards and use WGS84 spheroid instead
c     of International Astronomical Union 1966 spheroid as recommended by IAGA
c     in July 2003. Reference radius remains as 6371.2 km - it is NOT the mean
c     radius (= 6371.0 km) but 6371.2 km is what is used in determining the
c     coefficients. Adaptation by Susan Macmillan, August 2003 (for
c     9th generation), December 2004, December 2009 & December 2014;
c     by William Brown, December 2019, February 2020.
c
c     Coefficients at 1995.0 incorrectly rounded (rounded up instead of
c     to even) included as these are the coefficients published in Excel
c     spreadsheet July 2005.
c
      implicit double precision (a-h,o-z)
      dimension gh(3645),g0(120),g1(120),g2(120),g3(120),g4(120),
     1          g5(120),g6(120),g7(120),g8(120),g9(120),ga(120),
     2          gb(120),gc(120),gd(120),ge(120),gf(120),gg(120),
     3          gi(120),gj(120),gk(195),gl(195),gm(195),gp(195),
     4          gq(195),gr(195),gs(195),
     5          p(105),q(105),cl(13),sl(13)
      equivalence (g0,gh(1)),(g1,gh(121)),(g2,gh(241)),(g3,gh(361)),
     1            (g4,gh(481)),(g5,gh(601)),(g6,gh(721)),(g7,gh(841)),
     2            (g8,gh(961)),(g9,gh(1081)),(ga,gh(1201)),
     3            (gb,gh(1321)),(gc,gh(1441)),(gd,gh(1561)),
     4            (ge,gh(1681)),(gf,gh(1801)),(gg,gh(1921)),
     5            (gi,gh(2041)),(gj,gh(2161)),(gk,gh(2281)),
     6            (gl,gh(2476)),(gm,gh(2671)),(gp,gh(2866)),
     7            (gq,gh(3061)),(gr,gh(3256)),(gs,gh(3451))
c
      data g0/ -31543.,-2298., 5922., -677., 2905.,-1061.,  924., 1121., 1900
     1           1022.,-1469., -330., 1256.,    3.,  572.,  523.,  876., 1900
     2            628.,  195.,  660.,  -69., -361., -210.,  134.,  -75., 1900
     3           -184.,  328., -210.,  264.,   53.,    5.,  -33.,  -86., 1900
     4           -124.,  -16.,    3.,   63.,   61.,   -9.,  -11.,   83., 1900
     5           -217.,    2.,  -58.,  -35.,   59.,   36.,  -90.,  -69., 1900
     6             70.,  -55.,  -45.,    0.,  -13.,   34.,  -10.,  -41., 1900
     7             -1.,  -21.,   28.,   18.,  -12.,    6.,  -22.,   11., 1900
     8              8.,    8.,   -4.,  -14.,   -9.,    7.,    1.,  -13., 1900
     9              2.,    5.,   -9.,   16.,    5.,   -5.,    8.,  -18., 1900
     a              8.,   10.,  -20.,    1.,   14.,  -11.,    5.,   12., 1900
     b             -3.,    1.,   -2.,   -2.,    8.,    2.,   10.,   -1., 1900
     c             -2.,   -1.,    2.,   -3.,   -4.,    2.,    2.,    1., 1900
     d             -5.,    2.,   -2.,    6.,    6.,   -4.,    4.,    0., 1900
     e              0.,   -2.,    2.,    4.,    2.,    0.,    0.,   -6./ 1900
      data g1/ -31464.,-2298., 5909., -728., 2928.,-1086., 1041., 1065., 1905
     1           1037.,-1494., -357., 1239.,   34.,  635.,  480.,  880., 1905
     2            643.,  203.,  653.,  -77., -380., -201.,  146.,  -65., 1905
     3           -192.,  328., -193.,  259.,   56.,   -1.,  -32.,  -93., 1905
     4           -125.,  -26.,   11.,   62.,   60.,   -7.,  -11.,   86., 1905
     5           -221.,    4.,  -57.,  -32.,   57.,   32.,  -92.,  -67., 1905
     6             70.,  -54.,  -46.,    0.,  -14.,   33.,  -11.,  -41., 1905
     7              0.,  -20.,   28.,   18.,  -12.,    6.,  -22.,   11., 1905
     8              8.,    8.,   -4.,  -15.,   -9.,    7.,    1.,  -13., 1905
     9              2.,    5.,   -8.,   16.,    5.,   -5.,    8.,  -18., 1905
     a              8.,   10.,  -20.,    1.,   14.,  -11.,    5.,   12., 1905
     b             -3.,    1.,   -2.,   -2.,    8.,    2.,   10.,    0., 1905
     c             -2.,   -1.,    2.,   -3.,   -4.,    2.,    2.,    1., 1905
     d             -5.,    2.,   -2.,    6.,    6.,   -4.,    4.,    0., 1905
     e              0.,   -2.,    2.,    4.,    2.,    0.,    0.,   -6./ 1905
      data g2/ -31354.,-2297., 5898., -769., 2948.,-1128., 1176., 1000., 1910
     1           1058.,-1524., -389., 1223.,   62.,  705.,  425.,  884., 1910
     2            660.,  211.,  644.,  -90., -400., -189.,  160.,  -55., 1910
     3           -201.,  327., -172.,  253.,   57.,   -9.,  -33., -102., 1910
     4           -126.,  -38.,   21.,   62.,   58.,   -5.,  -11.,   89., 1910
     5           -224.,    5.,  -54.,  -29.,   54.,   28.,  -95.,  -65., 1910
     6             71.,  -54.,  -47.,    1.,  -14.,   32.,  -12.,  -40., 1910
     7              1.,  -19.,   28.,   18.,  -13.,    6.,  -22.,   11., 1910
     8              8.,    8.,   -4.,  -15.,   -9.,    6.,    1.,  -13., 1910
     9              2.,    5.,   -8.,   16.,    5.,   -5.,    8.,  -18., 1910
     a              8.,   10.,  -20.,    1.,   14.,  -11.,    5.,   12., 1910
     b             -3.,    1.,   -2.,   -2.,    8.,    2.,   10.,    0., 1910
     c             -2.,   -1.,    2.,   -3.,   -4.,    2.,    2.,    1., 1910
     d             -5.,    2.,   -2.,    6.,    6.,   -4.,    4.,    0., 1910
     e              0.,   -2.,    2.,    4.,    2.,    0.,    0.,   -6./ 1910
      data g3/ -31212.,-2306., 5875., -802., 2956.,-1191., 1309.,  917., 1915
     1           1084.,-1559., -421., 1212.,   84.,  778.,  360.,  887., 1915
     2            678.,  218.,  631., -109., -416., -173.,  178.,  -51., 1915
     3           -211.,  327., -148.,  245.,   58.,  -16.,  -34., -111., 1915
     4           -126.,  -51.,   32.,   61.,   57.,   -2.,  -10.,   93., 1915
     5           -228.,    8.,  -51.,  -26.,   49.,   23.,  -98.,  -62., 1915
     6             72.,  -54.,  -48.,    2.,  -14.,   31.,  -12.,  -38., 1915
     7              2.,  -18.,   28.,   19.,  -15.,    6.,  -22.,   11., 1915
     8              8.,    8.,   -4.,  -15.,   -9.,    6.,    2.,  -13., 1915
     9              3.,    5.,   -8.,   16.,    6.,   -5.,    8.,  -18., 1915
     a              8.,   10.,  -20.,    1.,   14.,  -11.,    5.,   12., 1915
     b             -3.,    1.,   -2.,   -2.,    8.,    2.,   10.,    0., 1915
     c             -2.,   -1.,    2.,   -3.,   -4.,    2.,    2.,    1., 1915
     d             -5.,    2.,   -2.,    6.,    6.,   -4.,    4.,    0., 1915
     e              0.,   -2.,    1.,    4.,    2.,    0.,    0.,   -6./ 1915
      data g4/ -31060.,-2317., 5845., -839., 2959.,-1259., 1407.,  823., 1920
     1           1111.,-1600., -445., 1205.,  103.,  839.,  293.,  889., 1920
     2            695.,  220.,  616., -134., -424., -153.,  199.,  -57., 1920
     3           -221.,  326., -122.,  236.,   58.,  -23.,  -38., -119., 1920
     4           -125.,  -62.,   43.,   61.,   55.,    0.,  -10.,   96., 1920
     5           -233.,   11.,  -46.,  -22.,   44.,   18., -101.,  -57., 1920
     6             73.,  -54.,  -49.,    2.,  -14.,   29.,  -13.,  -37., 1920
     7              4.,  -16.,   28.,   19.,  -16.,    6.,  -22.,   11., 1920
     8              7.,    8.,   -3.,  -15.,   -9.,    6.,    2.,  -14., 1920
     9              4.,    5.,   -7.,   17.,    6.,   -5.,    8.,  -19., 1920
     a              8.,   10.,  -20.,    1.,   14.,  -11.,    5.,   12., 1920
     b             -3.,    1.,   -2.,   -2.,    9.,    2.,   10.,    0., 1920
     c             -2.,   -1.,    2.,   -3.,   -4.,    2.,    2.,    1., 1920
     d             -5.,    2.,   -2.,    6.,    6.,   -4.,    4.,    0., 1920
     e              0.,   -2.,    1.,    4.,    3.,    0.,    0.,   -6./ 1920
      data g5/ -30926.,-2318., 5817., -893., 2969.,-1334., 1471.,  728., 1925
     1           1140.,-1645., -462., 1202.,  119.,  881.,  229.,  891., 1925
     2            711.,  216.,  601., -163., -426., -130.,  217.,  -70., 1925
     3           -230.,  326.,  -96.,  226.,   58.,  -28.,  -44., -125., 1925
     4           -122.,  -69.,   51.,   61.,   54.,    3.,   -9.,   99., 1925
     5           -238.,   14.,  -40.,  -18.,   39.,   13., -103.,  -52., 1925
     6             73.,  -54.,  -50.,    3.,  -14.,   27.,  -14.,  -35., 1925
     7              5.,  -14.,   29.,   19.,  -17.,    6.,  -21.,   11., 1925
     8              7.,    8.,   -3.,  -15.,   -9.,    6.,    2.,  -14., 1925
     9              4.,    5.,   -7.,   17.,    7.,   -5.,    8.,  -19., 1925
     a              8.,   10.,  -20.,    1.,   14.,  -11.,    5.,   12., 1925
     b             -3.,    1.,   -2.,   -2.,    9.,    2.,   10.,    0., 1925
     c             -2.,   -1.,    2.,   -3.,   -4.,    2.,    2.,    1., 1925
     d             -5.,    2.,   -2.,    6.,    6.,   -4.,    4.,    0., 1925
     e              0.,   -2.,    1.,    4.,    3.,    0.,    0.,   -6./ 1925
      data g6/ -30805.,-2316., 5808., -951., 2980.,-1424., 1517.,  644., 1930
     1           1172.,-1692., -480., 1205.,  133.,  907.,  166.,  896., 1930
     2            727.,  205.,  584., -195., -422., -109.,  234.,  -90., 1930
     3           -237.,  327.,  -72.,  218.,   60.,  -32.,  -53., -131., 1930
     4           -118.,  -74.,   58.,   60.,   53.,    4.,   -9.,  102., 1930
     5           -242.,   19.,  -32.,  -16.,   32.,    8., -104.,  -46., 1930
     6             74.,  -54.,  -51.,    4.,  -15.,   25.,  -14.,  -34., 1930
     7              6.,  -12.,   29.,   18.,  -18.,    6.,  -20.,   11., 1930
     8              7.,    8.,   -3.,  -15.,   -9.,    5.,    2.,  -14., 1930
     9              5.,    5.,   -6.,   18.,    8.,   -5.,    8.,  -19., 1930
     a              8.,   10.,  -20.,    1.,   14.,  -12.,    5.,   12., 1930
     b             -3.,    1.,   -2.,   -2.,    9.,    3.,   10.,    0., 1930
     c             -2.,   -2.,    2.,   -3.,   -4.,    2.,    2.,    1., 1930
     d             -5.,    2.,   -2.,    6.,    6.,   -4.,    4.,    0., 1930
     e              0.,   -2.,    1.,    4.,    3.,    0.,    0.,   -6./ 1930
      data g7/ -30715.,-2306., 5812.,-1018., 2984.,-1520., 1550.,  586., 1935
     1           1206.,-1740., -494., 1215.,  146.,  918.,  101.,  903., 1935
     2            744.,  188.,  565., -226., -415.,  -90.,  249., -114., 1935
     3           -241.,  329.,  -51.,  211.,   64.,  -33.,  -64., -136., 1935
     4           -115.,  -76.,   64.,   59.,   53.,    4.,   -8.,  104., 1935
     5           -246.,   25.,  -25.,  -15.,   25.,    4., -106.,  -40., 1935
     6             74.,  -53.,  -52.,    4.,  -17.,   23.,  -14.,  -33., 1935
     7              7.,  -11.,   29.,   18.,  -19.,    6.,  -19.,   11., 1935
     8              7.,    8.,   -3.,  -15.,   -9.,    5.,    1.,  -15., 1935
     9              6.,    5.,   -6.,   18.,    8.,   -5.,    7.,  -19., 1935
     a              8.,   10.,  -20.,    1.,   15.,  -12.,    5.,   11., 1935
     b             -3.,    1.,   -3.,   -2.,    9.,    3.,   11.,    0., 1935
     c             -2.,   -2.,    2.,   -3.,   -4.,    2.,    2.,    1., 1935
     d             -5.,    2.,   -2.,    6.,    6.,   -4.,    4.,    0., 1935
     e              0.,   -1.,    2.,    4.,    3.,    0.,    0.,   -6./ 1935
      data g8/ -30654.,-2292., 5821.,-1106., 2981.,-1614., 1566.,  528., 1940
     1           1240.,-1790., -499., 1232.,  163.,  916.,   43.,  914., 1940
     2            762.,  169.,  550., -252., -405.,  -72.,  265., -141., 1940
     3           -241.,  334.,  -33.,  208.,   71.,  -33.,  -75., -141., 1940
     4           -113.,  -76.,   69.,   57.,   54.,    4.,   -7.,  105., 1940
     5           -249.,   33.,  -18.,  -15.,   18.,    0., -107.,  -33., 1940
     6             74.,  -53.,  -52.,    4.,  -18.,   20.,  -14.,  -31., 1940
     7              7.,   -9.,   29.,   17.,  -20.,    5.,  -19.,   11., 1940
     8              7.,    8.,   -3.,  -14.,  -10.,    5.,    1.,  -15., 1940
     9              6.,    5.,   -5.,   19.,    9.,   -5.,    7.,  -19., 1940
     a              8.,   10.,  -21.,    1.,   15.,  -12.,    5.,   11., 1940
     b             -3.,    1.,   -3.,   -2.,    9.,    3.,   11.,    1., 1940
     c             -2.,   -2.,    2.,   -3.,   -4.,    2.,    2.,    1., 1940
     d             -5.,    2.,   -2.,    6.,    6.,   -4.,    4.,    0., 1940
     e              0.,   -1.,    2.,    4.,    3.,    0.,    0.,   -6./ 1940
      data g9/ -30594.,-2285., 5810.,-1244., 2990.,-1702., 1578.,  477., 1945
     1           1282.,-1834., -499., 1255.,  186.,  913.,  -11.,  944., 1945
     2            776.,  144.,  544., -276., -421.,  -55.,  304., -178., 1945
     3           -253.,  346.,  -12.,  194.,   95.,  -20.,  -67., -142., 1945
     4           -119.,  -82.,   82.,   59.,   57.,    6.,    6.,  100., 1945
     5           -246.,   16.,  -25.,   -9.,   21.,  -16., -104.,  -39., 1945
     6             70.,  -40.,  -45.,    0.,  -18.,    0.,    2.,  -29., 1945
     7              6.,  -10.,   28.,   15.,  -17.,   29.,  -22.,   13., 1945
     8              7.,   12.,   -8.,  -21.,   -5.,  -12.,    9.,   -7., 1945
     9              7.,    2.,  -10.,   18.,    7.,    3.,    2.,  -11., 1945
     a              5.,  -21.,  -27.,    1.,   17.,  -11.,   29.,    3., 1945
     b             -9.,   16.,    4.,   -3.,    9.,   -4.,    6.,   -3., 1945
     c              1.,   -4.,    8.,   -3.,   11.,    5.,    1.,    1., 1945
     d              2.,  -20.,   -5.,   -1.,   -1.,   -6.,    8.,    6., 1945
     e             -1.,   -4.,   -3.,   -2.,    5.,    0.,   -2.,   -2./ 1945
      data ga/ -30554.,-2250., 5815.,-1341., 2998.,-1810., 1576.,  381., 1950
     1           1297.,-1889., -476., 1274.,  206.,  896.,  -46.,  954., 1950
     2            792.,  136.,  528., -278., -408.,  -37.,  303., -210., 1950
     3           -240.,  349.,    3.,  211.,  103.,  -20.,  -87., -147., 1950
     4           -122.,  -76.,   80.,   54.,   57.,   -1.,    4.,   99., 1950
     5           -247.,   33.,  -16.,  -12.,   12.,  -12., -105.,  -30., 1950
     6             65.,  -55.,  -35.,    2.,  -17.,    1.,    0.,  -40., 1950
     7             10.,   -7.,   36.,    5.,  -18.,   19.,  -16.,   22., 1950
     8             15.,    5.,   -4.,  -22.,   -1.,    0.,   11.,  -21., 1950
     9             15.,   -8.,  -13.,   17.,    5.,   -4.,   -1.,  -17., 1950
     a              3.,   -7.,  -24.,   -1.,   19.,  -25.,   12.,   10., 1950
     b              2.,    5.,    2.,   -5.,    8.,   -2.,    8.,    3., 1950
     c            -11.,    8.,   -7.,   -8.,    4.,   13.,   -1.,   -2., 1950
     d             13.,  -10.,   -4.,    2.,    4.,   -3.,   12.,    6., 1950
     e              3.,   -3.,    2.,    6.,   10.,   11.,    3.,    8./ 1950
      data gb/ -30500.,-2215., 5820.,-1440., 3003.,-1898., 1581.,  291., 1955
     1           1302.,-1944., -462., 1288.,  216.,  882.,  -83.,  958., 1955
     2            796.,  133.,  510., -274., -397.,  -23.,  290., -230., 1955
     3           -229.,  360.,   15.,  230.,  110.,  -23.,  -98., -152., 1955
     4           -121.,  -69.,   78.,   47.,   57.,   -9.,    3.,   96., 1955
     5           -247.,   48.,   -8.,  -16.,    7.,  -12., -107.,  -24., 1955
     6             65.,  -56.,  -50.,    2.,  -24.,   10.,   -4.,  -32., 1955
     7              8.,  -11.,   28.,    9.,  -20.,   18.,  -18.,   11., 1955
     8              9.,   10.,   -6.,  -15.,  -14.,    5.,    6.,  -23., 1955
     9             10.,    3.,   -7.,   23.,    6.,   -4.,    9.,  -13., 1955
     a              4.,    9.,  -11.,   -4.,   12.,   -5.,    7.,    2., 1955
     b              6.,    4.,   -2.,    1.,   10.,    2.,    7.,    2., 1955
     c             -6.,    5.,    5.,   -3.,   -5.,   -4.,   -1.,    0., 1955
     d              2.,   -8.,   -3.,   -2.,    7.,   -4.,    4.,    1., 1955
     e             -2.,   -3.,    6.,    7.,   -2.,   -1.,    0.,   -3./ 1955
      data gc/ -30421.,-2169., 5791.,-1555., 3002.,-1967., 1590.,  206., 1960
     1           1302.,-1992., -414., 1289.,  224.,  878., -130.,  957., 1960
     2            800.,  135.,  504., -278., -394.,    3.,  269., -255., 1960
     3           -222.,  362.,   16.,  242.,  125.,  -26., -117., -156., 1960
     4           -114.,  -63.,   81.,   46.,   58.,  -10.,    1.,   99., 1960
     5           -237.,   60.,   -1.,  -20.,   -2.,  -11., -113.,  -17., 1960
     6             67.,  -56.,  -55.,    5.,  -28.,   15.,   -6.,  -32., 1960
     7              7.,   -7.,   23.,   17.,  -18.,    8.,  -17.,   15., 1960
     8              6.,   11.,   -4.,  -14.,  -11.,    7.,    2.,  -18., 1960
     9             10.,    4.,   -5.,   23.,   10.,    1.,    8.,  -20., 1960
     a              4.,    6.,  -18.,    0.,   12.,   -9.,    2.,    1., 1960
     b              0.,    4.,   -3.,   -1.,    9.,   -2.,    8.,    3., 1960
     c              0.,   -1.,    5.,    1.,   -3.,    4.,    4.,    1., 1960
     d              0.,    0.,   -1.,    2.,    4.,   -5.,    6.,    1., 1960
     e              1.,   -1.,   -1.,    6.,    2.,    0.,    0.,   -7./ 1960
      data gd/ -30334.,-2119., 5776.,-1662., 2997.,-2016., 1594.,  114., 1965
     1           1297.,-2038., -404., 1292.,  240.,  856., -165.,  957., 1965
     2            804.,  148.,  479., -269., -390.,   13.,  252., -269., 1965
     3           -219.,  358.,   19.,  254.,  128.,  -31., -126., -157., 1965
     4            -97.,  -62.,   81.,   45.,   61.,  -11.,    8.,  100., 1965
     5           -228.,   68.,    4.,  -32.,    1.,   -8., -111.,   -7., 1965
     6             75.,  -57.,  -61.,    4.,  -27.,   13.,   -2.,  -26., 1965
     7              6.,   -6.,   26.,   13.,  -23.,    1.,  -12.,   13., 1965
     8              5.,    7.,   -4.,  -12.,  -14.,    9.,    0.,  -16., 1965
     9              8.,    4.,   -1.,   24.,   11.,   -3.,    4.,  -17., 1965
     a              8.,   10.,  -22.,    2.,   15.,  -13.,    7.,   10., 1965
     b             -4.,   -1.,   -5.,   -1.,   10.,    5.,   10.,    1., 1965
     c             -4.,   -2.,    1.,   -2.,   -3.,    2.,    2.,    1., 1965
     d             -5.,    2.,   -2.,    6.,    4.,   -4.,    4.,    0., 1965
     e              0.,   -2.,    2.,    3.,    2.,    0.,    0.,   -6./ 1965
      data ge/ -30220.,-2068., 5737.,-1781., 3000.,-2047., 1611.,   25., 1970
     1           1287.,-2091., -366., 1278.,  251.,  838., -196.,  952., 1970
     2            800.,  167.,  461., -266., -395.,   26.,  234., -279., 1970
     3           -216.,  359.,   26.,  262.,  139.,  -42., -139., -160., 1970
     4            -91.,  -56.,   83.,   43.,   64.,  -12.,   15.,  100., 1970
     5           -212.,   72.,    2.,  -37.,    3.,   -6., -112.,    1., 1970
     6             72.,  -57.,  -70.,    1.,  -27.,   14.,   -4.,  -22., 1970
     7              8.,   -2.,   23.,   13.,  -23.,   -2.,  -11.,   14., 1970
     8              6.,    7.,   -2.,  -15.,  -13.,    6.,   -3.,  -17., 1970
     9              5.,    6.,    0.,   21.,   11.,   -6.,    3.,  -16., 1970
     a              8.,   10.,  -21.,    2.,   16.,  -12.,    6.,   10., 1970
     b             -4.,   -1.,   -5.,    0.,   10.,    3.,   11.,    1., 1970
     c             -2.,   -1.,    1.,   -3.,   -3.,    1.,    2.,    1., 1970
     d             -5.,    3.,   -1.,    4.,    6.,   -4.,    4.,    0., 1970
     e              1.,   -1.,    0.,    3.,    3.,    1.,   -1.,   -4./ 1970
      data gf/ -30100.,-2013., 5675.,-1902., 3010.,-2067., 1632.,  -68., 1975
     1           1276.,-2144., -333., 1260.,  262.,  830., -223.,  946., 1975
     2            791.,  191.,  438., -265., -405.,   39.,  216., -288., 1975
     3           -218.,  356.,   31.,  264.,  148.,  -59., -152., -159., 1975
     4            -83.,  -49.,   88.,   45.,   66.,  -13.,   28.,   99., 1975
     5           -198.,   75.,    1.,  -41.,    6.,   -4., -111.,   11., 1975
     6             71.,  -56.,  -77.,    1.,  -26.,   16.,   -5.,  -14., 1975
     7             10.,    0.,   22.,   12.,  -23.,   -5.,  -12.,   14., 1975
     8              6.,    6.,   -1.,  -16.,  -12.,    4.,   -8.,  -19., 1975
     9              4.,    6.,    0.,   18.,   10.,  -10.,    1.,  -17., 1975
     a              7.,   10.,  -21.,    2.,   16.,  -12.,    7.,   10., 1975
     b             -4.,   -1.,   -5.,   -1.,   10.,    4.,   11.,    1., 1975
     c             -3.,   -2.,    1.,   -3.,   -3.,    1.,    2.,    1., 1975
     d             -5.,    3.,   -2.,    4.,    5.,   -4.,    4.,   -1., 1975
     e              1.,   -1.,    0.,    3.,    3.,    1.,   -1.,   -5./ 1975
      data gg/ -29992.,-1956., 5604.,-1997., 3027.,-2129., 1663., -200., 1980
     1           1281.,-2180., -336., 1251.,  271.,  833., -252.,  938., 1980
     2            782.,  212.,  398., -257., -419.,   53.,  199., -297., 1980
     3           -218.,  357.,   46.,  261.,  150.,  -74., -151., -162., 1980
     4            -78.,  -48.,   92.,   48.,   66.,  -15.,   42.,   93., 1980
     5           -192.,   71.,    4.,  -43.,   14.,   -2., -108.,   17., 1980
     6             72.,  -59.,  -82.,    2.,  -27.,   21.,   -5.,  -12., 1980
     7             16.,    1.,   18.,   11.,  -23.,   -2.,  -10.,   18., 1980
     8              6.,    7.,    0.,  -18.,  -11.,    4.,   -7.,  -22., 1980
     9              4.,    9.,    3.,   16.,    6.,  -13.,   -1.,  -15., 1980
     a              5.,   10.,  -21.,    1.,   16.,  -12.,    9.,    9., 1980
     b             -5.,   -3.,   -6.,   -1.,    9.,    7.,   10.,    2., 1980
     c             -6.,   -5.,    2.,   -4.,   -4.,    1.,    2.,    0., 1980
     d             -5.,    3.,   -2.,    6.,    5.,   -4.,    3.,    0., 1980
     e              1.,   -1.,    2.,    4.,    3.,    0.,    0.,   -6./ 1980
      data gi/ -29873.,-1905., 5500.,-2072., 3044.,-2197., 1687., -306., 1985
     1           1296.,-2208., -310., 1247.,  284.,  829., -297.,  936., 1985
     2            780.,  232.,  361., -249., -424.,   69.,  170., -297., 1985
     3           -214.,  355.,   47.,  253.,  150.,  -93., -154., -164., 1985
     4            -75.,  -46.,   95.,   53.,   65.,  -16.,   51.,   88., 1985
     5           -185.,   69.,    4.,  -48.,   16.,   -1., -102.,   21., 1985
     6             74.,  -62.,  -83.,    3.,  -27.,   24.,   -2.,   -6., 1985
     7             20.,    4.,   17.,   10.,  -23.,    0.,   -7.,   21., 1985
     8              6.,    8.,    0.,  -19.,  -11.,    5.,   -9.,  -23., 1985
     9              4.,   11.,    4.,   14.,    4.,  -15.,   -4.,  -11., 1985
     a              5.,   10.,  -21.,    1.,   15.,  -12.,    9.,    9., 1985
     b             -6.,   -3.,   -6.,   -1.,    9.,    7.,    9.,    1., 1985
     c             -7.,   -5.,    2.,   -4.,   -4.,    1.,    3.,    0., 1985
     d             -5.,    3.,   -2.,    6.,    5.,   -4.,    3.,    0., 1985
     e              1.,   -1.,    2.,    4.,    3.,    0.,    0.,   -6./ 1985
      data gj/ -29775.,-1848., 5406.,-2131., 3059.,-2279., 1686., -373., 1990
     1           1314.,-2239., -284., 1248.,  293.,  802., -352.,  939., 1990
     2            780.,  247.,  325., -240., -423.,   84.,  141., -299., 1990
     3           -214.,  353.,   46.,  245.,  154., -109., -153., -165., 1990
     4            -69.,  -36.,   97.,   61.,   65.,  -16.,   59.,   82., 1990
     5           -178.,   69.,    3.,  -52.,   18.,    1.,  -96.,   24., 1990
     6             77.,  -64.,  -80.,    2.,  -26.,   26.,    0.,   -1., 1990
     7             21.,    5.,   17.,    9.,  -23.,    0.,   -4.,   23., 1990
     8              5.,   10.,   -1.,  -19.,  -10.,    6.,  -12.,  -22., 1990
     9              3.,   12.,    4.,   12.,    2.,  -16.,   -6.,  -10., 1990
     a              4.,    9.,  -20.,    1.,   15.,  -12.,   11.,    9., 1990
     b             -7.,   -4.,   -7.,   -2.,    9.,    7.,    8.,    1., 1990
     c             -7.,   -6.,    2.,   -3.,   -4.,    2.,    2.,    1., 1990
     d             -5.,    3.,   -2.,    6.,    4.,   -4.,    3.,    0., 1990
     e              1.,   -2.,    3.,    3.,    3.,   -1.,    0.,   -6./ 1990
      data gk/ -29692.,-1784., 5306.,-2200., 3070.,-2366., 1681., -413., 1995
     1           1335.,-2267., -262., 1249.,  302.,  759., -427.,  940., 1995
     2            780.,  262.,  290., -236., -418.,   97.,  122., -306., 1995
     3           -214.,  352.,   46.,  235.,  165., -118., -143., -166., 1995
     4            -55.,  -17.,  107.,   68.,   67.,  -17.,   68.,   72., 1995
     5           -170.,   67.,   -1.,  -58.,   19.,    1.,  -93.,   36., 1995
     6             77.,  -72.,  -69.,    1.,  -25.,   28.,    4.,    5., 1995
     7             24.,    4.,   17.,    8.,  -24.,   -2.,   -6.,   25., 1995
     8              6.,   11.,   -6.,  -21.,   -9.,    8.,  -14.,  -23., 1995
     9              9.,   15.,    6.,   11.,   -5.,  -16.,   -7.,   -4., 1995
     a              4.,    9.,  -20.,    3.,   15.,  -10.,   12.,    8., 1995
     b             -6.,   -8.,   -8.,   -1.,    8.,   10.,    5.,   -2., 1995
     c             -8.,   -8.,    3.,   -3.,   -6.,    1.,    2.,    0., 1995
     d             -4.,    4.,   -1.,    5.,    4.,   -5.,    2.,   -1., 1995
     e              2.,   -2.,    5.,    1.,    1.,   -2.,    0.,   -7., 1995
     f           75*0./                                                  1995
      data gl/ -29619.4,-1728.2, 5186.1,-2267.7, 3068.4,-2481.6, 1670.9, 2000
     1           -458.0, 1339.6,-2288.0, -227.6, 1252.1,  293.4,  714.5, 2000
     2           -491.1,  932.3,  786.8,  272.6,  250.0, -231.9, -403.0, 2000
     3            119.8,  111.3, -303.8, -218.8,  351.4,   43.8,  222.3, 2000
     4            171.9, -130.4, -133.1, -168.6,  -39.3,  -12.9,  106.3, 2000
     5             72.3,   68.2,  -17.4,   74.2,   63.7, -160.9,   65.1, 2000
     6             -5.9,  -61.2,   16.9,    0.7,  -90.4,   43.8,   79.0, 2000
     7            -74.0,  -64.6,    0.0,  -24.2,   33.3,    6.2,    9.1, 2000
     8             24.0,    6.9,   14.8,    7.3,  -25.4,   -1.2,   -5.8, 2000
     9             24.4,    6.6,   11.9,   -9.2,  -21.5,   -7.9,    8.5, 2000
     a            -16.6,  -21.5,    9.1,   15.5,    7.0,    8.9,   -7.9, 2000
     b            -14.9,   -7.0,   -2.1,    5.0,    9.4,  -19.7,    3.0, 2000
     c             13.4,   -8.4,   12.5,    6.3,   -6.2,   -8.9,   -8.4, 2000
     d             -1.5,    8.4,    9.3,    3.8,   -4.3,   -8.2,   -8.2, 2000
     e              4.8,   -2.6,   -6.0,    1.7,    1.7,    0.0,   -3.1, 2000
     f              4.0,   -0.5,    4.9,    3.7,   -5.9,    1.0,   -1.2, 2000
     g              2.0,   -2.9,    4.2,    0.2,    0.3,   -2.2,   -1.1, 2000
     h             -7.4,    2.7,   -1.7,    0.1,   -1.9,    1.3,    1.5, 2000
     i             -0.9,   -0.1,   -2.6,    0.1,    0.9,   -0.7,   -0.7, 2000
     j              0.7,   -2.8,    1.7,   -0.9,    0.1,   -1.2,    1.2, 2000
     k             -1.9,    4.0,   -0.9,   -2.2,   -0.3,   -0.4,    0.2, 2000
     l              0.3,    0.9,    2.5,   -0.2,   -2.6,    0.9,    0.7, 2000
     m             -0.5,    0.3,    0.3,    0.0,   -0.3,    0.0,   -0.4, 2000
     n              0.3,   -0.1,   -0.9,   -0.2,   -0.4,   -0.4,    0.8, 2000
     o             -0.2,   -0.9,   -0.9,    0.3,    0.2,    0.1,    1.8, 2000
     p             -0.4,   -0.4,    1.3,   -1.0,   -0.4,   -0.1,    0.7, 2000
     q              0.7,   -0.4,    0.3,    0.3,    0.6,   -0.1,    0.3, 2000
     r              0.4,   -0.2,    0.0,   -0.5,    0.1,   -0.9/         2000
      data gm/-29554.63,-1669.05, 5077.99,-2337.24, 3047.69,-2594.50,    2005
     1          1657.76, -515.43, 1336.30,-2305.83, -198.86, 1246.39,    2005
     2           269.72,  672.51, -524.72,  920.55,  797.96,  282.07,    2005
     3           210.65, -225.23, -379.86,  145.15,  100.00, -305.36,    2005
     4          -227.00,  354.41,   42.72,  208.95,  180.25, -136.54,    2005
     5          -123.45, -168.05,  -19.57,  -13.55,  103.85,   73.60,    2005
     6            69.56,  -20.33,   76.74,   54.75, -151.34,   63.63,    2005
     7           -14.58,  -63.53,   14.58,    0.24,  -86.36,   50.94,    2005
     8            79.88,  -74.46,  -61.14,   -1.65,  -22.57,   38.73,    2005
     9             6.82,   12.30,   25.35,    9.37,   10.93,    5.42,    2005
     a           -26.32,    1.94,   -4.64,   24.80,    7.62,   11.20,    2005
     b           -11.73,  -20.88,   -6.88,    9.83,  -18.11,  -19.71,    2005
     c            10.17,   16.22,    9.36,    7.61,  -11.25,  -12.76,    2005
     d            -4.87,   -0.06,    5.58,    9.76,  -20.11,    3.58,    2005
     e            12.69,   -6.94,   12.67,    5.01,   -6.72,  -10.76,    2005
     f            -8.16,   -1.25,    8.10,    8.76,    2.92,   -6.66,    2005
     g            -7.73,   -9.22,    6.01,   -2.17,   -6.12,    2.19,    2005
     h             1.42,    0.10,   -2.35,    4.46,   -0.15,    4.76,    2005
     i             3.06,   -6.58,    0.29,   -1.01,    2.06,   -3.47,    2005
     j             3.77,   -0.86,   -0.21,   -2.31,   -2.09,   -7.93,    2005
     k             2.95,   -1.60,    0.26,   -1.88,    1.44,    1.44,    2005
     l            -0.77,   -0.31,   -2.27,    0.29,    0.90,   -0.79,    2005
     m            -0.58,    0.53,   -2.69,    1.80,   -1.08,    0.16,    2005
     n            -1.58,    0.96,   -1.90,    3.99,   -1.39,   -2.15,    2005
     o            -0.29,   -0.55,    0.21,    0.23,    0.89,    2.38,    2005
     p            -0.38,   -2.63,    0.96,    0.61,   -0.30,    0.40,    2005
     q             0.46,    0.01,   -0.35,    0.02,   -0.36,    0.28,    2005
     r             0.08,   -0.87,   -0.49,   -0.34,   -0.08,    0.88,    2005
     s            -0.16,   -0.88,   -0.76,    0.30,    0.33,    0.28,    2005
     t             1.72,   -0.43,   -0.54,    1.18,   -1.07,   -0.37,    2005
     u            -0.04,    0.75,    0.63,   -0.26,    0.21,    0.35,    2005
     v             0.53,   -0.05,    0.38,    0.41,   -0.22,   -0.10,    2005
     w            -0.57,   -0.18,   -0.82/                               2005
      data gp/-29496.57,-1586.42, 4944.26,-2396.06, 3026.34,-2708.54,    2010
     1          1668.17, -575.73, 1339.85,-2326.54, -160.40, 1232.10,    2010
     2           251.75,  633.73, -537.03,  912.66,  808.97,  286.48,    2010
     3           166.58, -211.03, -356.83,  164.46,   89.40, -309.72,    2010
     4          -230.87,  357.29,   44.58,  200.26,  189.01, -141.05,    2010
     5          -118.06, -163.17,   -0.01,   -8.03,  101.04,   72.78,    2010
     6            68.69,  -20.90,   75.92,   44.18, -141.40,   61.54,    2010
     7           -22.83,  -66.26,   13.10,    3.02,  -78.09,   55.40,    2010
     8            80.44,  -75.00,  -57.80,   -4.55,  -21.20,   45.24,    2010
     9             6.54,   14.00,   24.96,   10.46,    7.03,    1.64,    2010
     a           -27.61,    4.92,   -3.28,   24.41,    8.21,   10.84,    2010
     b           -14.50,  -20.03,   -5.59,   11.83,  -19.34,  -17.41,    2010
     c            11.61,   16.71,   10.85,    6.96,  -14.05,  -10.74,    2010
     d            -3.54,    1.64,    5.50,    9.45,  -20.54,    3.45,    2010
     e            11.51,   -5.27,   12.75,    3.13,   -7.14,  -12.38,    2010
     f            -7.42,   -0.76,    7.97,    8.43,    2.14,   -8.42,    2010
     g            -6.08,  -10.08,    7.01,   -1.94,   -6.24,    2.73,    2010
     h             0.89,   -0.10,   -1.07,    4.71,   -0.16,    4.44,    2010
     i             2.45,   -7.22,   -0.33,   -0.96,    2.13,   -3.95,    2010
     j             3.09,   -1.99,   -1.03,   -1.97,   -2.80,   -8.31,    2010
     k             3.05,   -1.48,    0.13,   -2.03,    1.67,    1.65,    2010
     l            -0.66,   -0.51,   -1.76,    0.54,    0.85,   -0.79,    2010
     m            -0.39,    0.37,   -2.51,    1.79,   -1.27,    0.12,    2010
     n            -2.11,    0.75,   -1.94,    3.75,   -1.86,   -2.12,    2010
     o            -0.21,   -0.87,    0.30,    0.27,    1.04,    2.13,    2010
     p            -0.63,   -2.49,    0.95,    0.49,   -0.11,    0.59,    2010
     q             0.52,    0.00,   -0.39,    0.13,   -0.37,    0.27,    2010
     r             0.21,   -0.86,   -0.77,   -0.23,    0.04,    0.87,    2010
     s            -0.09,   -0.89,   -0.87,    0.31,    0.30,    0.42,    2010
     t             1.66,   -0.45,   -0.59,    1.08,   -1.14,   -0.31,    2010
     u            -0.07,    0.78,    0.54,   -0.18,    0.10,    0.38,    2010
     v             0.49,    0.02,    0.44,    0.42,   -0.25,   -0.26,    2010
     w            -0.53,   -0.26,   -0.79/                               2010
      data gq/-29441.46,-1501.77, 4795.99,-2445.88, 3012.20,-2845.41,    2015
     1          1676.35, -642.17, 1350.33,-2352.26, -115.29, 1225.85,    2015
     2           245.04,  581.69, -538.70,  907.42,  813.68,  283.54,    2015
     3           120.49, -188.43, -334.85,  180.95,   70.38, -329.23,    2015
     4          -232.91,  360.14,   46.98,  192.35,  196.98, -140.94,    2015
     5          -119.14, -157.40,   15.98,    4.30,  100.12,   69.55,    2015
     6            67.57,  -20.61,   72.79,   33.30, -129.85,   58.74,    2015
     7           -28.93,  -66.64,   13.14,    7.35,  -70.85,   62.41,    2015
     8            81.29,  -75.99,  -54.27,   -6.79,  -19.53,   51.82,    2015
     9             5.59,   15.07,   24.45,    9.32,    3.27,   -2.88,    2015
     a           -27.50,    6.61,   -2.32,   23.98,    8.89,   10.04,    2015
     b           -16.78,  -18.26,   -3.16,   13.18,  -20.56,  -14.60,    2015
     c            13.33,   16.16,   11.76,    5.69,  -15.98,   -9.10,    2015
     d            -2.02,    2.26,    5.33,    8.83,  -21.77,    3.02,    2015
     e            10.76,   -3.22,   11.74,    0.67,   -6.74,  -13.20,    2015
     f            -6.88,   -0.10,    7.79,    8.68,    1.04,   -9.06,    2015
     g            -3.89,  -10.54,    8.44,   -2.01,   -6.26,    3.28,    2015
     h             0.17,   -0.40,    0.55,    4.55,   -0.55,    4.40,    2015
     i             1.70,   -7.92,   -0.67,   -0.61,    2.13,   -4.16,    2015
     j             2.33,   -2.85,   -1.80,   -1.12,   -3.59,   -8.72,    2015
     k             3.00,   -1.40,    0.00,   -2.30,    2.11,    2.08,    2015
     l            -0.60,   -0.79,   -1.05,    0.58,    0.76,   -0.70,    2015
     m            -0.20,    0.14,   -2.12,    1.70,   -1.44,   -0.22,    2015
     n            -2.57,    0.44,   -2.01,    3.49,   -2.34,   -2.09,    2015
     o            -0.16,   -1.08,    0.46,    0.37,    1.23,    1.75,    2015
     p            -0.89,   -2.19,    0.85,    0.27,    0.10,    0.72,    2015
     q             0.54,   -0.09,   -0.37,    0.29,   -0.43,    0.23,    2015
     r             0.22,   -0.89,   -0.94,   -0.16,   -0.03,    0.72,    2015
     s            -0.02,   -0.92,   -0.88,    0.42,    0.49,    0.63,    2015
     t             1.56,   -0.42,   -0.50,    0.96,   -1.24,   -0.19,    2015
     u            -0.10,    0.81,    0.42,   -0.13,   -0.04,    0.38,    2015
     v             0.48,    0.08,    0.48,    0.46,   -0.30,   -0.35,    2015
     w            -0.43,   -0.36,   -0.71/                               2015
      data gr/ -29404.8, -1450.9,  4652.5, -2499.6,  2982.0, -2991.6,    2020
     1           1677.0,  -734.6,  1363.2, -2381.2,   -82.1,  1236.2,    2020
     2            241.9,   525.7,  -543.4,   903.0,   809.5,   281.9,    2020
     3             86.3,  -158.4,  -309.4,   199.7,    48.0,  -349.7,    2020
     4           -234.3,   363.2,    47.7,   187.8,   208.3,  -140.7,    2020
     5           -121.2,  -151.2,    32.3,    13.5,    98.9,    66.0,    2020
     6             65.5,   -19.1,    72.9,    25.1,  -121.5,    52.8,    2020
     7            -36.2,   -64.5,    13.5,     8.9,   -64.7,    68.1,    2020
     8             80.6,   -76.7,   -51.5,    -8.2,   -16.9,    56.5,    2020
     9              2.2,    15.8,    23.5,     6.4,    -2.2,    -7.2,    2020
     a            -27.2,     9.8,    -1.8,    23.7,     9.7,     8.4,    2020
     b            -17.6,   -15.3,    -0.5,    12.8,   -21.1,   -11.7,    2020
     c             15.3,    14.9,    13.7,     3.6,   -16.5,    -6.9,    2020
     d             -0.3,     2.8,     5.0,     8.4,   -23.4,     2.9,    2020
     e             11.0,    -1.5,     9.8,    -1.1,    -5.1,   -13.2,    2020
     f             -6.3,     1.1,     7.8,     8.8,     0.4,    -9.3,    2020
     g             -1.4,   -11.9,     9.6,    -1.9,    -6.2,     3.4,    2020
     h             -0.1,    -0.2,     1.7,     3.6,    -0.9,     4.8,    2020
     i              0.7,    -8.6,    -0.9,    -0.1,     1.9,    -4.3,    2020
     j              1.4,    -3.4,    -2.4,    -0.1,    -3.8,    -8.8,    2020
     k              3.0,    -1.4,     0.0,    -2.5,     2.5,     2.3,    2020
     l             -0.6,    -0.9,    -0.4,     0.3,     0.6,    -0.7,    2020
     m             -0.2,    -0.1,    -1.7,     1.4,    -1.6,    -0.6,    2020
     n             -3.0,     0.2,    -2.0,     3.1,    -2.6,    -2.0,    2020
     o             -0.1,    -1.2,     0.5,     0.5,     1.3,     1.4,    2020
     p             -1.2,    -1.8,     0.7,     0.1,     0.3,     0.8,    2020
     q              0.5,    -0.2,    -0.3,     0.6,    -0.5,     0.2,    2020
     r              0.1,    -0.9,    -1.1,     0.0,    -0.3,     0.5,    2020
     s              0.1,    -0.9,    -0.9,     0.5,     0.6,     0.7,    2020
     t              1.4,    -0.3,    -0.4,     0.8,    -1.3,     0.0,    2020
     u             -0.1,     0.8,     0.3,     0.0,    -0.1,     0.4,    2020
     v              0.5,     0.1,     0.5,     0.5,    -0.4,    -0.5,    2020
     w             -0.4,    -0.4,    -0.6/                               2020
      data gs/      5.7,     7.4,   -25.9,   -11.0,    -7.0,   -30.2,    2022
     1             -2.1,   -22.4,     2.2,    -5.9,     6.0,     3.1,    2022
     2             -1.1,   -12.0,     0.5,    -1.2,    -1.6,    -0.1,    2022
     3             -5.9,     6.5,     5.2,     3.6,    -5.1,    -5.0,    2022
     4             -0.3,     0.5,     0.0,    -0.6,     2.5,     0.2,    2022
     5             -0.6,     1.3,     3.0,     0.9,     0.3,    -0.5,    2022
     6             -0.3,     0.0,     0.4,    -1.6,     1.3,    -1.3,    2022
     7             -1.4,     0.8,     0.0,     0.0,     0.9,     1.0,    2022
     8             -0.1,    -0.2,     0.6,     0.0,     0.6,     0.7,    2022
     9             -0.8,     0.1,    -0.2,    -0.5,    -1.1,    -0.8,    2022
     a              0.1,     0.8,     0.3,     0.0,     0.1,    -0.2,    2022
     b             -0.1,     0.6,     0.4,    -0.2,    -0.1,     0.5,    2022
     c              0.4,    -0.3,     0.3,    -0.4,    -0.1,     0.5,    2022
     d              0.4,     0.0, 115*0.0/                               2022
C following added by RStoneback
Cf2py intent(in) isv,date,itype,alt,colat,elong
Cf2py intent(out) x,y,z,f



c
c     set initial values
c
      x     = 0.0
      y     = 0.0
      z     = 0.0
      if (date.lt.1900.0.or.date.gt.2030.0) go to 11
      if (date.gt.2025.0) write (6,960) date
  960 format (/' This version of the IGRF is intended for use up',
     1        ' to 2025.0.'/' values for',f9.3,' will be computed',
     2        ' but may be of reduced accuracy'/)
      if (date.ge.2020.0) go to 1
      t     = 0.2*(date - 1900.0)
      ll    = t
      one   = ll
      t     = t - one
c
c     SH models before 1995.0 are only to degree 10
c
      if (date.lt.1995.0) then
       nmx   = 10
       nc    = nmx*(nmx+2)
       ll    = nc*ll
       kmx   = (nmx+1)*(nmx+2)/2
      else
       nmx   = 13
       nc    = nmx*(nmx+2)
       ll    = 0.2*(date - 1995.0)
c
c     19 is the number of SH models that extend to degree 10
c
       ll    = 120*19 + nc*ll
       kmx   = (nmx+1)*(nmx+2)/2
      endif
      tc    = 1.0 - t
      if (isv.eq.1) then
       tc = -0.2
       t = 0.2
      end if
      go to 2
c
    1 t     = date - 2020.0
      tc    = 1.0
      if (isv.eq.1) then
       t = 1.0
       tc = 0.0
      end if
c
c     pointer for last coefficient in pen-ultimate set of MF coefficients...
c
      ll    = 3255
      nmx   = 13
      nc    = nmx*(nmx+2)
      kmx   = (nmx+1)*(nmx+2)/2
    2 r     = alt
      one   = colat
      !*0.017453292
      ct    = cos(one)
      st    = sin(one)
      one   = elong
      !*0.017453292
      cl(1) = cos(one)
      sl(1) = sin(one)
      cd    = 1.0
      sd    = 0.0
      l     = 1
      m     = 1
      n     = 0
      if (itype.eq.2) go to 3
c
c     conversion from geodetic to geocentric coordinates
c     (using the WGS84 spheroid)
c
      a2    = 40680631.6
      b2    = 40408296.0
      one   = a2*st*st
      two   = b2*ct*ct
      three = one + two
      rho   = sqrt(three)
      r     = sqrt(alt*(alt + 2.0*rho) + (a2*one + b2*two)/three)
      cd    = (alt + rho)/r
      sd    = (a2 - b2)/rho*ct*st/r
      one   = ct
      ct    = ct*cd -  st*sd
      st    = st*cd + one*sd
c
    3 ratio = 6371.2/r
      rr    = ratio*ratio
c
c     computation of Schmidt quasi-normal coefficients p and x(=q)
c
      p(1)  = 1.0
      p(3)  = st
      q(1)  = 0.0
      q(3)  =  ct
      do 10 k=2,kmx
       if (n.ge.m) go to 4
       m     = 0
       n     = n + 1
       rr    = rr*ratio
       fn    = n
       gn    = n - 1
    4  fm    = m
       if (m.ne.n) go to 5
       if (k.eq.3) go to 6
       one   = sqrt(1.0 - 0.5/fm)
       j     = k - n - 1
       p(k)  = one*st*p(j)
       q(k)  = one*(st*q(j) + ct*p(j))
       cl(m) = cl(m-1)*cl(1) - sl(m-1)*sl(1)
       sl(m) = sl(m-1)*cl(1) + cl(m-1)*sl(1)
       go to 6
    5  gmm    = m*m
       one   = sqrt(fn*fn - gmm)
       two   = sqrt(gn*gn - gmm)/one
       three = (fn + gn)/one
       i     = k - n
       j     = i - n + 1
       p(k)  = three*ct*p(i) - two*p(j)
       q(k)  = three*(ct*q(i) - st*p(i)) - two*q(j)
c
c     synthesis of x, y and z in geocentric coordinates
c
    6  lm    = ll + l
       one   = (tc*gh(lm) + t*gh(lm+nc))*rr
       if (m.eq.0) go to 9
       two   = (tc*gh(lm+1) + t*gh(lm+nc+1))*rr
       three = one*cl(m) + two*sl(m)
       x     = x + three*q(k)
       z     = z - (fn + 1.0)*three*p(k)
       if (st.eq.0.0) go to 7
       y     = y + (one*sl(m) - two*cl(m))*fm*p(k)/st
       go to 8
    7  y     = y + (one*sl(m) - two*cl(m))*q(k)*ct
    8  l     = l + 2
       go to 10
    9  x     = x + one*q(k)
       z     = z - (fn + 1.0)*one*p(k)
       l     = l + 1
   10 m     = m + 1
c
c     conversion to coordinate system specified by itype
c
      one   = x
      x     = x*cd +   z*sd
      z     = z*cd - one*sd
      f     = sqrt(x*x + y*y + z*z)
c
      return
c
c     error return if date out of bounds
c
   11 f     = 1.0d8
      write (6,961) date
  961 format (/' This subroutine will not work with a date of',
     1        f9.3,'.  Date must be in the range 1900.0.ge.date',
     2        '.le.2030.0. On return f = 1.0d8., x = y = z = 0.')
      return
      end
