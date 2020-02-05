      subroutine ecef_to_geodetic(latitude,lon,h,posx,posy,posz,num)
      ! converts from ECEF coordinates to Geodetic
      ! posx,y,z is the 3 elements of positions in km
      ! latitude returned in Radians
      ! longitude returned in Radians
      ! h is the geodetic altitude, returned in km

      ! pos is (x,y,z) in ECEF coordinates
      INTEGER num, i
      real*8 posx(num), posy(num), posz(num)
      real*8 a,b,ellip,e2,p,e_prime,theta,r_n
      real*8 lon(num), latitude(num), h(num)
      real*8 pi


cf2py intent(in) posx, posy, posz
Cf2py intent(out)  latitude, lon, h
Cf2py integer intent(hide), depend(posx) :: num=shape(posx,0)

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
      DO i=1,num
      p = dsqrt(posx(i)**2+posy(i)**2)
      e_prime = dsqrt((a**2-b**2)/b**2)
      theta = datan(posz(i)*a/(p*b))
      latitude(i) = datan( (posz(i)+e_prime**2*b*dsin(theta)**3)/
     &(p-ellip**2*a*dcos(theta)**3))

      r_n = a/dsqrt(1.d0-ellip**2*dsin(latitude(i))**2)
      ! geodetic height
      h(i) = p/dcos(latitude(i)) - r_n

      lon(i) = datan(posy(i)/posx(i))
      ! Based on values from atan2() from Python
      if (posx(i).lt.0) then
         if (posy(i).lt.0) lon(i)=lon(i)-pi
         if (posy(i).ge.0) lon(i)=lon(i)+pi
      end if
      if (posx(i).eq.0) then
         if (posy(i).gt.0) lon(i)=lon(i)+pi/2.D0
         if (posy(i).lt.0) lon(i)=lon(i)-pi/2.D0
      end if
      p=180.D0/pi
      latitude(i)=latitude(i)*p
      lon(i)=lon(i)*p
      end do

      return
      end
