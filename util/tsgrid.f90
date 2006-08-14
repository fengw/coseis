! Generate TeraShake grid from 2D mesh and topography
program grid
use m_tscoords
implicit none
real :: r, dx, h, o1, o2, xx, yy, h1, h2, h3, h4, ell(3), x0, y0, z0, xf(6), yf(6), rf(6), zf, exag
integer :: n(3), nn, npml, nrect, i, j, k, l, j1, k1, l1, j2, k2, l2, jf0, kf0, lf0, &
  nf, nf1, nf2, nf3, reclen
real, allocatable :: x(:,:,:,:), w(:,:,:,:), s(:,:,:), t(:,:)
character :: endian

! Model parameters
open( 1, file='dx', status='old' )
read( 1, * ) dx
close( 1 )
ell = (/ 600, 300, 80 /) * 1000
exag = 1.
npml = 10
xf = (/ 265864.,293831.,338482.,364062.,390075.,459348. /)
yf = (/ 183273.,187115.,200421.,212782.,215126.,210481. /)
zf = 16000.
x0 = .5 * ( minval(xf) + maxval(xf) )
y0 = .5 * ( minval(yf) + maxval(yf) )
!x0 = 365000.
!y0 = 202000.

! Dimentions
n = nint( ell / dx ) + 1
print *, 'n =', n
j = n(1)
k = n(2)
l = n(3)
nn = j * k * l
allocate( x(j,k,1,3), w(j,k,1,3), s(j,k,1) )
open( 1, file='nn' )
write( 1, * ) nn
close( 1 )

! 2D mesh
forall( i=1:n(1) ) x(i,:,:,1) = dx*(i-1)
forall( i=1:n(2) ) x(:,i,:,2) = dx*(i-1)

! Fault length
nf = size( xf, 1 )
rf(1) = 0
do i = 2, nf
  h1 = xf(i) - xf(i-1)
  h2 = yf(i) - yf(i-1)
  rf(i) = rf(i-1) + sqrt( h1*h1 + h2*h2 )
end do

! Fault indices
nf1 = nint( rf(nf) / dx )
nf2 = 0
nf3 = nint( zf / dx )
jf0 = nint( x0 / dx - .5*nf1 ) + 1
kf0 = nint( y0 / dx ) + 1
lf0 = n(3) - nf3

! Interpolate fault
j1 = 1 + npml
j2 = n(1) - npml
k = kf0
i = 1
do j = j1+1, j2-1
  do while( i < nf-1 .and. dx*(j-jf0) > rf(i+1) )
    i = i + 1
  end do
  x(j,k,1,1) = xf(i) + (xf(i+1)-xf(i)) / (rf(i+1)-rf(i)) * (dx*(j-jf0)-rf(i))
  x(j,k,1,2) = yf(i) + (yf(i+1)-yf(i)) / (rf(i+1)-rf(i)) * (dx*(j-jf0)-rf(i))
end do

! Orogonal elements next to the fault
j1 = jf0
j2 = jf0 + nf1
k  = kf0
h1 = x(j2,k,1,1) - x(j1,k,1,1)
h2 = x(j2,k,1,2) - x(j1,k,1,2)
h = sqrt( h1*h1 + h2*h2 )
do j = j1-1, j2+1
  h1 = 0
  do i = 1, 4
    h1 = x(j+i,k,1,1) - x(j-i,k,1,1)
    h2 = x(j+i,k,1,2) - x(j-i,k,1,2)
  end do
  h = sqrt( h1*h1 + h2*h2 )
  x(j,k-1,1,1) = x(j,k,1,1) + h2 * dx / h
  x(j,k-1,1,2) = x(j,k,1,2) - h1 * dx / h
  x(j,k+1,1,1) = x(j,k,1,1) - h2 * dx / h
  x(j,k+1,1,2) = x(j,k,1,2) + h1 * dx / h
end do

! Blend fault to x-bounaries
j1 = 1 + npml
j2 = jf0 - 1
forall( j=j1+1:j2-1 )
  x(j,:,:,:) = x(j1,:,:,:)*(j2-j)/(j2-j1) + x(j2,:,:,:)*(j-j1)/(j2-j1)
end forall
j1 = jf0 + nf1 + 1
j2 = n(1) - npml
forall( j=j1+1:j2-1 )
  x(j,:,:,:) = x(j1,:,:,:)*(j2-j)/(j2-j1) + x(j2,:,:,:)*(j-j1)/(j2-j1)
end forall

! Blend fault to y-bounaries
k1 = 1 + npml
k2 = kf0 - 1
forall( k=k1+1:k2-1 )
  x(:,k,:,:) = x(:,k1,:,:)*(k2-k)/(k2-k1) + x(:,k2,:,:)*(k-k1)/(k2-k1)
end forall
k1 = kf0 + 1
k2 = n(2) - npml
forall( k=k1+1:k2-1 )
  x(:,k,:,:) = x(:,k1,:,:)*(k2-k)/(k2-k1) + x(:,k2,:,:)*(k-k1)/(k2-k1)
end forall

! lon/lat
w = x
call ts2ll( w, 1, 2 )
if ( any( w /= w ) ) stop 'NaNs in lon/lat'
print *, 'longitude range: ', minval( w(:,:,:,1) ), maxval( w(:,:,:,1) )
print *, 'latgitude range: ', minval( w(:,:,:,2) ), maxval( w(:,:,:,2) )

! Topo
allocate( t(960,780) )
endian = 'l'
if ( iachar( transfer( 1, 'a' ) ) == 0 ) endian = 'b'
inquire( iolength=reclen ) t
open( 1, file='topo.'//endian, recl=reclen, form='unformatted', access='direct', status='old' )
read( 1, rec=1 ) t
close( 1 )
t = t * exag
o1 = 15. - 121.5 * 3600.
o2 = 15. +  30.5 * 3600.
h  = 30.
do k1 = 1, size(w,2)
do j1 = 1, size(w,1)
  xx = ( ( w(j1,k1,1,1) * 3600 ) - o1 ) / h
  yy = ( ( w(j1,k1,1,2) * 3600 ) - o2 ) / h
  j = int( xx ) + 1
  k = int( yy ) + 1
  h1 =  xx - j + 1
  h2 = -xx + j
  h3 =  yy - k + 1
  h4 = -yy + k
  w(j1,k1,1,3) = ( &
    h2 * h4 * t(j,k)   + &
    h1 * h4 * t(j+1,k) + &
    h2 * h3 * t(j,k+1) + &
    h1 * h3 * t(j+1,k+1) )
end do
end do
x(:,:,:,3) = w(:,:,:,3)

! PML regions are orthogonal
j = n(1)
k = n(2)
do i = npml-1,0,-1
  w(i+1,:,:,:) = w(i+2,:,:,:)
  w(j-i,:,:,:) = w(j-i-1,:,:,:)
  w(:,i+1,:,:) = w(:,i+2,:,:)
  w(:,k-i,:,:) = w(:,k-i-1,:,:)
end do
z0 = sum( w(:,:,:,3) ) / ( n(1) * n(2) )
print *, 'elevation range: ', minval( w(:,:,:,3) ), maxval( w(:,:,:,3) )

! 2D files
inquire( iolength=reclen ) x(:,:,:,1)
open( 1, file='x', recl=reclen, form='unformatted', access='direct' )
open( 2, file='y', recl=reclen, form='unformatted', access='direct' )
open( 3, file='z', recl=reclen, form='unformatted', access='direct' )
write( 1, rec=1 ) x(:,:,:,1)
write( 2, rec=1 ) x(:,:,:,2)
write( 3, rec=1 ) x(:,:,:,3)
close( 1 )
close( 2 )
close( 3 )

! 3D files
inquire( iolength=reclen ) x(:,:,:,1)
open( 1, file='x1', recl=reclen, form='unformatted', access='direct' )
open( 2, file='x2', recl=reclen, form='unformatted', access='direct' )
open( 3, file='x3', recl=reclen, form='unformatted', access='direct' )
open( 7, file='rlon', recl=reclen, form='unformatted', access='direct' )
open( 8, file='rlat', recl=reclen, form='unformatted', access='direct' )
open( 9, file='rdep', recl=reclen, form='unformatted', access='direct' )
do l = 1, n(3)
  write( 1, rec=l ) x(:,:,:,1)
  write( 2, rec=l ) x(:,:,:,2)
  write( 7, rec=l ) w(:,:,:,1)
  write( 8, rec=l ) w(:,:,:,2)
end do
s = 0
l1 = npml + 1
l2 = n(3) - nf3
do l = 1, l1
  write( 3, rec=l ) -dx*(n(3)-l) + z0 + s
  write( 9, rec=l )  dx*(n(3)-l1) - z0 + w(:,:,:,3)
end do
do l = l1+1, l2-1
  write( 3, rec=l ) -dx*(n(3)-l) + z0*(l2-l)/(l2-l1) + w(:,:,:,3)*(l-l1)/(l2-l1)
  write( 9, rec=l )  dx*(n(3)-l) + (w(:,:,:,3)-z0)*(l2-l)/(l2-l1)
end do
do l = l2, n(3)
  write( 3, rec=l ) -dx*(n(3)-l) + w(:,:,:,3)
  write( 9, rec=l )  dx*(n(3)-l) + s
end do
close( 1 )
close( 2 )
close( 3 )
close( 7 )
close( 8 )
close( 9 )

! Fault prestress
deallocate( t, x, s, w )
allocate( s(n(1),1,n(3)), t(1991,161) )
i = nint( dx / 100. )
j1 = jf0
j2 = jf0 + nf1
nf1 = min( nf1, (size(t,1)-1)/i )
j1 = j2 - nf1
nf3 = min( nf3, (size(t,2)-1)/i )
lf0 = n(3) - nf3
l1 = lf0
l2 = lf0 + nf3
inquire( iolength=reclen ) t
open( 1, file='tn.'//endian, recl=reclen, form='unformatted', access='direct', status='old' )
read( 1, rec=1 ) t
close( 1 )
print *, 'normal traction range: ', minval( t ), maxval( t )
s = -maxval( abs( t ) )
do l = l1, l2
do j = j1, j2
  k1 = i * (j2-j) + 1
  k2 = i * (l2-l) + 1
  s(j,1,l) = t(k1,k2)
end do
end do
inquire( iolength=reclen ) s
open( 1, file='tn', recl=reclen, form='unformatted', access='direct' )
write( 1, rec=1 ) s
close( 1 )
inquire( iolength=reclen ) t
open( 1, file='th.'//endian, recl=reclen, form='unformatted', access='direct', status='old' )
read( 1, rec=1 ) t
close( 1 )
print *, 'shear traction range: ', minval( t ), maxval( t )
s = 0.
do l = l1, l2
do j = j1, j2
  k1 = i * (j2-j) + 1
  k2 = i * (l2-l) + 1
  s(j,1,l) = t(k1,k2)
end do
end do
inquire( iolength=reclen ) s
open( 1, file='th', recl=reclen, form='unformatted', access='direct' )
write( 1, rec=1 ) s
close( 1 )

! Metadata
i = nf3 / 2
open( 1, file='insord.m' )
write( 1, * ) 'dx      = ', dx, ';'
write( 1, * ) 'npml    = ', npml, ';'
write( 1, * ) 'n       = [ ', n, ' ];'
write( 1, * ) 'nn      = [ ', n + (/ 0, 1, 0 /), ' ];'
write( 1, * ) 'ihypo   = [ ', jf0+i,     kf0, -1-i, ' ];'
write( 1, * ) 'ihypo   = [ ', jf0-i+nf1, kf0, -1-i, ' ];'
write( 1, * ) 'mus     = [ 1. ''zone''', jf0, 0, -1-nf3, jf0+nf1, 0, -1, ' ];'
write( 1, * ) 'endian  = ''', endian, ''';'
close( 1 )

end program

