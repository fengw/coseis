! Field input and output
! XXX TODO: fill, interpolation
module m_fieldio
implicit none
type t_io
  character(32) :: filename
  character(4) :: field
  character(8) :: tfunc
  character(3) :: mode
  integer :: ii(3,4), nb, ib, fh
  real :: x1(3), x2(3), val, period
  real, pointer :: buff(:,:)
  type( t_io ), pointer :: next
end type t_io
type( t_io ), pointer :: pio0, p, pprev
real :: iotimer
integer, private :: itdebug = -1, idebug
contains

! Append linked list item
subroutine pappend
allocate( p%next )
p => p%next
p%next => pio0
end subroutine

! Remove linked list item
subroutine pdelete
pprev%next => p%next
if ( associated( p%buff ) ) deallocate( p%buff )
deallocate( p )
p => pprev
end subroutine

! Field I/O initialization
subroutine fieldio_init
use m_globals
use m_util
use m_collective
real :: rr
integer :: i1(3), i2(3), n(3), noff(3), i

! Store locations
if ( master ) open( 1, file='locations', status='replace' )

! Loop over output zones
p => pio0
loop: do while( p%next%field /= 'head' )
pprev => p
p => p%next

! Locate nearest node/cell to given location
i = scan( p%mode, 'xX' )
if ( i > 0 ) then
  p%mode(i:i) = ' '
  n = nn + 2 * nhalo
  noff = nnoff + nhalo
  rr = huge( rr )
  select case( p%mode(i:i) )
  case( 'x' )
    i1 = i1core
    i2 = i2core
    call radius( s2, w1, p%x1, i1, i2 )
  case( 'X' )
    i1 = max( i1core, i1cell )
    i2 = min( i2core, i2cell )
    call radius( s2, w2, p%x1, i1, i2 )
  end select
  call scalar_set_halo( s2, rr, i1, i2 )
  call reduceloc( rr, i1, s2, 'allmin', n, noff, 0 )
  p%ii(1,1:3) = i1
  p%ii(2,1:3) = i1
  if ( rr > dx * dx ) call pdelete
  if ( master ) write( 1, * ) i1, p%x1, trim( p%filename )
end if

end do loop
if ( master ) close( 1 )

end subroutine

!------------------------------------------------------------------------------!

! Field I/O sequence
subroutine fieldio( passes, field, f )
use m_globals
use m_util
use m_collective
character(*), intent(in) :: passes, field
real, intent(inout) :: f(:,:,:)
character(4) :: pass
integer :: i1(3), i2(3), i3(3), i4(3), di(3), m(4), n(4), o(4), &
  it1, it2, dit, i, j, k, l, ipass
real :: val

! Start timer
val = timer( 2 )

! Pass loop
do ipass = 1, len( passes )
pass = passes(ipass:ipass)
p => pio0

! I/O list loop
loop: do while( p%next%field /= 'head' )
pprev => p
p => p%next

! 4D slice
i1 = p%ii(1,1:3) - nnoff
i2 = p%ii(2,1:3) - nnoff
di = p%ii(3,1:3)
it1 = p%ii(1,4)
it2 = p%ii(2,4)
dit = p%ii(3,4)

! Time indices
if ( it > it2 ) then
  call pdelete
  cycle loop
end if
if ( it < it1 ) cycle loop
if ( modulo( it - it1, dit ) /= 0 ) cycle loop

! Spatial indices
i3 = i1
i4 = i2
where( i1 < i1core ) i1 = i1 + ( ( i1core - i1 - 1 ) / di + 1 ) * di
where( i2 > i2core ) i2 = i1 + (   i2core - i1     ) / di       * di
m(1:3) = ( i4 - i3 ) / di + 1
n(1:3) = ( i2 - i1 ) / di + 1
o(1:3) = ( i1 - i3 ) / di

! Dimensionality
do i = 1, 3
  if ( size( f, i ) == 1 ) then
    if ( n(i) < 1 ) then
      call pdelete
      cycle loop
    end if
    i1(i) = 1
    i2(i) = 1
    m(i) = 1
    n(i) = 1
    o(i) = 0
  end if
end do

! Pass test
if ( field /= p%field ) cycle loop
if ( pass == '<' .and. p%mode(2:2) == 'w' ) cycle loop
if ( pass == '>' .and. p%mode(2:2) /= 'w' ) cycle loop

! I/O
val = p%val * time_function( p%tfunc, tm, dt, p%period )
select case( p%mode )
case( '=c', '+c' )
  call setcube( f, w1, i1, i2, p%x1, p%x2, val, p%mode )
case( '=C', '+C' )
  call setcube( f, w2, i1, i2, p%x1, p%x2, val, p%mode )
case( '=' )
  do l = i1(3), i2(3), di(3)
  do k = i1(2), i2(2), di(2)
  do j = i1(1), i2(1), di(1)
    f(j,k,l) = val
  end do
  end do
  end do
case( '+' )
  do l = i1(3), i2(3), di(3)
  do k = i1(2), i2(2), di(2)
  do j = i1(1), i2(1), di(1)
    f(j,k,l) = f(j,k,l) + val
  end do
  end do
  end do
case( '=s' )
  call random_number( s2 )
  do l = i1(3), i2(3), di(3)
  do k = i1(2), i2(2), di(2)
  do j = i1(1), i2(1), di(1)
    f(j,k,l) = val * s2(j,k,l)
  end do
  end do
  end do
case( '+s' )
  call random_number( s2 )
  do l = i1(3), i2(3), di(3)
  do k = i1(2), i2(2), di(2)
  do j = i1(1), i2(1), di(1)
    f(j,k,l) = f(j,k,l) + val * s2(j,k,l)
  end do
  end do
  end do
case( '=r', '+r' )
  if ( .not. associated( p%buff ) ) then
    allocate( p%buff(n(1)*n(2)*n(3),p%nb) )
    p%ib = p%nb
    p%fh = -1
    if ( mpin /= 0 ) p%fh = file_null
  end if
  if ( p%ib == p%nb ) then
    n(4) = min( p%nb, ( it2 - it  ) / dit + 1 )
    m(4) = ( it2 - it1 ) / dit + 1
    o(4) = ( it  - it1 ) / dit
    str = 'in/' // p%filename
    if ( any( n(1:3) /= m(1:3) ) .and. mpin == 0 ) write( str, '(a,i6.6)' ) trim( str ), ipid
    call rio2( p%fh, p%buff(:,:n(4)), 'w', str, m, n, o, mpin )
    p%ib = 0
    if ( any( n < 1 ) ) then
      call pdelete
      cycle loop
    end if
  end if
  i = 0
  p%ib = p%ib + 1
  select case( p%mode )
  case( '=r' )
    do l = i1(3), i2(3), di(3)
    do k = i1(2), i2(2), di(2)
    do j = i1(1), i2(1), di(1)
      i = i + 1
      f(j,k,l) = p%buff(i,p%ib)
    end do
    end do
    end do
  case( '+r' )
    do l = i1(3), i2(3), di(3)
    do k = i1(2), i2(2), di(2)
    do j = i1(1), i2(1), di(1)
      i = i + 1
      f(j,k,l) = f(j,k,l) + p%buff(i,p%ib)
    end do
    end do
    end do
  end select
  ! XXX TODO: fill, interpolate
case( '=w' )
  if ( .not. associated( p%buff ) ) then
    allocate( p%buff(n(1)*n(2)*n(3),p%nb) )
    p%ib = 0
    p%fh = -1
    if ( mpout /= 0 ) p%fh = file_null
  end if
  if ( modulo( it, itstats ) /= 0 ) then
    select case( p%field )
    case( 'vm2' ); call vector_norm( f, vv, i1, i2, di )
    case( 'um2' ); call vector_norm( f, uu, i1, i2, di )
    case( 'wm2' ); call tensor_norm( f, w1, w2, i1, i2, di )
    case( 'am2' ); call vector_norm( f, w1, i1, i2, di )
    end select
  end if
  i = 0
  p%ib = p%ib + 1
  do l = i1(3), i2(3), di(3)
  do k = i1(2), i2(2), di(2)
  do j = i1(1), i2(1), di(1)
    i = i + 1
    p%buff(i,p%ib) = f(j,k,l)
  end do
  end do
  end do
  if ( p%ib == p%nb .or. it == it2 .or. modulo( it, itcheck ) == 0 ) then
    n(4) = p%ib
    m(4) = ( it2 - it1 ) / dit + 1
    o(4) = ( it  - it1 ) / dit + 1 - n(4)
    str = 'out/' // p%filename
    if ( any( n(1:3) /= m(1:3) ) .and. mpout == 0 ) write( str, '(a,i6.6)' ) trim( str ), ipid
    call rio2( p%fh, p%buff(:,:n(4)), 'w', str, m, n, o, mpout )
    p%ib = 0
    if ( any( n < 1 ) ) then
      call pdelete
      cycle loop
    end if
  end if
case default
  write( 0, * ) "bad i/o mode '", trim( p%mode ), "' for ", trim( p%filename )
  stop
end select

end do loop
end do

! Debug output
if ( debug > 2 .and. it <= 8 ) then
  if ( itdebug /= it ) then
    itdebug = it
    idebug = 0
  end if
  idebug = idebug + 1
  write( str, '(a,i6.6,a,i6.6,a)' ) 'debug/f', it, '-', ipid, field
  open( 1, file=str, status='new' )
  do l = 1, size( f, 3 )
    write( 1, * ) ip3, field, l
    do k = 1, nm(2)-1
      write( 1, * ) f(:,k,l)
    end do
  end do
  close( 1 )
end if

! Timer
iotimer = iotimer + timer(2)

end subroutine

!------------------------------------------------------------------------------!

subroutine setcube( f, x, i1, i2, x1, x2, r, mode )
real, intent(inout) :: f(:,:,:)
real, intent(in) :: x(:,:,:,:), x1(3), x2(3), r
integer, intent(in) :: i1(3), i2(3)
character(*), intent(in) :: mode
integer :: n(3), o(3), j, k, l
n = (/ size(f,1), size(f,2), size(f,3) /)
o = 0
where ( n == 1 ) o = 1 - i1
select case( mode(1:1) )
case( '=' )
  do l = i1(3), i2(3)
  do k = i1(2), i2(2)
  do j = i1(1), i2(1)
  if( x(j,k,l,1) >= x1(1) .and. x(j,k,l,1) <= x2(1) .and. &
      x(j,k,l,2) >= x1(2) .and. x(j,k,l,2) <= x2(2) .and. &
      x(j,k,l,3) >= x1(3) .and. x(j,k,l,3) <= x2(3) ) &
      f(j+o(1),k+o(2),l+o(3)) = r
  end do
  end do
  end do
case( '+' )
  do l = i1(3), i2(3)
  do k = i1(2), i2(2)
  do j = i1(1), i2(1)
  if( x(j,k,l,1) >= x1(1) .and. x(j,k,l,1) <= x2(1) .and. &
      x(j,k,l,2) >= x1(2) .and. x(j,k,l,2) <= x2(2) .and. &
      x(j,k,l,3) >= x1(3) .and. x(j,k,l,3) <= x2(3) ) &
      f(j+o(1),k+o(2),l+o(3)) = f(j+o(1),k+o(2),l+o(3)) + r
  end do
  end do
  end do
case default; stop 'error in cube'
end select
end subroutine

end module
