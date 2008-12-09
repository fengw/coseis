! Resample material arrays
module m_resample
implicit none
contains

subroutine resample
use m_globals
use m_collective
use m_bc
use m_util
integer :: i, i1(3), i2(3), bc(3)

if ( master ) write( 0, * ) 'Resample material model'

! Mass ratio
s2 = mr * vc
call average( mr, s2, i1node, i2node, -1 )
call invert( mr )
call scalar_swap_halo( mr, nhalo )
call scalar_bc( mr, bc1, bc2, i1bc, i2bc )

! Invert cell volume
call invert( vc )

! Viscosity, bc=4 means copy into halo for resampling at the node
bc = 4
i1 = i1bc - 1
i2 = i2bc
call scalar_bc( gam, bc, bc, i1, i2 )
s2 = gam * dt
call average( gam, s2, i1node, i2node, -1 )
call set_halo( gam, 0., i1bc, i2bc )
call scalar_swap_halo( gam, nhalo )
call scalar_bc( gam, bc1, bc2, i1bc, i2bc )

! Zero hourglass viscosity outside boundary, and at fault cell
i1 = i1bc
i2 = i2bc - 1
call set_halo( yy, 0., i1, i2 )
select case( ifn )
case( 1 ); i = ihypo(1); yy(i,:,:) = 0.
case( 2 ); i = ihypo(2); yy(:,i,:) = 0.
case( 3 ); i = ihypo(3); yy(:,:,i) = 0.
end select

! Initial state
tm = 0.
vv = 0.
uu = 0.
w1 = 0.
!z1 = 0.
!z2 = 0.
sl = 0.
p1 = 0.
p2 = 0.
p3 = 0.
p4 = 0.
p5 = 0.
p6 = 0.
g1 = 0.
g2 = 0.
g3 = 0.
g4 = 0.
g5 = 0.
g6 = 0.

w2 = 0.
s1 = 0.
s2 = 0.

end subroutine

end module

