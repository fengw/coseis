#!/usr/bin/env python 

""" Simulation setup and run """

import os, sys, string
import numpy, pylab, pyproj, sord
from glob import glob

from sord.extras import source, coord

mpout = 0   # mpi separate files ( this is just for Kernel )

_flag = 3
mm = 'squashed'

if _flag == 1:
    print 'do the homogeneous test...'
    
    np3 = 5,6,7  # 210
    vm = 'uhs'
    simtest = 'homo_test'
    T = 30.
    L = 90000., 60000., -50000.0
    dx = 500.0, 500.0, -500.0; npml = 10

elif _flag == 2:
    np3 = 5,6,7
    vm = 'm1d'
    simtest = 'm1d_test'
    T = 50.
    L = 90000., 60000., -50000.0
    dx = 500.0, 500.0, -500.0; npml = 10

else:
    np3 = 5,6,7 
    np3 = 10,10,7
    #np3 = 25, 25, 10
    np3 = 15,12,10
    vm = 'cvm'
    simtest = 'cvm_test'
    T = 30.
    # terashake
    # L = 600000.0, 300000.0, -80000.0
    # chino hills(14383980)
    #L = 160000.0, 120000.0, -30000.0
    L = 90000.0, 60000.0, -30000.0
    #dx = 500.0, 500.0, -500.0; npml = 10
    #dx = 300.0, 300.0, -300.0; npml = 20
    dx = 200.0, 200.0, -200.0; npml = 20
    
# number of mesh nodes, nx ny nz
nn = [
   int( L[0] / dx[0] + 1.00001 ),
   int( L[1] / dx[1] + 1.00001 ),
   int( L[2] / dx[2] + 1.00001 ),
   ]
print nn   # number of nodes

bc1 = 10, 10, 0	    # PML boundary conditions & free surface in Z
bc2 = 10, 10, 10    # PML boundary conditions
dt = dx[0] / 12500.
print 'dt = ', dt
nt = int( T / dt + 1.00001 )

# viscosity, density, and velocity model
if vm == 'uhs':
   fieldio = [
      ( '=', 'gam', [], 0.0 ),
      ( '=', 'rho', [], 3000.),
      ( '=', 'vp',  [], 6500.),
      ( '=', 'vs',  [], 3500.),
   ]
elif vm == 'cvm':
   runname = 'kernels/'+string.join(('ker','cvm4',str(dx[0])),'-')
   cvmrun = '~/run/'+runname+'/'
   fieldio = [
      ( '=', 'gam', [], 0.2 ),
      ( '=r', 'rho', [], cvmrun+'rho' ),
      ( '=r', 'vp', [], cvmrun+'vp' ),
      ( '=r', 'vs', [], cvmrun+'vs' ),
   ]
elif vm == 'm1d':
    fieldio = [
	( '=', 'gam', [], 0.2 ),
    ]
    _layers = [
        (  0.0, 5.5, 3.18, 2.4  ),
        (  5.5, 6.3, 3.64, 2.67 ),
        (  8.4, 6.3, 3.64, 2.67 ),
        ( 16.0, 6.7, 3.87, 2.8  ),
        ( 35.0, 7.8, 4.5,  3.0  ),
    ]
    for _dep, _vp, _vs, _rho in _layers:
        _i = int( -1000.0 * _dep / dx[2] + 1.5 )
        fieldio += [
            ( '=',  'rho', [(),(),(_i,-1),()], 1000.0 * _rho ),
            ( '=',  'vp',  [(),(),(_i,-1),()], 1000.0 * _vp  ),
            ( '=',  'vs',  [(),(),(_i,-1),()], 1000.0 * _vs  ),
        ]
else:
   sys.exit( 'bad vm' )

# topography
if mm == 'topo_flat':
   fieldio += [ ( '=r', 'x3', [], 'tmp/z3' ) ]

# source and receiver locations
if simtest == 'homo_test':
    xs, ys, zs = 20000.0, 30000.0, -25000.
    xr, yr, zr = 70000.0, 30000.0, -25000.    
    stanam = 'sta'
    srcnam = 'doub'
    srcnam = 'expl'

elif simtest == 'm1d_test':
    xs, ys, zs = 20000.0, 30000.0, -25000.
    xr, yr, zr = 70000.0, 30000.0, 0.0    
    stanam = 'sta'
    srcnam = 'expl'
    srcnam = 'doub'

elif simtest == 'cvm_test':
    # real earthquake and real stations
    # (you can read the id and lat,lon,dep from pre-files)
    srcnam = '14383980'  # use the event id might be a good idea
    lons, lats, dep = -117.761, 33.953, 14700.0
    
    dtype = dict( names=('name', 'lat', 'lon'), formats=('S8', 'f4', 'f4') )
    sta = numpy.loadtxt( 'data/station-list', dtype, usecols=(0,1,2) )
    
    #select stations which will show the basin excitation
    # attention to the cvm model used for different stations
    #stanam = 'CI.LTP'
    stanam = 'CI.DLA'
    #stanam = 'CI.WTT'
    #stanam = 'CI.USC'
    #stanam = 'CI.LAF'
    #stanam = 'CI.STS'

    # depend on the stanam, to find the locations...
    for j in xrange( len(sta) ):
	if sta[j]['name'] == stanam:
	    lonr = sta['lon'][j]
	    latr = sta['lat'][j]
	    srlon = lons, lonr
	    srlat = lats, latr
	    break

    # define projection based on each source and receiver pair
    #_origin = srlon, srlat
    #_proj = pyproj.Proj( proj='utm', zone=11, ellps='WGS84' )
    #_projection = sord.coord.Transform( _proj, origin=_origin )
    
    # define projection baseon on NEV coord( centered at soruce )
    # consider the projection very carefully
    _elon = -118.0, -118.1
    _elat = 33.90,33.90   # for CI.DLA
    
    _origin_ne = _elon, _elat 
    _proj = pyproj.Proj( proj='utm', zone=11, ellps='WGS84' )
    projection_ne = sord.coord.Transform( _proj, origin=_origin_ne )
    
    # use general NE projection
    xr, yr = projection_ne(lonr, latr )
    zr = 0.0    #you can specify the depth of the receiver
    xs, ys = projection_ne(lons, lats )
    zs = -dep
    # correction of location in local system under the projection of utm 
    # which use center of the the source-receiver pair as the origin
    xr, yr = xr+0.5*L[0], yr+0.5*L[1]
    xs, ys = xs+0.5*L[0], ys+0.5*L[1]
    print 'source logical horizontal location ', xs,ys
    print 'receiver logical horizontal location: ', xr, yr

else: 
    pass

# source and receiver grid
xgs,ygs,zgs = (	
	xs / dx[0] + 1.0,   
	ys / dx[1] + 1.0,
	zs / dx[2] + 1.0,
	)
# receiver grid coord
xgr,ygr,zgr = (
	 xr / dx[0] + 1.0, 
	 yr / dx[1] + 1.0, 
	 zr / dx[2] + 1.0,
	 )

# compute center of source-receiver pair ( origin )
# and the normal vector of s-r pair ( normal )
if 2:
    # you can write them in to files
    print 'source logical horizontal grid ', xgs,ygs
    print 'receiver logical horizontal grid: ', xgr, ygr
    _dim = 2
    _srorigin = (0.5*(xgs+xgr)/_dim+nn[0]/_dim,0.5*(ygs+ygr)/_dim+nn[1]/_dim,0.5*nn[2]/_dim+nn[2]/_dim)
    _theta = numpy.arctan2(ygs-ygr,xgs-xgr)
    _srnormal = ( -numpy.sin(_theta), numpy.cos(_theta), 0 )
    print 'origin is: ',_srorigin
    print 'normal is: ',_srnormal

# source-receiver distance
_d = numpy.sqrt( (xr-xs)**2 + (yr-xs)**2 + (zr-zs)**2 )
print 'source-receiver distance is: ', _d

# The reason why we use from 1.5 to -1.5 for slice is because this will be used in fortran
# and the index in fortran begins at 1 not at 0 as we want it to be.
# At the same time, the velocity model is also from 1.5 to -1.5 because of the 
# same reason. !!!
vol = (1.5,-1.5,1),(1.5,-1.5,1),(1.5,-1.5,1),(1,-1,1)

cmpid = 1
fd = (1,2,3)
ssr = string.join((vm,srcnam,stanam,'run',str(dx[0])),'.')
ss = string.join((vm,srcnam,stanam),'.')

ssc = string.join(( string.join((ss,str(dx[0]),'ker'),'-'),str(cmpid)),'.')

# variable need to be delete if you want to use sord.run( local() )
# common variables:
# vm, mm, vol, fd, T, L, simsr, stanam, srcnam,  xr, yr, zr, xs, ys, zs 
# variables for simsr == 'cvm_test'
# lons, lats, dep, dtype, sta, lonr, latr, srlon, srlat


# simulation source and receiver and field to store
if __name__ == '__main__':
    
    _kersim = sys.argv[1]
    #_kersim = 'ker'
    

    if _kersim == 'ker':

	# read in time window information
	
	# run cpt_ker
	_tmp = os.path.expanduser('~/kernels_f90/sim_ker/tmp/')
	_tmpdir = '~/kernels_f90/sim_ker/tmp/'
	infiles = [_tmpdir+'input_file',
	           _tmpdir+'mesh_grid',
		   _tmpdir+'GSDF_files',]
	outfiles = ['ker_files','mesh_grid','debug','viz']
	_ss = ss
	_ssc = ssc
	_DI = ['E','N','Z']
	_flist = glob( _tmp+'GSDF_files/isf_file/'+string.join(( _ss,str(dx[0]),_DI[cmpid-1]),'-')+'-*-isf.G' )   # get t1 and t2
	
	if simtest == 'cvm_test': 
	    del( lons, lats, dep, dtype, sta, lonr, latr, srlon, srlat, projection_ne,cvmrun, runname )
	del( vm, mm, T, L, simtest, stanam, srcnam, vol)
	del( xs, ys, zs, xr, yr, zr, xgr, ygr,zgr, xgs,ygs,zgs )
	del( ssr, ss,ssc,cmpid,fd)
	
	for _fpnam in _flist:
	    _fnam = _fpnam.split('/')[-1]
	    print _fnam
	    _fls = _fnam.split('-')
	    _ssct1t2 = string.join((_ssc,_fls[3],_fls[4]),'-')
	    rundir = '~/run/kernels/'+_ssct1t2
	    print rundir
	    
	    sord.run_ker(locals())
	
    else:
	# run four forward simulations
	# _kersim = 'sim'

	import sord.extras.source as source_util
	
	print xs,ys,zs
	print xr,yr,zr

	_simcase = sys.argv[2]
	print _simcase
	_simst = sys.argv[3]  # source type
	print _simst
	_simstf = sys.argv[4] # source time function type ( 'hanning' or 'Gaussian' )
	print _simstf

	# source parameters
	if _simcase == 'forward':
	    
	    # moment source
	    source = 'moment'
	    timefunction = 'none' # read stf from file or generate below
	    nsource = 1
	    infiles = ['tmp/src_*']
	    rundir = '~/run/kernels/'+ssr+'/forward-' + vm
	   
	    if _simst == 'expl':
		# explosive source'expl'
		w1 = 1.0e17, 1.0e17, 1.0e17
		w2 = 0.0, 0.0, 0.0
	    else:
		# double couple 'doub'
		if srcnam=='14383980':  # Chino Hills EQ
		    # moment tensor ( use moment tensor in global coord )
		    # w1 = Mtt, Mpp, Mrr    # m11, m22, m33
		    # w2 = Mpr, Mtr, Mtp    # m23, m13, m12
                    print 'src is: ', srcnam
		    
		    if 1:
			print 'SCSN source'
			w1 = -1450e14, 602e14, 844e14
			w2 = 689e14, -198e14, -495e14
			
			#print 'harvard source'
			# Harvard CMT
			#_M = 1.96e17   # ( scalar moment in N*m )
			#w1 = _M * numpy.array( [-1.870, 0.952, 0.918] )
			#w2 = _M * numpy.array( [0.840, -0.493, -0.513] )
		    else:
			print 'SLU source'
			# SLU solution
			w1 = -1.20e17, 0.419e17, 0.778e17
			w2 = 0.601e17, -0.200e17, -0.389e17

		    # This rotation is for the case that the initial moment tensor is in (rtp)
		    # rotate source(very important)
		    # from rtp(spherical, t: theta, or colatitude) to ENZ
		    # since the logical coord is in ENZ(123)
		    if 0:
			_rot = (0,1,0),(-1,0,0),(0,0,1)
		        w1,w2 = sord.coord.rot_sym_tensor(w1,w2,_rot)
		    else:
			mtt,mpp,mrr = w1[0],w1[1],w1[2]
			mpr,mtr,mtp = w2[0],w2[1],w2[2]
			w1 = mpp,mtt,mrr
			w2 = -mtr,mpr,-mtp
			del(mtt,mpp,mrr,mtp,mpr,mtr)
			
		    _rot = numpy.eye(3)
		    _rot[:2,:2] = sord.coord.rotation( lons, lats, projection_ne )[0]
		    w1, w2 = sord.coord.rot_sym_tensor( w1, w2, _rot )
		else:
		    # arbitrary double couple source
		    import utils.seism 
		    #strike = 291; dip = 59.; rake = 142.
		    strike = 45.0; dip = 45.; rake = 45.
		    Mt = utils.seism.focal_m( strike, dip, rake )
		    M = Mt*1.0e17
		    w1 = M[0],M[1],M[2]
		    w2 = M[3],M[4],M[5]
		    del( strike, dip, rake, Mt, M )
		    
	    source1 = w1[0], w1[1], w1[2]
	    source2 = w2[0], w2[1], w2[2]
	    
	    del( w1, w2 )
	    
	    # source grid coord
	    _xs, _ys, _zs = xs, ys, zs
	    _xr, _yr, _zr = xr, yr, zr
	    
	    # source is added on stress( cell variable )
	    ihypo = (
		    _xs / dx[0] + 1.0,   
		    _ys / dx[1] + 1.0,
		    _zs / dx[2] + 1.0,
		    )

	    # receiver grid coord
	    _xhypo = (
		     _xr / dx[0] + 1.0, 
		     _yr / dx[1] + 1.0, 
		     _zr / dx[2] + 1.0,
		     )
	    
	    _ista = [ _xhypo[0], _xhypo[1], _xhypo[2], () ]    # whole time history
	    
	    # time function
	    if _simstf == 'Gaussian':
		
		S = 0.25
		a = 0.5/S**2; b = 8.*S
		t = dt * numpy.arange( nt )
		history = numpy.exp( -a * (t - b/2.)**2 )
		t0 = 0.0 
		source_util.src_write( history, nt, dt, t0, ihypo, source1, source2, source, 'tmp' )
		del( history, S, a, b, t, t0 )
		
	    else:    # _simstf == 'Hanning'
		# hanning
		nh = 69
		history = numpy.r_[ numpy.hanning(nh), numpy.zeros(nt-nh)]
		t0 = 0.0 
		source_util.src_write( history, nt, dt, t0, ihypo, source1, source2, source, 'tmp' )
		del( history, t0, nh )
	    
	    # forward wavefield output
	    fieldio += [
		
		# for green's function reciprocity
		( '=wi', 'u1',  _ista, ss+'.u1' ),    # ux displacement at receiver
		( '=wi', 'u2',  _ista, ss+'.u2' ),    # uy displacement at receiver
		( '=wi', 'u3',  _ista, ss+'.u3' ),    # uz displacement at receiver
		
		# output velocity and density
		( '=w', 'vp', [], 'vp'),   
		( '=w', 'vs', [], 'vs'),
		( '=w', 'rho',[], 'rho'),
		
		# for kernel computation
		( '=w', 'e11', vol, 'Ve_11' ),	    # du: strain tensor
		( '=w', 'e22', vol, 'Ve_22' ),	    # du: strain tensor
		( '=w', 'e33', vol, 'Ve_33' ),	    # du: strain tensor
		( '=w', 'e23', vol, 'Ve_23' ),	    # du: strain tensor
		( '=w', 'e31', vol, 'Ve_31' ),	    # du: strain tensor
		( '=w', 'e12', vol, 'Ve_12' ),      # du: strain tensor
	    ]

	    if simtest == 'cvm_test': 
		del( lons, lats, dep, dtype, sta, lonr, latr, srlon, srlat, projection_ne,cvmrun,runname )
	    del( vm, mm, T, L, simtest, stanam, srcnam, vol)
	    del( xs, ys, zs, xr, yr, zr, xgr, ygr,zgr, xgs,ygs,zgs )
	    del( ssr, ss,ssc,cmpid,fd)
	else :

	    if _simcase == 'sgt1':
		_fd = 1
	    elif _simcase == 'sgt2':
		_fd = 2
	    else:
		_fd = 3

	    # single force
	    source = 'force'
	    timefunction = 'none' # read stf from file or generate below
	    nsource = 1   # number of finite source
	    infiles = ['tmp/src_*']
	    rundir = '~/run/kernels/'+ssr+'/'+vm+'-sgt_ij' + str(_fd)
	    
	    # force vector
	    source1 = [0.0, 0.0, 0.0]
	    source1[_fd-1] = 1.0
	    source2 = 0.0, 0.0, 0.0  # for single force

	    # source grid coord
	    _xs, _ys, _zs = xr, yr, zr
	    _xr, _yr, _zr = xs, ys, zs
	    
	    # source is added on stress( cell variable )
	    ihypo = (
		    _xs / dx[0] + 1.0,   
		    _ys / dx[1] + 1.0,
		    _zs / dx[2] + 1.0,
		    )

	    # receiver grid coord
	    _xhypo = (
		     _xr / dx[0] + 1.0, 
		     _yr / dx[1] + 1.0, 
		     _zr / dx[2] + 1.0,
		     )
	    
	    _ista = [ _xhypo[0], _xhypo[1], _xhypo[2], () ]    # whole time history
	    
	    # time function
	    if _simstf == 'Gaussian':
		
		S = 0.25
		a = 0.5/S**2; b = 8.*S
		t = dt * numpy.arange( nt )
		history = numpy.exp( -a * (t - b/2.)**2 )
		t0 = 0.0 
		source_util.src_write( history, nt, dt, t0, ihypo, source1, source2, source, 'tmp' )
		del( history, S, a, b, t, t0 )
		
	    else:    # _simstf == 'hanning'
		# hanning
		nh = 69
		history = numpy.r_[ numpy.hanning(nh), numpy.zeros(nt-nh)]
		t0 = 0.0 
		source_util.src_write( history, nt, dt, t0, ihypo, source1, source2, source, 'tmp' )
		del( history, t0, nh )

	    # RGT and SGT field output
	    fieldio += [
	    
	    # attention_index of RGT(second order tensor) and SGT(three order tensor)
	    # RGT_in means, n direction force cause ith component displacement
	    # SGT_ijn means, RGT_in,j, ',j' is the x_j derivative
		
		#( '=wi', 'u1', _ista, 'rgt_' + '1' + str( _fd ) ),    # Green's function
		#( '=wi', 'u2', _ista, 'rgt_' + '2' + str( _fd ) ),    # Green's function
		#( '=wi', 'u3', _ista, 'rgt_' + '3' + str( _fd ) ),    # Green's function
		
		# for green's reciprocity
		#( '=wi', 'e11', _ista, 'sgt_11'+ str(_fd) ),       # dG: Green's function
		#( '=wi', 'e22', _ista, 'sgt_22'+ str(_fd) ),       # dG: Green's function
		#( '=wi', 'e33', _ista, 'sgt_33'+ str(_fd) ),       # dG: Green's function
		#( '=wi', 'e23', _ista, 'sgt_23'+ str(_fd) ),       # dG: Green's function
		#( '=wi', 'e23', _ista, 'sgt_32'+ str(_fd) ),       # dG: Green's function
		#( '=wi', 'e31', _ista, 'sgt_13'+ str(_fd) ),       # dG: Green's function
		#( '=wi', 'e31', _ista, 'sgt_31'+ str(_fd) ),       # dG: Green's function
		#( '=wi', 'e12', _ista, 'sgt_12'+ str(_fd) ),       # dG: Green's function
		#( '=wi', 'e12', _ista, 'sgt_21'+ str(_fd) ),       # dG: Green's function
		
		# for kernel computation 
		( '=w', 'e11', vol, 'Vsgt_11'+ str(_fd) ),       # dG: Green's function
		( '=w', 'e22', vol, 'Vsgt_22'+ str(_fd) ),       # dG: Green's function
		( '=w', 'e33', vol, 'Vsgt_33'+ str(_fd) ),       # dG: Green's function
		( '=w', 'e23', vol, 'Vsgt_23'+ str(_fd) ),       # dg: green's function
		( '=w', 'e31', vol, 'Vsgt_31'+ str(_fd) ),       # dG: Green's function
		( '=w', 'e12', vol, 'Vsgt_12'+ str(_fd) ),       # dG: Green's function
	    ]
	    

	    if simtest == 'cvm_test': 
		del( lons, lats, dep, dtype, sta, lonr, latr, srlon, srlat, projection_ne, cvmrun,runname )
	
	    del( vm, mm, T, L, simtest, stanam, srcnam, vol )
	    del( xs, ys, zs, xr, yr, zr, xgs,ygs,zgs, xgr,ygr,zgr )
	    del( ssr, ss,ssc,cmpid,fd )
	sord.run( locals() )


