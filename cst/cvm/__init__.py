#!/usr/bin/env python
"""
SCEC Community Velocity Model
"""
import os, sys, re, shutil, urllib, tarfile
import numpy as np
from ..conf import launch

path = os.path.dirname( os.path.realpath( __file__ ) )

input_template = """\
%(nsample)s
%(lon_file)s
%(lat_file)s
%(dep_file)s
%(rho_file)s
%(vp_file)s
%(vs_file)s
"""

def _build( mode=None, optimize=None ):
    """
    Build CVM code.
    """
    import cst

    # configure
    cf = cst.conf.configure( 'cvm' )[0]
    if not optimize:
        optimize = cf.optimize
    if not mode:
        mode = cf.mode
    if not mode:
        mode = 'asm'

    # download source code
    url = 'http://www.data.scec.org/3Dvelocity/Version4.tar.gz'
    url = 'http://earth.usc.edu/~gely/coseis/download/cvm4.tgz'
    tarball = os.path.join( cf.repo, os.path.basename( url ) )
    if not os.path.exists( tarball ):
        if not os.path.exists( cf.repo ):
            os.makedirs( cf.repo )
        print( 'Downloading %s' % url )
        urllib.urlretrieve( url, tarball )

    # build directory
    cwd = os.getcwd()
    os.chdir( path )
    if not os.path.exists( 'build' ):
        os.makedirs( 'build' )
        fh = tarfile.open( tarball, 'r:gz' )
        fh.extractall( 'build' )
        if os.system( 'patch -p0 < cvm4.patch' ):
            sys.exit( 'Error patching CVM' )
    os.chdir( 'build' )

    # compile ascii, binary, and MPI versions
    new = False
    if 'a' in mode:
        source = 'iotxt.f', 'version4.0.f'
        for opt in optimize:
            compiler = cf.fortran_serial, cf.fortran_flags[opt], '-o'
            object_ = 'cvm4-a' + opt
            new |= cst.conf.make( compiler, object_, source )
    if 's' in mode:
        source = 'iobin.f', 'version4.0.f'
        for opt in optimize:
            compiler = cf.fortran_serial, cf.fortran_flags[opt], '-o'
            object_ = 'cvm4-s' + opt
            new |= cst.conf.make( compiler, object_, source )
    if 'm' in mode and cf.fortran_mpi:
        source = 'iompi.f', 'version4.0.f'
        for opt in optimize:
            object_ = 'cvm4-m' + opt
            compiler = cf.fortran_mpi, cf.fortran_flags[opt], '-o'
            new |= cst.conf.make( compiler, object_, source )
    os.chdir( cwd )
    return

def stage( inputs={}, **kwargs ):
    """
    Stage job
    """
    import cst

    print( 'CVM setup' )

    # update inputs
    inputs = inputs.copy()
    inputs.update( kwargs )

    # configure
    job, inputs = cst.conf.configure( 'cvm', **inputs )
    if inputs:
        sys.exit( 'Unknown parameter: %s' % inputs )
    if not job.mode:
        job.mode = 's'
        if job.nproc > 1:
            job.mode = 'm'
    job.command = os.path.join( '.', 'cvm4' + '-' + job.mode + job.optimize )
    job = cst.conf.prepare( job )

    # build
    if not job.prepare:
        return job
    _build( job.mode, job.optimize )

    # check minimum processors needed for compiled memory size
    file = os.path.join( path, 'build', 'in.h' )
    string = open( file ).read()
    pattern = 'ibig *= *([0-9]*)'
    n = int( re.search( pattern, string ).groups()[0] )
    minproc = int( job.nsample / n )
    if job.nsample % n != 0:
        minproc += 1
    if minproc > job.nproc:
        sys.exit( 'Need at lease %s processors for this mesh size' % minproc )

    # create run directory
    if job.force == True and os.path.isdir( job.rundir ):
        shutil.rmtree( job.rundir )
    if not os.path.exists( job.rundir ):
        f = os.path.join( path, 'build' )
        shutil.copytree( f, job.rundir )
    else:
        for f in (
            job.lon_file, job.lat_file, job.dep_file,
            job.rho_file, job.vp_file, job.vs_file
        ):
            ff = os.path.join( job.rundir, f )
            if os.path.exists( ff ):
                os.remove( ff )

    # process machine templates
    cst.conf.skeleton( job, new=False )

    # save input file and configuration
    f = os.path.join( job.rundir, 'cvm-input' )
    open( f, 'w' ).write( input_template % job.__dict__ )
    f = os.path.join( job.rundir, 'conf.py' )
    cst.util.save( f, job.__dict__ )
    return job

def extract( lon, lat, dep, prop=None, **kwargs ):
    """
    Simple CVM extraction

    Parameters
    ----------
        lon, lat, dep: Coordinate arrays
        prop: 'rho', 'vp', or 'vs'. None=all
        nproc: Optional, number of processes
        rundir: Optional, job staging directory

    Returns
    -------
        rho, vp, vs: Material arrays
    """
    lon = np.asarray( lon, 'f' )
    lat = np.asarray( lat, 'f' )
    dep = np.asarray( dep, 'f' )
    shape = dep.shape
    job = stage( nsample=dep.size, **kwargs )
    path = job.rundir + os.sep
    lon.tofile( path + 'lon' )
    lat.tofile( path + 'lat' )
    dep.tofile( path + 'dep' )
    del( lon, lat, dep )
    launch( job, run='exec' )
    if prop is not None:
        f = {'rho': job.rho_file, 'vp': job.vp_file, 'vs': job.vs_file}
        f = np.fromfile( path + f[prop], 'f' ).reshape( shape )
        return f
    else:
        rho = np.fromfile( path + job.rho_file, 'f' ).reshape( shape )
        vp =  np.fromfile( path + job.vp_file,  'f' ).reshape( shape )
        vs =  np.fromfile( path + job.vs_file,  'f' ).reshape( shape )
        return rho, vp, vs

