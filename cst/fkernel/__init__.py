#!/usr/bin/env python
"""
Frechet kernel computation
"""
import os, sys, shutil
from ..conf import launch

path = os.path.dirname( os.path.realpath( __file__ ) )

def _build( mode=None, optimize=None ):
    """
    Build code
    """
    import cst
    cf = cst.conf.configure()[0]
    if not optimize:
        optimize = cf.optimize
    mode = cf.mode
    if not mode:
        mode = 'm'
    
    source = 'signal.f90', 'ker_utils.f90', 'cpt_ker.f90'
    new = False
    cwd = os.getcwd()
    src = os.path.join( path, 'src' )
    bld = os.path.join( os.path.dirname( path ), 'build' ) + os.sep
    os.chdir( src )

    if not os.path.isdir( bld ):
        os.mkdir( bld )
    if 'm' in mode and cf.fortran_mpi[0]:
        for opt in optimize:
            object_ = bld + 'cpt_ker-m' + opt
            fflags = cf.fortran_flags['f'], cf.fortran_flags[opt]
            compiler = (cf.fortran_mpi,) + fflags + ('-o',)
            new |= cst.conf.make( compiler, object_, source )
    if new:
        cst._archive()
    os.chdir( cwd )
    return

def stage( inputs={}, **kwargs ):
    """
    Stage job
    """
    import cst

    print( 'Frechet kernel setup' )

    # update inputs
    inputs = inputs.copy()
    inputs.update( kwargs )

    # configure
    job, inputs = cst.conf.configure( **inputs )
    if inputs:
        sys.exit( 'Unknown parameter: %s' % inputs )
    if not job.mode:
        job.mode = 'm'
    if job.mode != 'm':
        sys.exit( 'Must be MPI' )
    
    # partition and resources
    
    # configure options
    job.command = os.path.join( '.', 'cpt_ker-' + job.mode + job.optimize + ' in/input_files' )
    job = cst.conf.prepare( job )  # display the resouces used in the computation

    # build
    if not job.prepare:
        return job
    _build( job.mode, job.optimize )

    # create run directory (need work)
    cst.conf.skeleton( job )
    shutil.copytree( 'tmp', os.path.join( job.rundir, 'in' ) )
    f = os.path.join( job.rundir, 'out' )
    os.mkdir( f )

    return job

