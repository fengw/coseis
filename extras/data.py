#!/usr/bin/env python
"""
Mapping data utilities
"""
import os
import numpy as np

def tsurf( path ):
    """
    Read GOCAD (http://www.gocad.org) trigulated surface "Tsurf" files.
    """
    fh = open( path )
    tsurf = []
    for line in fh.readlines():
        f = line.split()
        if line.startswith( 'GOCAD TSurf' ):
            tface, vrtx, trgl, border, bstone, name, color = [], [], [], [], [], None, None
        elif f[0] in ('VRTX', 'PVRTX'):
            vrtx += [[float(f[2]), float(f[3]), float(f[4])]]
        elif f[0] in ('ATOM', 'PATOM'):
            i = int( f[2] ) - 1
            vrtx += [ vrtx[i] ]
        elif f[0] == 'TRGL':
            trgl += [[int(f[1]) - 1, int(f[2]) - 1, int(f[3]) - 1]]
        elif f[0] == 'BORDER':
            border += [[int(f[2]) - 1, int(f[3]) - 1]]
        elif f[0] == 'BSTONE':
            bstone += [int(f[1]) - 1]
        elif f[0] == 'TFACE':
            if trgl != []:
                tface += [ np.array( trgl, 'i' ).T ]
            trgl = []
        elif f[0] == 'END':
            vrtx   = np.array( vrtx, 'f' ).T
            border = np.array( border, 'i' ).T
            bstone = np.array( bstone, 'i' ).T
            tface += [ np.array( trgl, 'i' ).T ]
            tsurf += [[vrtx, tface, border, bstone, name, color]]
        elif line.startswith( 'name:' ):
            name = line.split( ':', 1 )[1].strip()
        elif line.startswith( '*solid*color:' ):
            f = line.split( ':' )[1].split()
            color = float(f[0]), float(f[1]), float(f[2])
    return tsurf

def etopo1( indices=None, downsample=1, path='mapdata', download=True ):
    """
    Download ETOPO1 Global Relief Model.
    http://www.ngdc.noaa.gov/mgg/global/global.html
    """
    import urllib, zipfile, sord
    filename = os.path.join( path, 'etopo%02d-ice.f32' % downsample )
    if download and not os.path.exists( filename ):
        if path != '' and not os.path.exists( path ):
            os.makedirs( path )
        url = 'ftp://ftp.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/binary/etopo1_ice_g_i2.zip'
        f = os.path.join( path, os.path.basename( url ) )
        if not os.path.exists( f ):
            print( 'Retrieving %s' % url, f )
            urllib.urlretrieve( url, f )
        z = zipfile.ZipFile( f, 'r' ).read( 'etopo1_ice_g_i2.bin' )
        z = np.fromstring( z, '<i2' ).reshape( [21601, 10801] )
        z = np.array( z, 'f' )
        if downsample > 1:
            z = sord.coord.downsample_sphere( z, downsample )
        open( filename, 'wb' ).write( z )
    if indices != None:
        shape = (21601 - 1) / downsample + 1, (10801 - 1) / downsample + 1
        return sord.util.ndread( filename, shape, indices, 'f' )
    else:
        return

def globe( indices=None, path='mapdata', download=True ):
    """
    Global Land One-km Base Elevation Digital Elevation Model.
    http://www.ngdc.noaa.gov/mgg/topo/globe.html
    """
    import urllib, gzip, sord
    filename = os.path.join( path, 'globe30.i16' )
    if download and not os.path.exists( filename ):
        if path != '' and not os.path.exists( path ):
            os.makedirs( path )
        print( 'Building %s' % filename )
        n = 90 * 60 * 2
        url = 'http://www.ngdc.noaa.gov/mgg/topo/DATATILES/elev/%s10g.gz'
        tiles = 'abcd', 'efgh', 'ijkl', 'mnop'
        fd = open( path, 'wb' )
        for j in range( len( tiles ) ):
            row = []
            for k in range( len( tiles[j] ) ):
                u = url % tiles[j][k]
                f = os.path.join( path, os.path.basename( u ) )
                if not os.path.exists( f ):
                    print( 'Retrieving %s' % u )
                    urllib.urlretrieve( u, f )
                z = gzip.open( f, mode='rb' ).read()
                z = np.fromstring( z, '<i2' ).reshape( [-1, n] )
                row += [z]
            row = np.hstack( row )
            row.tofile( fd )
        fd.close()
        del( z, row )
    if indices != None:
        shape = 43200, 21600
        return sord.util.ndread( filename, shape, indices, '<i2' )
    else:
        return

def topo( extent, scale=1.0, cache='', path='mapdata', download=True ):
    """
    Extrat merged GLOBE/ETOPO1 digital elvation model for given region.
    """
    if cache and os.path.exists( cache + '.npz' ):
        c = np.load( cache + '.npz' )
        return c['z'], c['lon'], c['lat']
    o = 0.25
    lon, lat = extent
    j = int( lon[0] * 60 + 10801 - o ), int( np.ceil( lon[1] * 60 + 10801 + o ) )
    k = int( -lat[1] * 60 + 5401 - o ), int( np.ceil( -lat[0] * 60 + 5401 + o ) )
    z = etopo1( [j, k], 1, path, download )
    j = 2 * j[0] - 1, 2 * j[1] - 2
    k = 2 * k[0] - 1, 2 * k[1] - 2
    n = j[1] - j[0] + 1, k[1] - k[0] + 1
    z *= 0.0625
    z1 = np.empty( n, z.dtype )
    z1[0::2,0::2] = 9 * z[:-1,:-1] + 3 * z[:-1,1:] + 3 * z[1:,:-1] +     z[1:,1:]
    z1[0::2,1::2] = 3 * z[:-1,:-1] + 9 * z[:-1,1:] +     z[1:,:-1] + 3 * z[1:,1:]
    z1[1::2,0::2] = 3 * z[:-1,:-1] +     z[:-1,1:] + 9 * z[1:,:-1] + 3 * z[1:,1:]
    z1[1::2,1::2] =     z[:-1,:-1] + 3 * z[:-1,1:] + 3 * z[1:,:-1] + 9 * z[1:,1:]
    z = globe( [j, k], path, download )
    i = z != -500
    z1[i] = z[i]
    z = z1
    z *= scale
    lon = (j[0] - 21600.5) / 120, (j[1] - 21600.5) / 120
    lat = (10800.5 - k[1]) / 120, (10800.5 - k[0]) / 120
    if cache:
        np.savez( cache + '.npz', z=z, lon=lon, lat=lat )
    return z[:,::-1], (lon, lat)

def mapdata( kind='coastlines', resolution='high', extent=None, min_area=0.0, min_level=0, max_level=4, clip=1, path='mapdata', download=True ):
    """
    Reader for the Global Self-consistent, Hierarchical, High-resolution Shoreline
    database (GSHHS) by Wessel and Smith.  WGS-84 ellipsoid.

    kind: 'coastlines', 'rivers', 'borders'
    resolution: 'crude', 'low', 'intermediate', 'high', 'full'
    extent: (min_lon, max_lon), (min_lat, max_lat)

    Reference:
    Wessel, P., and W. H. F. Smith, A Global Self-consistent, Hierarchical,
    High-resolution Shoreline Database, J. Geophys. Res., 101, 8741-8743, 1996.
    http://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html
    http://www.soest.hawaii.edu/wessel/gshhs/index.html
    """
    nh = 11
    url = 'http://www.ngdc.noaa.gov/mgg/shorelines/data/gshhs/version2.0/gshhs_2.0.zip'
    filename = os.path.join( path, os.path.basename( url ) )
    kind = dict(c='gshhs', r='wdb_rivers', b='wdb_borders')[kind[0]]
    member = 'gshhs/%s_%s.b' % (kind, resolution[0])
    if kind != 'gshhs':
        min_area = 0.0
    if extent != None:
        lon, lat = extent
        lon = lon[0] % 360, lon[1] % 360
        extent = lon, lat
    if download and not os.path.exists( filename ):
        print( 'Downloading %s' % url )
        import urllib
        if path != '' and not os.path.exists( path ):
            os.makedirs( path )
        urllib.urlretrieve( url, filename )
    import zipfile
    data = np.fromstring( zipfile.ZipFile( filename ).read( member ), '>i' )
    xx = []
    yy = []
    ii = 0
    nkeep = 0
    ntotal = 0
    while ii < data.size:
        ntotal += 1
        hdr = data[ii:ii+nh]
        n = hdr[1]
        ii += nh + 2 * n
        level = hdr[2:3].view( 'i1' )[3]
        if level > max_level:
            break
        if level < min_level:
            continue
        area = hdr[7] * 0.1
        if area < min_area:
            continue
        if extent != None:
            west, east, south, north = hdr[3:7] * 1e-6
            west, east, south, north = hdr[3:7] * 1e-6
            if east < lon[0] or west > lon[1] or north < lat[0] or south > lat[1]:
                continue
        nkeep += 1
        x, y = 1e-6 * np.array( data[ii-2*n:ii].reshape(n, 2).T, 'f' )
        if extent != None and clip != 0:
            loose = clip > 0
            x, y = clipdata( x, y, extent, loose )[:2]
        xx += [ x, [np.nan] ]
        yy += [ y, [np.nan] ]
    print '%s, resolution: %s, selected %s of %s' % (member, resolution, nkeep, ntotal)
    if nkeep:
        xx = np.concatenate( xx )[:-1]
        yy = np.concatenate( yy )[:-1]
    return np.array( [xx, yy], 'f' )

def clipdata( x, y, extent, loose=True ):
    xlim, ylim = extent
    i = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    if loose:
        i[:-1] = i[:-1] | i[1:]
        i[1:] = i[:-1] | i[1:]
    x[~i] = np.nan
    y[~i] = np.nan
    i[1:] = i[:-1] | i[1:]
    x = x[i]
    y = y[i]
    return x, y, i

def engdahlcat( path='engdahl-centennial-cat.f32', fields=['lon', 'lat', 'depth', 'mag'] ):
    """
    Engdahl Centennial Earthquake Catalog to binary file.
    http://earthquake.usgs.gov/research/data/centennial.php
    """
    import urllib
    if not os.path.exists( path ):
        fmt = [
            6, ('icat',   'S6'),
            1, ('asol',   'S1'),
            5, ('isol',   'S5'),
            4, ('year',   'i4'),
            3, ('month',  'i4'),
            3, ('day',    'i4'),
            4, ('hour',   'i4'),
            3, ('minute', 'i4'),
            6, ('second', 'f4'),
            9, ('lat',    'f4'),
            8, ('lon',    'f4'),
            6, ('depth',  'f4'),
            4, ('greg',   'i4'),
            4, ('ntel',   'i4'),
            4, ('mag',    'f4'),
            3, ('msc',    'S3'),
            6, ('mdo',    'S6'),
        ]
        url = 'http://earthquake.usgs.gov/research/data/centennial.cat'
        url = urllib.urlopen( url )
        data = np.genfromtxt( url, dtype=fmt[1::2], delimiter=fmt[0::2] )
        out = []
        for f in fields:
            out += [data[:][f]]
        np.array( out, 'f' ).T.tofile( path )
    else:
        out = np.fromfile( path, 'f' ).reshape( (-1,4) ).T
    return out

def upsample( f ):
    n = list( f.shape )
    n[:2] = [ n[0] * 2 - 1, n[1] * 2 - 1 ]
    g = np.empty( n, f.dtype )
    g[0::2,0::2] = f
    g[0::2,1::2] = 0.5 * (f[:,:-1] + f[:,1:])
    g[1::2,0::2] = 0.5 * (f[:-1,:] + f[1:,:])
    g[1::2,1::2] = 0.25 * (f[:-1,:-1] + f[1:,1:] + f[:-1,1:] + f[1:,:-1])
    return g

def downsample_sphere( f, d ):
    """
    Down-sample node-registered spherical surface with averaging.

    The indices of the 2D array f are, respectively, longitude and latitude.
    d is the decimation interval which should be odd to preserve nodal
    registration.
    """
    n = f.shape
    ii = np.arange( d ) - (d - 1) / 2
    jj = np.arange( 0, n[0], d )
    kk = np.arange( 0, n[1], d )
    nn = jj.size, kk.size
    ff = np.zeros( nn, f.dtype )
    jj, kk = np.ix_( jj, kk )
    for dk in ii:
        k = n[1] - 1 - abs( n[1] - 1 - abs( dk + kk ) )
        for dj in ii:
            j = (jj + dj) % n[0]
            ff = ff + f[j,k]
    ff[:,0] = ff[:,0].mean()
    ff[:,-1] = ff[:,-1].mean()
    ff *= 1.0 / (d * d)
    return ff
