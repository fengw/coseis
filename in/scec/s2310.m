% TPV3
  np       = [    1   1  16 ];
  np       = [    1   1  32 ];
  nn       = [  601 200  64 ];
  n1expand = [   40   0   0 ];
  n2expand = [   40   0   0 ];
  bc1      = [    0  10  10 ];
  bc2      = [    0  10  -2 ];
  ihypo    = [  276 100  -2 ];
  fixhypo  =     -2;
  xhypo    = [  50.  0.  0. ];
  affine   = [ 1. 1. 1.   0. 1. 0.   0. 0. 1. ];
  nt  = 1500;
  dx  = 100;
  dt  = .008;

  rho = 2670.;
  vp  = 6000.;
  vs  = 3464.;
  gam = .2;
  gam = { .02 'cube' -15001. -7501. -4000.   15001. 7501. 4000. };
  hourglass = [ 1. 2. ];

  faultnormal = 3;
  vrup = -1.;
  dc  = 0.4;
  mud = .525;
  mus = 10000.;
  mus = { .677   'cube' -15001. -7501. -1.  15001. 7501. 1. };
  tn  = -120e6;
  ts1 = 70e6;
  ts1 = { 81.6e6 'cube'  -1501. -1501. -1.   1501. 1501. 1. };

  out = { 'x'     1   1  1 -2  0   -1 -1 -2  0 };
  out = { 'pv2'   1   1  1 -2 -1   -1 -1 -2 -1 };
  out = { 'vm2' 100   1  1 -2  0   -1 -1 -2 -1 };
  out = { 'vm2'   5   1  0 -2  0   -1  0 -2 -1 };
  out = { 'vm2'   5   0  1 -2  0    0 -1 -2 -1 };
  out = { 'tsm'   1   1  1  0  0   -1 -1  0  0 };
  out = { 'su'    1   1  1  0 -1   -1 -1  0 -1 };
  out = { 'psv'   1   1  1  0 -1   -1 -1  0 -1 };
  out = { 'trup'  1   1  1  0 -1   -1 -1  0 -1 };
  timeseries = { 'su' -7499.    -1. 0. };
  timeseries = { 'sv' -7499.    -1. 0. };
  timeseries = { 'ts' -7499.    -1. 0. };
  timeseries = { 'su'    -1. -5999. 0. };
  timeseries = { 'sv'    -1. -5999. 0. };
  timeseries = { 'ts'    -1. -5999. 0. };
  timeseries = { 'su'  7499.     1. 0. };
  timeseries = { 'sv'  7499.     1. 0. };
  timeseries = { 'ts'  7499.     1. 0. };
  timeseries = { 'su'     1.  5999. 0. };
  timeseries = { 'sv'     1.  5999. 0. };
  timeseries = { 'ts'     1.  5999. 0. };

