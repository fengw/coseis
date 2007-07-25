% TPV3
  np       = [   1   1  16 ];
  np       = [   1   1   4 ];
  nn       = [ 351 201 102 ];
  ihypo    = [  -2  -2  -2 ];
  n1expand = [   0   0   0 ];
  n2expand = [   0   0   0 ];
  bc1      = [  10  10  10 ];
  bc2      = [  -2   2  -2 ];
  fixhypo  =    -2;
  affine   = [ 1. 0. 0.   0. 1. 0.   0. 0. 1. ];
  nt  = 3000;
  dx  = 50;
  dt  = .004;

  vp  = 6000.;
  vs  = 3464.;
  rho = 2670.;
  gam = .02;
  hourglass = [ 1. 2. ];

  faultnormal = 3;
  vrup = -1.;
  dc  = 0.4;
  mud = .525;
  mus = 10000.;
  mus = { .677    'cube' -15001. -7501. -1.  15001. 7501. 1. };
  tn  = -120e6;
  ts1 = 70e6;
  ts1 = { 81.6e6 'cube'  -1501. -1501. -1.   1501. 1501. 1. };

  out = { 'x'    1   1 1 -2  0   -1 -1 -2  0 };
  out = { 'pv2'  1   1 1 -2 -1   -1 -1 -2 -1 };
  out = { 'su'   1   1 1  0 -1   -1 -1  0 -1 };
  out = { 'psv'  1   1 1  0 -1   -1 -1  0 -1 };
  out = { 'trup' 1   1 1  0 -1   -1 -1  0 -1 };
  out = { 'vm2'  5   1 0 -2  0   -1  0 -2 -1 };
  out = { 'vm2'  5   0 1 -2  0    0 -1 -2 -1 };
  out = { 'vm2'  5   0 0  1  0    0  0 -2 -1 };
  timeseries = { 'su' -7499.    -1. 0. };
  timeseries = { 'sv' -7499.    -1. 0. };
  timeseries = { 'ts' -7499.    -1. 0. };
  timeseries = { 'su'  7499.    -1. 0. };
  timeseries = { 'sv'  7499.    -1. 0. };
  timeseries = { 'ts'  7499.    -1. 0. };
  timeseries = { 'su'    -1. -5999. 0. };
  timeseries = { 'sv'    -1. -5999. 0. };
  timeseries = { 'ts'    -1. -5999. 0. };
  timeseries = { 'su'    -1.  5999. 0. };
  timeseries = { 'sv'    -1.  5999. 0. };
  timeseries = { 'ts'    -1.  5999. 0. };
