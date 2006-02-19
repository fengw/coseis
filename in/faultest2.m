% Fault test

  debug = 0;
  faultnormal = 3;
  dx  = 300;
  dt  = .024;
  vp  = 6000.;
  vs  = 3464.;
  rho = 2670.;
  vrup = -1.;
  dc  = 0.4;
  mus = .677;
  mud = .525;
  tn  = -120e6;
  td  = 0.;
  th  = -70e6;
  th  = { -81.6e6 'zone' 3 3 0  6 6 0 };
  npml = 2;
  nt  = 2;
  ihypo = [ 4 4 4 ];
  bc1   = [ 0 0 0 ];
  bc2   = [ 0 0 0 ];

  nn    = [ 8 8 4 ];
  np    = [ 1 1 2 ];

  out = { 'x'    1      1 1 1   0 0 0 };
  out = { 'a'    1      1 1 1   0 0 0 };
  out = { 'v'    1      1 1 1   0 0 0 };
  out = { 'u'    1      1 1 1   0 0 0 };
  out = { 'w'    1      1 1 1   0 0 0 };

  out = { 'am'   1      1 1 1   0 0 0 };
  out = { 'vm'   1      1 1 1   0 0 0 };
  out = { 'um'   1      1 1 1   0 0 0 };
  out = { 'wm'   1      1 1 1   0 0 0 };

  out = { 'mr'   1      1 1 1   0 0 0 };
  out = { 'mu'   1      1 1 1   0 0 0 };
  out = { 'lam'  1      1 1 1   0 0 0 };
  out = { 'y'    1      1 1 1   0 0 0 };

  out = { 't0'   1      1 1 1   0 0 0 };
  out = { 't3'   1      1 1 1   0 0 0 };
  out = { 'mus'  1      1 1 1   0 0 0 };
  out = { 'mud'  1      1 1 1   0 0 0 };
  out = { 'dc'   1      1 1 1   0 0 0 };
  out = { 'co'   1      1 1 1   0 0 0 };
  out = { 'sv'   1      1 1 1   0 0 0 };
  out = { 'sl'   1      1 1 1   0 0 0 };
  out = { 'trup' 1      1 1 1   0 0 0 };
  out = { 'tarr' 1      1 1 1   0 0 0 };
  out = { 'tn'   1      1 1 1   0 0 0 };
  out = { 'ts'   1      1 1 1   0 0 0 };
