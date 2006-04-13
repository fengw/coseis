% Benchmark

  faultnormal = 1;
  vrup = -1.;
  dc  = 0.4;
  mud = .525;
  mus = .677;
  tn  = -120e6;
  td  = 0.;
  th  = -81.6e6;
  affine = [ 1. 1. 0.; 0. 1. 0.; 0. 0. 1. ] / 1.;
  nt = 10;

  nn = [ 512 512 512 ];
  np = [ 4 4 4 ];
  np = [ 1 8 8 ];
  np = [ 1 1 64 ];

  nn	= [ 256 256 256 ];
  np	=			[ 2 2 2 ];
  np = [ 1 2 4 ];
  np = [ 1 1 8 ];

  nn = [ 1024 1024 1024 ];
  np = [ 8 8 8 ];
  np = [ 1 16 32 ];
  np = [ 1 1 512 ];

  nn = [ 128 128 128 ];
  np = [ 1 1 4 ];

