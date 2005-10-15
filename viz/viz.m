% Viz

if init
  init = 0;
  if ~ishandle(1), figure(1), end
  set( 0, 'CurrentFigure', 1 )
  clf
  drawnow
  holdmovie = 0;
  savemovie = 0;
  vizfield = 'v';
  comp = 0;
  isofrac = .5;
  glyphcut = .1;
  glyphexp = 1;
  glyphtype = 1;
  dark = 1;
  colorexp = .5;
  alim = -1;
  vlim = -1;
  ulim = -1;
  wlim = -1;
  xlim = 0;
  uslim = -1;
  vslim = -1;
  tnlim = -1;
  tslim = -1;
  camdist = -1;
  look = 4;
  icursor = ihypo;
  if ifn, islice = ifn; else islice = crdsys(2); end
  if dark, foreground = [ 1 1 1 ]; background = [ 0 0 0 ]; linewidth = 1;
  else     foreground = [ 0 0 0 ]; background = [ 1 1 1 ]; linewidth = 1;
  end
  hhud = [];
  hmsg = [];
  hhelp = [];
  frame = {};
  itpause = nt;
  showframe = 0;
  count = 0;
  keymod = '';
  keypress = 'f1';
  helpon = 0;
  set( 1, ...
    'Color', background, ...
    'KeyPressFcn', 'control', ...
    'WindowButtonDownFcn', 'itstep = 0;', ...
    'DefaultAxesColorOrder', foreground, ...
    'DefaultAxesColor', background, ...
    'DefaultAxesXColor', foreground, ...
    'DefaultAxesYColor', foreground, ...
    'DefaultAxesZColor', foreground, ...
    'DefaultLineColor', foreground, ...
    'DefaultLineLinewidth', linewidth, ...
    'DefaultTextColor', foreground, ...
    'DefaultTextFontSize', 13, ...
    'DefaultTextFontName', 'FixedWidth', ...
    'DefaultTextHitTest', 'off', ...
    'DefaultLineClipping', 'off', ...
    'DefaultTextVerticalAlignment', 'top', ...
    'DefaultTextHorizontalAlignment', 'center', ...
    'DefaultAxesPosition', [ 0 0 1 1 ], ...
    'DefaultAxesVisible', 'off' )
  haxes = axes( 'Position', [ .02 .1 .96 .88 ], 'Tag', 'mainaxes' );
  cameramenu
  cameratoolbar
  cameratoolbar( 'SetMode', 'orbit' )
  cameratoolbar( 'SetCoordSys', 'z' )
  set( 1, ...
    'KeyPressFcn', 'control', ...
    'WindowButtonDownFcn', 'itstep = 0; cameratoolbar(''down'')' )
  return
end

switch plotstyle
case 'hold'
otherwise
  doglyph = 0;
  domesh = 0;
  dosurf = 0;
  doisosurf = 0;
  dooutline = 1;
  volviz = 0;
  switch plotstyle
  case 'outline'
  case 'slice',      dosurf = 1;
  case 'cube',       dosurf = 1; volviz = 1;
  case 'glyphs',     doglyph = 1;
  case 'isosurface', doisosurf = 1; volviz = 1;
  otherwise error 'plot style'
  end
  plotstyle = 'hold';
end

set( 0, 'CurrentFigure', 1 )
if holdmovie
  set( [ frame{:} ], 'Visible', 'off' )
  set( [ frame{:} ], 'HandleVisibility', 'off' )
else
  delete( [ frame{:} ] )
  frame = {};
end
delete( [ hhud hmsg hhelp ] )
hhud = []; hmsg = []; hhelp = [];
colorscale
set( gcf, 'CurrentAxes', haxes(2) )
text( .50, .05, titles( comp + 1 ) );
text( .98, .98, sprintf( '%.3fs', time ), 'Hor', 'right' )
set( gcf, 'CurrentAxes', haxes(1) )

i1volume =  [ 1 1 1 ];
i2volume = -[ 1 1 1 ];
if ifn
  i1volume = [ i1volume; i1volume ];
  i2volume = [ i2volume; i2volume ];
  i1volume(2,ifn) = 0;
  i2volume(1,ifn) = 0;
end
i1slice =  [ 1 1 1 ];
i2slice = -[ 1 1 1 ];
i = islice;
i1slices(i) = icursor(i) - nnoff(i);
i2slices(i) = icursor(i) - nnoff(i) + cellfocus;
if ifn && islice ~= ifn
  i1slice = [ i1slice; i1slice ];
  i2slice = [ i2slice; i2slice ];
  i1slice(2,ifn) = 0;
  i2slice(1,ifn) = 0;
end

if ifn,              faultviz,   end
if doglyph,          glyphviz,   end
if doisosurf,        isosurfviz, end
if domesh || dosurf, surfviz,    end
if dooutline,        outlineviz, end
if look,             lookat,     end

clear xg mg vg xga mga vga
drawnow

kids = get( haxes, 'Children' );
kids = [ kids{1}; kids{2} ]';
frame{end+1} = kids;
showframe = length( frame );
if savemovie && ~holdmovie
  count = count + 1;
  file = sprintf( 'out/viz/%06d', count );
  saveas( gcf, file )
end
