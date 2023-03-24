function GradientDescentDemo(dim,xo,alpha)
% Demonstration of Gradient Descent in 1D and 2D.
%
% GradientDescentDemo(dim,xo,alpha)
%
% dim = 2 for 2D example, 3 for 3D example
% xo = [x1; y1] in 2D, [x1; y1; z1] in 3D
% alpha > 0 is the learning rate
%
%Examples:
%  GradientDescentDemo(2,[-0.6; -0.6], 0.005) % Local Min
%  GradientDescentDemo(2,[-0.5; -0.7], 0.005) % Global Min
%  GradientDescentDemo(2,[-0.17; 1.235], 0.005) % saddle point 1
%  GradientDescentDemo(2,[-0.2; 1.235], 0.005) % saddle point 2
%  GradientDescentDemo(3,[-0.6; -0.6], 0.005)

% clear previous graphics windows and dock figures
% clf; close all;
set(0,'DefaultFigureWindowStyle','docked')
cla
pause
% or, set up graphics window size
% gfx_win_posn_size = [100, 100, 1000, 1000];
% if not docked for screen recording

syms x y z
if dim == 2
  % from https://www.johndcook.com/blog/2017/10/04/no-critical-point-between-two-peaks/
  % f(x,y) = (x-1).^2 + (x.^2.*y-x-1).^2;
  % some other choices to play with
  % f(x,y) = x.*y.*(4*x^2-2*y^2)+3*cos(y-x^2);
  f(x,y) = 3*(1-x)^2*exp(-(x^2) - (y+1)^2) ...
  - 10*(x/5 - x^3 - y^5).*exp(-x^2-y^2) ...    
  - 1/3*exp(-(x+1)^2 - y^2);
  % % add a small ripple onto the previous one...
  % f(x,y) = 3*(1-x)^2*exp(-(x^2) - (y+1)^2) ...
  % - 10*(x/5 - x^3 - y^5).*exp(-x^2-y^2) ...    
  % - 1/3*exp(-(x+1)^2 - y^2) ...
  % + 0.9*sin(4*pi*x)*sin(4*pi*y);
  latex(f)
  
  % get the gradient of the function
  gf = gradient(f, [x,y]);
  % plot it as a surface
  fsurf(f)
  %set(gcf, 'Position', gfx_win_posn_size)
  colormap(parula); colormap(summer)
  light
  shading interp; grid off
  %light('Style','infinite')
  light('Position',[-5 -5 50],'Style','local')
  %camlight(180,80); % lighting gouraud
  %camlight left

  axis([-3 3, -3, 3, -10, 10]);
  view([-44,66])

  xlabel('v_1'); ylabel('v_2'); zlabel('f(v_1,v_2)');
  x0 = xo(1); y0 = xo(2); z0 = f(xo(1),xo(2));
  % x0=-0.5;  y0=1.4;
  % x0= 0.33; y0=1.4;
  % x0= 0.22; y0=1.5;
  %d=0.005; z0 = f(x0,y0);
  not_stopping = 1;
  while not_stopping
    x = [x0; y0] - alpha*gf(x0,y0);
    x1=double(x(1)); y1=double(x(2)); z1 = double(f(x1,y1));
    line([x0,x1],[y0,y1],[z0,z1],'LineWidth',3,'Color','red')  
    pause(0.1)
    if abs(z1-z0) < 0.001 
      not_stopping = 0;
    end
    x0=x1; y0=y1; z0=z1;
  end
elseif dim == 4  % not useful
  syms x y
  f = sin(x)+sin(y)-(x^2+y^2)/20
  fsurf(f,'ShowContours','on')
  view(-19,56)
fcontour(f,[-5 5 -5 5],'LevelStep',0.1,'Fill','on')
colorbar
hold on
Fgrad = gradient(f,[x,y])

[xgrid,ygrid] = meshgrid(-5:5,-5:5);
Fx = subs(Fgrad(1),{x,y},{xgrid,ygrid});
Fy = subs(Fgrad(2),{x,y},{xgrid,ygrid});
quiver(xgrid,ygrid,Fx,Fy,'k')
hold off  
  
  
  
elseif dim == 3
%   [X,Y,Z] = meshgrid(-6:.2:3);
%   V = X.*exp(-(Y.^2+Z.^2).*(sin(X.*Y.*Z)).^2);
%   xslice = [-4.5, -3.5, -2.5,-1.2,0.8,2.5];
%   yslice = [];
%   zslice = [];
%   hsurf = slice(X,Y,Z,V,xslice,yslice,zslice);
%   set(hsurf,'FaceColor','interp','FaceColor','interp')
%   colormap jet
%   colorbar
%   axis(6*[-1 1, -1, 1, -1, 1]);
%   view(3)
% 
%   grid on
%   pause 
  
  load wind
  xmin = min(x(:));
  xmax = max(x(:));
  ymax = max(y(:));
  zmin = min(z(:));
  % Add Slice Planes for Visual Context
  % Calculate the magnitude of the vector field (which represents wind speed) to generate scalar data for the slice command. Create slice planes along the x-axis at xmin, 100, and xmax, along the y-axis at ymax, and along the z-axis at zmin. Specify interpolated face coloring so the slice coloring indicates wind speed, and do not draw edges (sqrt, slice, FaceColor, EdgeColor).

  wind_speed = sqrt(u.^2 + v.^2 + w.^2);
  hsurfaces = slice(x,y,z,wind_speed,[xmin,100,xmax],ymax,zmin);
  set(hsurfaces,'FaceColor','interp','EdgeColor','none')
  colormap jet
  colorbar
  % 3. Add Contour Lines to the Slice Planes
  % Draw light gray contour lines on the slice planes to help quantify the color mapping (contourslice, EdgeColor, LineWidth).
  view([-37.5 30])
  view(3)
  daspect([2,2,1])
  axis tight
  pause
  view([-37.5 30])
  view(3)
  daspect([2,2,1])
  axis tight
  pause
  
  hcont = ...
  contourslice(x,y,z,wind_speed,[xmin,100,xmax],ymax,zmin);
  set(hcont,'EdgeColor',[0.7 0.7 0.7],'LineWidth',0.5)
  % 4. Define the Starting Points for Stream Lines
  % In this example, all stream lines start at an x-axis value of 80 and span the range 20 to 50 in the y-direction and 0 to 15 in the z-direction. Save the handles of the stream lines and set the line width and color (meshgrid, streamline, LineWidth, Color).

  [sx,sy,sz] = meshgrid(80,20:10:50,0:5:15);
  hlines = streamline(x,y,z,u,v,w,sx,sy,sz);
  set(hlines,'LineWidth',2,'Color','r')
  % 5. Define the View
  % Set up the view, expanding the z-axis to make it easier to read the graph (view, daspect, axis).

  view(3)
  daspect([2,2,1])
  axis tight

  % set(gcf, 'Position', gfx_win_posn_size)

end



