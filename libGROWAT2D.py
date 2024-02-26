"""
GROWAT2D
library for GROundWATer modelling in 2D
2024-02-09
Georg Kaufmann
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.sparse
import sys

#================================#
def readParameter2D(infile='GROWAT2D_parameter.in',path='work/',control=False):
    """
    ! read GROWAT2D parameter file
    ! input:
    !  (from file infile)
    ! output:
    !  xmin,xmax,nx         - min/max for x coordinate [m], discretisation
    !  ymin,ymax,ny         - min/max for y coordinate [m], discretisation
    !  whichtime            - flag for time units used
    !  time_start,time_end  - start/end point for time scale [s]
    !  time_step            - time step [s]
    !  time_scale           - scaling coefficient for user time scale
    ! use:
    !  xmin,xmax,nx,ymin,ymax,ny,time_start,time_end,time_step,time_scale,whichtime = libGROWAT2D.readParameter2D()
    ! note:
    !  file structure given!
    !  uses readline(),variables come in as string,
    !  must be separated and converted ...
    """
    # read in data from file
    f = open(path+infile,'r')
    line = f.readline()
    xmin,xmax,nx = float(line.split()[0]),float(line.split()[1]),int(line.split()[2])
    line = f.readline()
    ymin,ymax,ny = float(line.split()[0]),float(line.split()[1]),int(line.split()[2])
    line = f.readline()
    whichtime = line.split()[0]
    line = f.readline()
    time_start,time_end,time_step = float(line.split()[0]),float(line.split()[1]),float(line.split()[2])
    f.close()
    # convert times from user times to seconds, based on whichtime flag
    min2sec  = 60.               # seconds per minute
    hour2sec = 3600.             # seconds per hour
    day2sec  = 24.*3600.         # seconds per day
    year2sec = 365.25*day2sec    # seconds per year
    month2sec = year2sec/12.00   # seconds per month
    if (whichtime=='min'):   time_scale = min2sec
    if (whichtime=='hour'):  time_scale = hour2sec
    if (whichtime=='day'):   time_scale = day2sec
    if (whichtime=='month'): time_scale = month2sec
    if (whichtime=='year'):  time_scale = year2sec
    time_start *= time_scale
    time_end   *= time_scale
    time_step  *= time_scale
    # control output to screen
    if (control):
        print('== GROWAT2D ==')
        print('%20s %20s' % ('path:',path))
        print('%20s %10.2f %10.2f' % ('xmin,xmax [m]:',xmin,xmax))
        print('%20s %10.2f %10.2f' % ('ymin,ymax [m]:',ymin,ymax))
        print('%20s %10i %10i' % ('nx,ny:',nx,ny))
        
        print('%20s %10.2f' % ('time_start['+whichtime+']:',time_start/time_scale))
        print('%20s %10.2f' % ('time_end['+whichtime+']:',time_end/time_scale))
        print('%20s %10.2f' % ('time_step['+whichtime+']:',time_step/time_scale))
        print('%20s %s' % ('whichtime:',whichtime))
    return xmin,xmax,nx,ymin,ymax,ny,time_start,time_end,time_step,time_scale,whichtime


#================================#
def readHEADBC2D(infile='GROWAT2D_bc.in',path='work/',control=False):
    """
    ! read GROWAT2D boundary conditions file
    ! input:
    ! (from file infile)
    ! output:
    !  dataBC      - array of boundary conditions
    ! use:
    !  dataBC = libGROWAT2D.readHEADBC2D()
    ! note:
    !  uses np.loadtxt(), data read as (float) array
    !  first two lines are meta data and are mandatory!
    """
    # read in data from file
    dataBC = np.loadtxt(path+infile,skiprows=2)
    if (control):
        for i in range(dataBC.shape[0]):
            print('%20s %30s' % ('BC:',dataBC[i]))
    return dataBC


#================================#
def readMaterials2D(infile='GROWAT2D_materials.in',path='work/',control=False):
    """
    ! read GROWAT2D material areas file
    ! input:
    !  (from file infile)
    ! output:
    !  dataMAT      - array of boundary conditions
    ! use:
    !  dataMAT = libGROWAT2D.readMaterials2D()
    ! note:
    !  uses np.loadtxt(), data read as (float) array
    !  first two lines meta data!
    """
    # read in data from file
    dataMAT = np.loadtxt(path+infile,skiprows=2,dtype='str')
    if (dataMAT.ndim == 1): dataMAT = np.array([dataMAT])
    if (control):
        for i in range(dataMAT.shape[0]):
            print('%20s %30s' % ('Materials:',dataMAT[i]))
    return dataMAT


#================================#
def createNodes2D(xmin,xmax,nx,ymin,ymax,ny,control=False):
    """
    !-----------------------------------------------------------------------
    ! create nodes in 2D domain
    ! first, a rectangular box with the given limits
    ! Input:
    !  xmin,xmax         - min/max for x
    !  ymin,ymax         - min/max for y
    !  nx,ny             - number of nodes in x,y direction
    ! Output:
    !  X(ijk)            - x-coordinate [m] of node ijk
    !  Y(ijk)            - y-coordinate [m] of node ijk
    !  dx,dy             - spatial discretisation [m]
    ! use:
    !  X,Y,dx,dy = libGROWAT2D.createNodes2D(xmin,xmax,nx,ymin,ymax,ny)
    ! notes:
    !  ij,ijk            - node counter in x/y and x/y/z direction
    !-----------------------------------------------------------------------
    """
    # first linear set
    x,dx = np.linspace(xmin,xmax,nx,retstep=True)
    y,dy = np.linspace(ymin,ymax,ny,retstep=True)
    # then 3D arrays
    X = np.zeros(nx*ny); Y = np.zeros_like(X)
    for i in range(nx):
        for j in range(ny):
            k = 0
            ijk = i + (j)*nx + (k)*nx*ny
            X[ijk] = x[i]
            Y[ijk] = y[j]
    # control output to screen
    if (control):
        print('%20s %10.2f %10.2f' % ('X.min,X.max [m]:',X.min(),X.max()))
        print('%20s %10.2f %10.2f' % ('Y.min,Y.max [m]:',Y.min(),Y.max()))
    return X,Y,dx,dy


#================================#
def createFields2D(nx,ny):
    """
    ! function initializes all field with appropriate dimensions
    ! input:
    !  nx,ny           - coordinate increments
    ! output:
    !  K               - conductivity  of node i
    !  S               - storativity of node i
    !  flow ,head      - flow, head of node i
    !  vx,vy           - x- and y-velocity components
    ! use:
    !  K,S,head,flow,vx,vy = libGROWAT2D.createFields2D(nx,ny)
    ! notes:
    """
    K        = np.zeros(nx*ny)
    S        = np.zeros(nx*ny)
    head     = np.zeros(nx*ny)
    flow     = np.zeros(nx*ny)
    vx,vy    = np.zeros(nx*ny),np.zeros(nx*ny)
    return K,S,head,flow,vx,vy


#================================#
def createProperties2D(dataMAT,K,S,X,Y,nx,ny,control=False):
    """
    ! input:
    !  dataMAT    - dictionary of materials
    !  K,S        - hydraulic conductivity [m/s], specific storage [1/s]
    !  X,Y        - x- and y-coordinate [m] of node i
    !  nx,ny      - coordinate increments
    ! output:
    !  K,S        - hydraulic conductivity [m/s], specific storage [1/s]
    ! use:
    !  K,S = libGROWAT2D.createProperties2D(dataMAT,K,S,X,Y,nx,ny)
    ! notes:
    """
    # main material (first line in GROWAT2D_materials.in)
    for i in range(nx):
        for j in range(ny):
            ij = i + (j)*nx
            K[ij] = float(dataMAT[0][5])
            S[ij] = float(dataMAT[0][6])
    #print('Primary material used')
    # additional materials (other lines in GROWAT2D_materials.in)
    if (dataMAT.shape[0] > 1):
        for ib in range(1,dataMAT.shape[0]):
            mat = dataMAT[ib][0]
            x1 = float(dataMAT[ib][1])
            x2 = float(dataMAT[ib][2])
            y1 = float(dataMAT[ib][3])
            y2 = float(dataMAT[ib][4])
            if (x1 < X.min()): sys.exit ('x1 < X.min()')
            if (x2 > X.max()): sys.exit ('x2 > X.max()')
            if (y1 < Y.min()): sys.exit ('y1 < Y.min()')
            if (y2 > Y.max()): sys.exit ('y2 > Y.max()')

            for i in range(nx):
                for j in range(ny):
                    ij = i + (j)*nx
                    if (X[ij] >= x1 and X[ij] <= x2 and Y[ij] >= y1 and Y[ij] <= y2):
                        K[ij] = float(dataMAT[ib][5])
                        S[ij] = float(dataMAT[ib][6])
        #print('Secondary material used: ',ib)
    if (control):
        print('%20s %10.6f %10.6f' % ('K.min,K.max [m/s]:',K.min(),K.max()))
        print('%20s %10.6f %10.6f' % ('S.min,S.max [1/s]:',S.min(),S.max()))
    return K,S


#================================#
def buildHEADBC2D(dataBC,nx,ny,dx,dy,time,time_scale,head,flow,control=False):
    """
    ! set nodes marked with boundary conditions for current time step
    !  ibound(i)   - boundary flag for node i
    !            0 - unknown head
    !            1 - fixed resurgence
    !            2 - fixed head
    !            3 - fixed recharge
    !            4 - fixed sink
    ! use:
    !  ibound,irecharge,head,flow = libGROWAT2D.buildHEADBC2D(dataBC,dx,dy,time,time_scale,head,flow)
    ! notes:
    """
    ifixhead     = 0; ifixres      = 0
    ifixrecharge = 0; ifixsink     = 0
    # set flow to zero to initialize
    flow = np.zeros(nx*ny)
    # open arrays for boundary index and values
    ibound    = np.zeros(nx*ny,dtype='int')
    irecharge = np.zeros(nx*ny,dtype='int')
    for ib in range(dataBC.shape[0]):
        itype = int(dataBC[ib,0])
        i1 = int(dataBC[ib,1]);i2 = int(dataBC[ib,2])
        j1 = int(dataBC[ib,3]);j2 = int(dataBC[ib,4])
        t1 = float(dataBC[ib,5]);t2 = float(dataBC[ib,6])
        value = float(dataBC[ib,7])
        if (i1 >= nx): sys.exit ('i1 > nx-1')
        if (i2 >= nx): sys.exit ('i2 > nx-1')
        if (j1 >= ny): sys.exit ('j1 > ny-1')
        if (j2 >= ny): sys.exit ('j2 > ny-1')
    # assign values to arrays
        for i in range(i1,i2+1):
            for j in range(j1,j2+1):
                k=0
                ijk = i + (j)*nx + (k)*nx*ny
                surf = dx*dy
                if (i==0 or i==nx-1): surf = 0.5*dx*dy
                if (j==0 or j==ny-1): surf = 0.5*dx*dy
                if (i==0 and j==0): surf = 0.25*dx*dy
                if (i==0 and j==ny-1): surf = 0.25*dx*dy
                if (i==nx-1 and j==0): surf = 0.25*dx*dy
                if (i==nx-1 and j==ny-1): surf = 0.25*dx*dy
                # check, if BC is active
                if (time >= t1*time_scale and time <= t2*time_scale):
                    # fixed resurgence head node
                    if (itype==1):
                        ibound[ijk] = itype
                        head[ijk]   = value
                        ifixres += 1
                    # fixed head node
                    if (itype==2):
                        ibound[ijk] = itype
                        head[ijk]   = value
                        ifixhead += 1
                    # fixed recharge node (mm/timeUnit -> m3/s)
                    if (itype==3):
                        irecharge[ijk] = itype
                        flow[ijk]      = value/time_scale/1000.*surf
                        ifixrecharge += 1
                    # fixed sink node (m3/timeUnit -> m3/s)
                    if (itype==4):
                        ibound[ijk] = itype
                        flow[ijk]      = value/time_scale
                        ifixsink += 1
    if (control):
        print('ifixres:      ',ifixres)
        print('ifixhead:     ',ifixhead)
        print('ifixrecharge: ',ifixrecharge)
        print('ifixsink:     ',ifixsink)
    return ibound,irecharge,head,flow


#================================#
def buildHeadEquations2D_ss(dx,dy,nx,ny,Km,head,flow,ibound,bc='noflow'):
    """
    ! function assembles the element entries for the global conductance
    ! matrix and the rhs vector for the steady-state case
    ! input:
    !  dx,dy      - discretisation [m]
    !  nx,ny      - coordinate increments
    !  K,S        - hydraulic conductivity [m/s], specific storage [1/s]
    !  ibound     - array for boundary markers
    !  head,flow  - head [m] and flow [m3/s] fields
    !  bc         - boundary condition flag (
    !               initial - set  boundary nodes to initial head 
    !               noflow  - set  boundary nodes to no-flow
    ! output:
    !  matrix     - global conductivity matrix (sparse)
    !  rhs        -  rhs vector
    ! use:
    !  matrix,rhs = libGROWAT2D.buildHeadEquations2D_ss(dx,dy,nx,ny,K,head,flow,ibound)
    ! notes:
    """
    # initialize fields for sparse matrix and rhs vector
    rhs    = np.zeros([nx*ny])
    matrix = scipy.sparse.lil_array((nx*ny,nx*ny))
    #-----------------------------------------------------------------------
    # assemble matrix, loop over all interior nodes
    #-----------------------------------------------------------------------
    dx2 = 1. / dx**2
    dy2 = 1. / dy**2
    for j in range(ny):
        for i in range(nx):
            ij = i + (j)*nx
            matrix[ij,ij]   = 0.
            # fixed-head boundary condition node
            if (np.abs(ibound[ij])==1 or np.abs(ibound[ij])==2): 
                matrix[ij,ij]   = 1e10
            # other nodes
            else:
                # diffusion operator in interior region
                if (i != 0 and i != nx-1 and j != 0 and j != ny-1):
                    # diffusivity in x direction
                    Kleft  = (Km[ij-1]+Km[ij])/2
                    Kright = (Km[ij]+Km[ij+1])/2
                    matrix[ij,ij]   += -1*dx2*(Kleft + Kright)
                    matrix[ij,ij+1] += +1*dx2*Kright
                    matrix[ij,ij-1] += +1*dx2*Kleft
                    # diffusivity in y direction
                    Kbottom = (Km[ij-nx]+Km[ij])/2
                    Ktop    = (Km[ij]+Km[ij+nx])/2
                    matrix[ij,ij]    += -1*dy2*(Kbottom + Ktop)
                    matrix[ij,ij+nx] += +1*dy2*Ktop
                    matrix[ij,ij-nx] += +1*dy2*Kbottom
                # boundary conditions along the sides ...
                if (bc=='initial'):
                    if (i==0):
                        ibound[ij]    = -1
                        matrix[ij,ij] = 1e10
                    if (i==nx-1):
                        ibound[ij]    = -1
                        matrix[ij,ij] = 1e10
                    if (j==0):
                        ibound[ij]    = -1
                        matrix[ij,ij] = 1e10
                    if (j==ny-1):
                        ibound[ij]    = -1
                        matrix[ij,ij] = 1e10
                if (bc=='noflow'):
                    # left side
                    if (i==0):
                        if (j==0):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij+1+nx] = -Km[ij+1+nx]/dx
                        elif (j==ny-1):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij+1-nx] = -Km[ij+1-nx]/dx
                        else:
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij+1] = -Km[ij+1]/dx
                    # right side
                    if (i==nx-1):
                        if (j==0):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij-1+nx] = -Km[ij-1+nx]/dx
                        elif (j==ny-1):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij-1-nx] = -Km[ij-1-nx]/dx
                        else:
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij-1] = -Km[ij-1]/dx
                    #bottom side    
                    if (j==0):
                        if (i==0):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij+1+nx] = -Km[ij+1+nx]/dx
                        elif (i==nx-1):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij-1+nx] = -Km[ij-1+nx]/dx
                        else:
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij+nx] = -Km[ij+nx]/dx
                    # top side
                    if (j==ny-1):
                        if (i==0):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij+1-nx] = -Km[ij+1-nx]/dx
                        elif (i==nx-1):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij-1-nx] = -Km[ij-1-nx]/dx
                        else:
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij-nx] = -Km[ij-nx]/dx
                
                # flow boundary conditions not checked!
                #elif (i == 0):
                    #matrix[ij,ij]   +=  2*dx2*Km[ij]
                    #matrix[ij,ij+1] += -5*dx2*Km[ij+1]
                    #matrix[ij,ij+2] +=  4*dx2*Km[ij+2]
                    #matrix[ij,ij+3] += -2*dx2*Km[ij+2]
                #elif (i == nx-1):
                    #matrix[ij,ij-3] += -1*dx2*Km[ij-3]
                    #matrix[ij,ij-2] += +4*dx2*Km[ij-2]
                    #matrix[ij,ij-1] += -5*dx2*Km[ij-1]
                    #matrix[ij,ij]   += +2*dx2*Km[ij]
                #elif (j == 0):
                    #matrix[ij,ij]      +=  2*dy2*Km[ij]
                    #matrix[ij,ij+nx+1] += -5*dy2*Km[ij+nx+1]
                    #matrix[ij,ij+nx+2] +=  4*dy2*Km[ij+nx+2]
                    #matrix[ij,ij+nx+3] += -2*dy2*Km[ij+nx+3]
                #elif (j == ny-1):
                    #matrix[ij,ij-nx-3] += -1*dy2*Km[ij-nx-3]
                    #matrix[ij,ij-nx-2] += +4*dy2*Km[ij-nx-2]
                    #matrix[ij,ij-nx-1] += -5*dy2*Km[ij-nx-1]
                    #matrix[ij,ij-nx]   += +2*dy2*Km[ij-nx]
    #-----------------------------------------------------------------------
    # assemble right-hand side
    #-----------------------------------------------------------------------
    for j in range(ny):
        for i in range(nx):
            ij = i + (j)*nx
            rhs[ij] = 0.
            # fixed-head boundary condition node
            if (np.abs(ibound[ij])==1 or np.abs(ibound[ij])==2): 
                rhs[ij] = matrix[ij,ij]*head[ij]
            # other nodes
            else:
            # surface inflow
                surface = dx*dy
                if ((i==0) or (i==nx-1)):     surface = 0.5*dx*dy
                if ((j==0) or (j==ny-1)):     surface = 0.5*dx*dy
                if ((i==0)  and (j==0)):      surface = 0.25*dx*dy
                if ((i==0)  and (j==ny-1)):   surface = 0.25*dx*dy
                if ((i==nx-1) and (j==0)):    surface = 0.25*dx*dy
                if ((i==nx-1) and (j==ny-1)): surface = 0.25*dx*dy
                # classical diffusion operator in interior region
                if (i != 0 and i != nx-1 and j != 0 and j != ny-1):
                    rhs[ij] = -flow[ij]/surface
                # check for no-flow boundary condition
                if (bc=='noflow'):
                    if (i==0):    rhs[ij] = 0.
                    if (i==nx-1): rhs[ij] = 0.
                    if (j==0):    rhs[ij] = 0.
                    if (j==ny-1): rhs[ij] = 0.
    # convert sparse lil format to csr format
    matrix = matrix.tocsr()
    return matrix,rhs


#================================#
def buildHeadEquations2D_t(dx,dy,nx,ny,time_step,Km,Sm,head,headOld,flow,flowOld,ibound,omega=1.,bc='noflow'):
    """
    ! function assembles the element entries for the global conductance
    ! matrix and the rhs vector for the transient case
    ! input:
    !  dx,dy      - discretisation [m]
    !  nx,ny      - coordinate increments
    !  K,S        - hydraulic conductivity [m/s], specific storage [1/s]
    !  ibound     - array for boundary markers
    !  head,flow  - head [m] and flow [m3/s] fields
    !  omega      - relaxation parameter (default: 1)
    !               0-explicit, 1-implicit, 0.5-Crank-Nicholson
    !  bc         - boundary condition flag (default: noflow)
    !               initial - set  boundary nodes to initial head 
    !               noflow  - set  boundary nodes to no-flow
    ! output:
    !  matrix     - global conductivity matrix (sparse)
    !  rhs        -  rhs vector
    ! use:
    !  matrix,rhs = libGROWAT2D.buildHeadEquations2D_t(dx,dy,nx,ny,K,S,head,headOld,flow,flowOld,ibound)
    ! notes:
    """
    # initialize fields for sparse matrix and rhs vector
    rhs    = np.zeros([nx*ny])
    matrix = scipy.sparse.lil_array((nx*ny,nx*ny))
    #-----------------------------------------------------------------------
    # assemble matrix, loop over all interior nodes
    # omega = 0: fully explicit
    # omega = 1: fully implicit
    #-----------------------------------------------------------------------
    dtdx2 = time_step / dx**2
    dtdy2 = time_step / dy**2
    for j in range(ny):
        for i in range(nx):
            ij = i + (j)*nx
            matrix[ij,ij] = 1.
            # fixed-head boundary condition node
            if (np.abs(ibound[ij])==1 or np.abs(ibound[ij])==2):
                matrix[ij,ij]   = 1e10
            # other nodes
            else:
                # diffusion operator in interior region (implicit)
                if (i != 0 and i != nx-1 and j != 0 and j != ny-1):
                    # diffusivity in x direction (implicit)
                    Kleft  = (Km[ij-1]+Km[ij])/2
                    Kright = (Km[ij]+Km[ij+1])/2
                    matrix[ij,ij]   += +1*omega*dtdx2*(Kleft + Kright)/Sm[ij]
                    matrix[ij,ij+1] += -1*omega*dtdx2*Kright/Sm[ij]
                    matrix[ij,ij-1] += -1*omega*dtdx2*Kleft/Sm[ij]
                    # diffusivity in y direction (implicit)
                    Kbottom = (Km[ij-nx]+Km[ij])/2
                    Ktop    = (Km[ij]+Km[ij+nx])/2
                    matrix[ij,ij]    += +1*omega*dtdy2*(Kbottom + Ktop)/Sm[ij]
                    matrix[ij,ij+nx] += -1*omega*dtdy2*Ktop/Sm[ij]
                    matrix[ij,ij-nx] += -1*omega*dtdy2*Kbottom/Sm[ij]
                # boundary conditions along the sides ...
                if (bc=='initial'):
                    if (i==0):
                        ibound[ij]    = -1
                        matrix[ij,ij] = 1e10
                    if (i==nx-1):
                        ibound[ij]    = -1
                        matrix[ij,ij] = 1e10
                    if (j==0):
                        ibound[ij]    = -1
                        matrix[ij,ij] = 1e10
                    if (j==ny-1):
                        ibound[ij]    = -1
                        matrix[ij,ij] = 1e10
                if (bc=='noflow'):
                    # left side
                    if (i==0):
                        if (j==0):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij+1+nx] = -Km[ij+1+nx]/dx
                        elif (j==ny-1):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij+1-nx] = -Km[ij+1-nx]/dx
                        else:
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij+1] = -Km[ij+1]/dx
                    # right side
                    if (i==nx-1):
                        if (j==0):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij-1+nx] = -Km[ij-1+nx]/dx
                        elif (j==ny-1):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij-1-nx] = -Km[ij-1-nx]/dx
                        else:
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij-1] = -Km[ij-1]/dx
                    #bottom side    
                    if (j==0):
                        if (i==0):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij+1+nx] = -Km[ij+1+nx]/dx
                        elif (i==nx-1):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij-1+nx] = -Km[ij-1+nx]/dx
                        else:
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij+nx] = -Km[ij+nx]/dx
                    # top side
                    if (j==ny-1):
                        if (i==0):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij+1-nx] = -Km[ij+1-nx]/dx
                        elif (i==nx-1):
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij-1-nx] = -Km[ij-1-nx]/dx
                        else:
                            matrix[ij,ij] = Km[ij]/dx
                            matrix[ij,ij-nx] = -Km[ij-nx]/dx
    #-----------------------------------------------------------------------
    # assemble right-hand side
    #-----------------------------------------------------------------------
    for j in range(ny):
        for i in range(nx):
            ij = i + (j)*nx
            rhs[ij] = headOld[ij]
            # fixed-head boundary condition node
            if (np.abs(ibound[ij])==1 or np.abs(ibound[ij])==2): 
                rhs[ij] = matrix[ij,ij]*head[ij]
            # other nodes
            else:        
                surface = dx*dy
                if ((i==0) or (i==nx-1)):     surface = 0.5*dx*dy
                if ((j==0) or (j==ny-1)):     surface = 0.5*dx*dy
                if ((i==0)  and (j==0)):      surface = 0.25*dx*dy
                if ((i==0)  and (j==ny-1)):   surface = 0.25*dx*dy
                if ((i==nx-1) and (j==0)):    surface = 0.25*dx*dy
                if ((i==nx-1) and (j==ny-1)): surface = 0.25*dx*dy
                # interior nodes
                if (i != 0 and i != nx-1 and j != 0 and j != ny-1):
                    # flow as right-hand side condition
                    rhs[ij] += (1-omega)*flowOld[ij]/surface*time_step/Sm[ij]
                    rhs[ij] += omega*flow[ij]/surface*time_step/Sm[ij]
                    # diffusion operator in interior region (explicit)
                    # diffusivity in x direction (explicit)
                    Kleft  = (Km[ij-1]+Km[ij])/2
                    Kright = (Km[ij]+Km[ij+1])/2
                    rhs[ij] += -1*headOld[ij]*(1-omega)*dtdx2*(Kleft + Kright)/Sm[ij] 
                    rhs[ij] += +1*headOld[ij+1]*(1-omega)*dtdx2*Kright/Sm[i] 
                    rhs[ij] += +1*headOld[ij-1]*(1-omega)*dtdx2*Kleft/Sm[ij]
                    # diffusivity in y direction (explicit)
                    Kbottom = (Km[ij-nx]+Km[ij])/2
                    Ktop    = (Km[ij]+Km[ij+nx])/2
                    rhs[ij] += -2*headOld[ij]*(1-omega)*dtdy2*(Kbottom + Ktop)/Sm[ij] 
                    rhs[ij] += +1*headOld[ij+nx]*(1-omega)*dtdy2*Ktop/Sm[ij] 
                    rhs[ij] += +1*headOld[ij-nx]*(1-omega)*dtdy2*Kbottom/Sm[ij]
                # check for no-flow boundary condition
                if (bc=='noflow'):
                    if (i==0):    rhs[ij] = 0.
                    if (i==nx-1): rhs[ij] = 0.
                    if (j==0):    rhs[ij] = 0.
                    if (j==ny-1): rhs[ij] = 0.
    # convert sparse lil format to csr format
    matrix = matrix.tocsr()
    return matrix,rhs


#================================#
def solveLinearSystem2D(matrix,rhs):
    """
    ! Solve linear system of equations
    ! with sparse matrix solver
    ! input:
    !  matrix     - global conductivity matrix (sparse)
    !  rhs        -  rhs vector
    ! output:
    !  head       - head [m] field
    ! use:
    !  head = libGROWAT2D.solveLinearSystem2D(matrix,rhs)
    """
    head = scipy.sparse.linalg.spsolve(matrix,rhs,permc_spec='MMD_AT_PLUS_A')
    return head


#================================#
def solveVelocities2D(nx,ny,time_scale,x,y,K,head):
    """
    ! function calculates velocity components in center of block
    ! input:
    !  nx                - coordinate increments
    !  time_scale        - time scale to convert velocities to user timescale
    !  x,y               - x,y coordinates of node i
    !  K                 - conductivity  of node i
    !  head              - head of node i
    ! output:
    !  xc,yc             - x,y coordinates of velocity node j
    !  vc,vc             - x,y velocity components of velocity node j
    ! use:
    !  xc,yc,vcx,vcy = libGROWAT2D.solveVelocities2D(nx,ny,time_scale,x,y,K,head)
    """
    # define velocity components in center of grid
    nvel     = (nx-1)*(ny-1)
    xc       = np.zeros(nvel)
    yc       = np.zeros(nvel)
    vcx      = np.zeros(nvel)
    vcy      = np.zeros(nvel)
    dndx = np.zeros(4)
    dndy = np.zeros(4)
    iv = 0
    for j in range(ny-1):
        for i in range(nx-1):
            ij = i + (j)*nx
            i1 = ij
            i2 = ij+1
            i3 = ij+nx+1
            i4 = ij+nx
            twoA = np.abs(y[i1]-y[i3])
            twoB = np.abs(x[i1]-x[i2])
            t = twoA/2.;s = twoB/2.
            dndx[0] = (t-twoA) / (twoA*twoB)
            dndx[1] = -(t-twoA) / (twoA*twoB)
            dndx[2] = t / (twoA*twoB)
            dndx[3] = -t / (twoA*twoB)
            dndy[0] = (s-twoB) / (twoA*twoB)
            dndy[1] = -s / (twoA*twoB)
            dndy[2] = s / (twoA*twoB)
            dndy[3] = -(s-twoB) / (twoA*twoB)

            dhdx = dndx[0]*head[i1] + dndx[1]*head[i2] + dndx[2]*head[i3] + dndx[3]*head[i4]
            dhdy = dndy[0]*head[i1] + dndy[1]*head[i2] + dndy[2]*head[i3] + dndy[3]*head[i4]

            kave = 0.25*(K[i1] + K[i2] + K[i3] + K[i4])
            xc[iv] = 0.25*(x[i1] + x[i2] + x[i3] + x[i4])
            yc[iv] = 0.25*(y[i1] + y[i2] + y[i3] + y[i4])
            vcx[iv] = -time_scale*kave*dhdx
            vcy[iv] = -time_scale*kave*dhdy
            iv = iv+1
    return xc,yc,vcx,vcy


#================================#
def saveToScreen2D(saved,time,time_scale,head,vabs):
    """
    function creates on-line screen output
    """
    format = "%3s t:%10i h: %8.2f +/- %8.2f %8.2f - %8.2f v: %8.2f - %8.2f"
    headMean,headStd = round(head.mean(),2),round(head.std(),2)
    headMin,headMax  = round(head.min(),2),round(head.max(),2)
    vabsMin,vabsMax  = round(vabs.min(),2),round(vabs.max(),2)
    print(format % (saved,time/time_scale,headMean,headStd,headMin,headMax,vabsMin,vabsMax)) 
    return


#================================#
def saveHeadsAndVelocities2D(itime,time,time_scale,whichtime,x,y,head,flow,xc,yc,vcx,vcy,ibound,irecharge,path='work/',name='FD_'):
    """
    ! function saves head, flow and velocity components to file
    ! input:
    !  itime,time        - time increment and current time [s]
    !  whichtime         - time flag
    !  time_scale        - time scale to convert velocities to user timescale
    !  x,y               - x coordinate of node i
    !  xc,yc             - x coordinate of velocity node j
    !  vcx,vcy           - x velocity component of velocity node j
    !  K                 - conductivity [m/s] of node i
    !  head              - head [m] of node i
    !  flow              - flow [m3/s] of node i
    !  ibound            - flag for boundary nodes
    !  irecharge         - flag for recharge nodes
    ! output:
    !  (to files)
    ! use:
    !  libGROWAT2D.saveHeadsAndVelocities2D(itime,time,time_scale,whichtime,x,y,head,flow,xc,yc,vcx,vcy,ibound,irecharge)
    """
    # save heads and flow to filename1
    format1 = "%10i %12.2f %12.2f %12.2f %12.2f %2i %2i"
    filename1 = path+name+f"{itime:04}.heads"
    f = open(filename1,'w')
    print('time,whichtime: ',time/time_scale,whichtime,file=f)
    for i in range(x.shape[0]):
        print(format1 % (i,x[i],y[i],head[i],flow[i],ibound[i],irecharge[i]),file=f)
    f.close()
    
    # save velocities to filename2
    format2 = "%10i %12.2f %12.2f %12.2f %12.2f"
    filename2 = path+name+f"{itime:04}.vel"
    f = open(filename2,'w')
    print('time,whichtime: ',time/time_scale,whichtime,file=f)
    for i in range(xc.shape[0]):
        print(format2 % (i,xc[i],yc[i],vcx[i],vcy[i]),file=f)
    f.close()
    return


#================================#
def plotHeadsAndVelocities2D(itime,time,time_scale,x,y,head,xc,yc,vcx,vcy,ibound,irecharge,
                             vmin=0.,vmax=4.,vstep=21,plot=False,path='work/',name='FD_'):
    """
    function plots heads and velocities
    input:
    itime,time,time_scale - time iterator, time, time scale
    x,y [m]               - x- and y-coordinates for heads
    head [m]              - hydraulic head
    xc,yc [m]             - x- and y-coordinates for velocities
    vcx,vcy [m/s]         - velocity in x- and y-direction
    vmin,vmax             - min/max value for colorbar (defaults=0,4)
    vstep                 - steps for colorbar (default=21)
    plot                  - flag for showing plot (default=False)
    name                  - prefix for filename (default='FD_')
    output:
    figure, if plot flag set to True
    """
    filename = path+name+f"{itime:04}.png"
    fig,axs = plt.subplots(1,1,figsize=(9,6))
    
    axs.set_xlabel('x [m] ')
    axs.set_ylabel('y [m]')
    axs.set_title('Time: '+str(round(time/time_scale,0))+' d')
    axs.set_xlim([x.min(),x.max()])
    axs.set_ylim([y.min(),y.max()])
    
    #axs.tricontour(x, y, head, levels=levels, linewidths=0.3, colors='k',vmin=vmin,vmax=vmax)
    norm = plt.Normalize(vmin,vmax)
    #norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
    levels = np.linspace(vmin,vmax,vstep)
    img0 = axs.tricontourf(x, y, head, levels=levels, cmap="RdBu_r",vmin=vmin,vmax=vmax,norm=norm)
    cbar0=fig.colorbar(img0,ax=axs,orientation='vertical',location='left',shrink=0.8,ticks=[0,1,2,3,4])
    cbar0.set_label('heads [m]')
    
    axs.plot(x[ibound==1],y[ibound==1],lw=0,marker='o',markersize=10,markerfacecolor='none',
             markeredgecolor='gray',alpha=0.5,label='1')
    axs.plot(x[ibound==2],y[ibound==2],lw=0,marker='s',markersize=10,markerfacecolor='none',
             markeredgecolor='gray',alpha=0.5,label='2')
    axs.plot(x[irecharge==3],y[irecharge==3],lw=0,marker='v',markersize=10,markerfacecolor='none',
             markeredgecolor='gray',alpha=0.5,label='3')
    axs.plot(x[ibound==4],y[ibound==4],lw=0,marker='x',markersize=10,markerfacecolor='none',
             markeredgecolor='gray',alpha=0.5,label='4')
    
    cmapVel = mpl.cm.jet
    vcabs = np.sqrt(vcx**2 + vcy**2)
    normVel = plt.Normalize(0,1)
    levelsVel = np.linspace(0,1,21)
    vcabs += 0.001
    cbar1=axs.quiver(xc,yc,vcx/vcabs,vcy/vcabs,vcabs,alpha=0.4,width=0.005,scale=20,pivot="middle",
                     cmap=cmapVel,norm=normVel)
    clabel1=plt.colorbar(cbar1,extend='both',orientation='vertical',location='right',shrink=0.8)
    clabel1.set_label('v [m/d]')
    plt.tight_layout()
    plt.savefig(filename)
    if (not plot):
        plt.close()
    return