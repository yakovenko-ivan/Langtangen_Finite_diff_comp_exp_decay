from numpy import *

def solver(I, a, T, dt, theta):

    """ Solver for decay eq. u'(t)= -a * u(t),
    initial condition u(0)=I, time interval (0, T]
    theta = 1 - Backward Euler scheme
    theta = 0 - Forward Euler scheme
    theta = 0.5 - Crank-Nicolson mehtod
    """

    dt = float(dt)
    Nt = int(round(T/dt))
    T = Nt*dt
    u = zeros(Nt+1)
    t = linspace(0, T, Nt + 1)

    u[0] = I
    for n in range (0, Nt):
        u[n+1] = (1 - (1-theta)*a*dt)/(1 + theta*dt*a)*u[n]
    return u, t

def u_exact(t, I, a):
    return I*exp(-a*t)

from matplotlib.pyplot import *

def plot_numerical_and_exact(theta, I, a, T, dt):
    """ Compare the numerical and exact solution in a plot """
    u, t = solver(I=I, a=a, T=T, dt=dt, theta=theta)

    t_e = linspace(0,T, 1001)
    u_e = u_exact(t_e,I,a)
    
    plot(t, u, 'r--o', t_e, u_e, 'b-')

    legend(['numerical', 'exact'])
    xlabel('t')
    ylabel('u')
    title('theta=%g, dt=%g' % (theta,dt))
    savefig('plot_%s_%g.png' % (theta,dt))

def explore(I, a, T, dt, theta=0.5, makeplot=True):
    """
    Run a case with the solver, compute error, measure,
    and plot the numerical and exact solutions
    """

    u, t = solver(I, a, T, dt, theta)
    u_e  = u_exact(t, I, a)

    e = u_e - u
    E = sqrt(dt*sum(e**2))

    if makeplot:
        figure()
        t_e = linspace(0,T,1001)
        u_e = u_exact(t_e, I, a)
        plot(t, u, 'r--o', linewidth = 2, markersize = 5, markeredgewidth = 0.5, markeredgecolor = "black")
        plot(t_e, u_e, 'b-', linewidth = 2)
        legend(['numerical', 'exact'], fontsize = 15)
        xticks(fontsize=15)
        yticks(fontsize=15)
        xlabel('t', fontsize = 20, fontweight = "bold")
        ylabel('u', fontsize = 20, fontweight = "bold")
        theta2name = {0:'Forward Euler', 1:'Backward Euler', 0.5:'Crank-Nicolson'}
        title('%s, dt=%g' % (theta2name[theta],dt), fontsize = 25, fontweight ="bold", color = "red", fontstyle = "oblique")
        theta2name = {0:'FE', 1:'BE', 0.5:'CN'}
        savefig('%s_%g.png' % (theta2name[theta], dt))
    #    savefig('%s_%g.pdf' % (theta2name[theta], dt))
        show()
    return E

def solver_memsave(I, a, T, dt, theta, filename='sol.dat'):
    """ 
    Solve u'=-a*u, u(0) = I, for t in (0,T] with steps of dt.
    Minimum use of memory. The solution is stored in a file
    (with name filename) for later plotting
    """

    dt = float(dt)
    Nt = int(round(T/dt))

    outfile = open(filename,'w')
    # u: time level n+1, u_1: time level n
    t = 0
    u_1 = I
    outfile.write('%.16E %16.E\n' % (t, u_1))
    for n in range(1, Nt+1):
        u = (1-(1-theta)*a*dt)/(1 + theta*dt*a)*u_1
        u_1 = u
        t += dt
        outfile.write('%.16E %.16E\n' % (t,u))
    outfile.close()
    return u, t

def read_file(filename='sol.dat'):
    infile = open(filename, 'r')
    u = []; t = []
    for line in infile:
        words = line.split()
        if len(words) != 2:
            print ('Found more than two numbers on a line!', words)
            sys.exit(1) # abort
        t.append(float(words[0]))
        u.append(float(words[1]))
    return np.array(t), np.array(u)

def read_file_numpy(filename='sol.dat'):
    data = np.loadtxt(filename)
    t = data[:,0]
    u = data[:,1]
    return t, u


def main(I, a, T, dt_values, theta_values = (0, 0.5, 1)):
    print('theta dt error')
    theta2name = {0:'FE', 1:'BE', 0.5:'CN'}
    for theta in theta_values:
        for dt in dt_values:
            E = explore(I, a, T, dt, theta, makeplot = True)
            print('%s %4.1f %6.2f: %12.3E' % (theta2name[theta], theta, dt, E))

main(I=1, a=2, T=5, dt_values=[0.4,0.04])