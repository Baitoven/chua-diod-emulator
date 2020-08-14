from scipy import *
from pylab import *
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pypl
fig = pypl.figure ()
ax = fig.gca(projection="3d")

"""Parametre systeme"""
m0 = -0.41 
m1 = -0.79 
m2 = 4.54 #valeur théorique
m3 = -0.45 #valeur théorique
pc1 = 1.8 #valeur théorique, pc1 = 1.957
pc2 = 13.068 # valeur théorique
alpha = 10 # à bidouiller
beta = 14.32 #théoriquement, beta = 14


def carac(x) :
    return (m0+m2)*x+0.5*(m1-m0)*(abs(x+pc1)-abs(x-pc1))+(m3-m2)*(abs(x+pc2)-abs(x-pc2))

def carac2(x) :
    return m1*x+0.5*m0*(abs(x+pc1)-abs(x-pc1))

def deriv(syst, t):
    [x,y,z] = syst                # Variables
    dxdt=alpha*(y-x-carac2(x))                                           
    dydt=x-y+z                      
    dzdt=-beta*y                                           
    return [dxdt,dydt,dzdt]        

# Paramètres d'intégration
start = 0
end = 50
numsteps = 1000
t = linspace(start,end,numsteps)

# Conditions initiales et résolution
""" A déterminer !"""
x = 0
y = 0.1
z = 0
syst_CI=array([x,y,z])    # Tableau des CI
Sols=odeint(deriv,syst_CI,t)            # Résolution numérique des équations différentielles

# Récupération des solutions
[x,y,z] = Sols . T        # Décomposition du tableau des solutions

# Graphiques des solutions
ax.plot(x, y, z)   # Solution numérique
axis('equal')
legend()
show()