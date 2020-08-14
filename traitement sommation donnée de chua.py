import math as math
from scipy import *
from pylab import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt


"""Parametre systeme"""
m0 = -0.41 
m1 = -0.79 
m2 = 4.54 #valeur théorique
m3 = -0.45 #valeur théorique
pc1 = 1.8 #valeur théorique, pc1 = 1.957
pc2 = 13.068 # valeur théorique
alpha = 10 # à bidouiller
beta = 14.32 #théoriquement, beta = 14
R9 = 100000
R10 = 100000
R11 = 100000
R12 = 100000
R21 = 100000
R22 = 100000
R23 = 100000
R24 = 100000

def sommetableau(t1, t2) :
    """Parametres : t1 (Array)
                    t2 (Array)
       Précondition : t1 et t2 de meme taille
       Resultat : t3 (Array)"""
    t3 = []
    for k in range (0, len(t1)) :
        t3.append(t1[k] + t2[k])
    return t3

def signal(t) :
    """Signal Input
       Parametre : table temporelle (Array)
       Précondition : existence de function (function)
       Resultat : table de valeurs de la fonction (Array)"""
    tf = []
    for i in range(0,len(t)) :
        tf.append(function(t[i]))
    return tf 
        
    
def function(k) :
    return cos(k) #à choisir

def moins(t1) :
    """Parametres : t1 (array)
       Resultat : t2 (array)
       Effet : pour tout i indice du tableau, -t1[i] = t2[i]"""
    t2 = []
    for i in range (0, len(t1)) :
        t2.append(-t1[i])
    return t2





def solve_Chua() :
   
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
    [xs,ys,zs] = Sols.T       # Décomposition du tableau des solutions

    # Graphiques des solutions
    plt.plot(t, sommetableau(xs,signal(t)))   # Solution numérique
    plt.show()
    return tf


def UnSolve_Chua(tf2) :
    
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
    [xs,ys,zs] = Sols.T       # Décomposition du tableau des solutions

    # Graphiques des solutions
    plt.plot(t, sommetableau(moins(xs),tf2))   # Solution numérique 
    plt.show()
    return None
    
    
    
    
    
def tauxidentité(f1, f2) :
    """ Parametre : f1 et f2 resp. signaux input et output (function)
        Resultat : pourcentage d'identité entre f1 et f2 (float)"""
    n = 10 #largeur de l'intervalle étudié, à modifier à volonté
    h = 1000 #pas
    T = []
    for t in range (0, h -1) :
        u = ((t/h) * n)
        T.append(abs(f1(u)-f2(u)))
    sum = 0
    for k in range (0,len(T) -1) :
        sum = sum + T[k]
    return (1 - (sum/h))
    
