import math
from scipy import *
from pylab import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


"""Parametre systeme"""
m0 = -0.41 
m1 = -0.79 
m2 = 4.54 #valeur théorique
m3 = -0.45 #valeur théorique
pc1 = 1.8 #valeur théorique, pc1 = 1.957
pc2 = 13.068 # valeur théorique
"""Valeur théoriques de la diode : alpha = 10 # à bidouiller
                                   beta = 14.32 #théoriquement, beta = 14"""
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
        
    
def function(k) : #fonction de test
    return cos(k) #à choisir

def moins(t1) :
    """Parametres : t1 (array)
       Resultat : t2 (array)
       Effet : pour tout i indice du tableau, -t1[i] = t2[i]"""
    t2 = []
    for i in range (0, len(t1)) :
        t2.append(-t1[i])
    return t2




def XY(alpha,beta) :
    """ Parametres : alpha, beta parametres systemes (float)
        Précondition : on suppose disposer de signal: array -> array correspondant à l'input
        Effet : affiche le XY de la diode de chua
        Résultat : Aucun"""
        
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
    plt.plot(xs, ys)   # Solution numérique
    plt.show()
    return None




def solve_Chua(alpha,beta) :
    """ Parametres : alpha, beta parametres systemes (float)
        Précondition : on suppose disposer de signal: array -> array correspondant à l'input
        Résultat : tf signal output (array)"""
        
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
    tf = sommetableau(xs,signal(t))
    plt.plot(t, tf, linewidth=2.0)   # Solution numérique
    plt.show()
    return tf


def UnSolve_Chua(tf2, alpha, beta) :
    """Parametres : tf2 signal à décrypter (Array)
                    alpha, beta parametres systemes (float)
        Résultat : tf3 signal output (array)"""
        
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
    tf3 = sommetableau(moins(xs),tf2)
    plt.plot(t, tf3, linewidth=2.0)   # Solution numérique 
    plt.show()
    return tf3
    
    
    
    
    
def tauxidentité(f1, f2) :
    """ Parametre : f1 et f2 resp. signaux input et output (array)
        Resultat : taux d'identité entre f1 et f2 (float)"""
    h = len(f1)
    T = []
    for x in range (0, h-1) :
        T.append((abs(f1[x]-f2[x])))#à completer il faut ici une bonne fonction d'identité d'une courbe
    sum = 0
    for k in range (0,len(T)-1) :
        sum = sum + T[k]
    return log10(1/sum)
    
    
    
    
    
    
def solve_ChuaNoEffect(alpha,beta) :
    """ Parametres : alpha, beta parametres systemes (float)
        Précondition : on suppose disposer de signal: array -> array correspondant à l'input
        Résultat : tf signal output (array)
        Effet : aucun"""
        
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
    tf = sommetableau(xs,signal(t))
    return tf
    
    
    
    
def UnSolve_ChuaNoEffect(tf2, alpha, beta) :
    """Parametres : tf2 signal à décrypter (Array)
                    alpha, beta parametres systemes (float)
        Résultat : tf3 signal output (array)
        Effet : aucun"""
        
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
    tf3 = sommetableau(moins(xs),tf2)
    return tf3
    
    
def graphId3D(alpha,beta,alphmin,alphmax,betmin,betmax) :
    """Parametres : alpha, beta parametres systemes
       Resultat : aucun
       Effet : affiche le graph 3D correspondant a une variation de (alpha, beta) du décrypteur [x, y] et du coefficient de correlation associé [z]"""
    t = linspace(0,50,1000) #précision
    
    #création des tables de données
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    A = np.arange(alphmin,alphmax,0.1) #précision
    B = np.arange(betmin,betmax, 0.1) #précision
    A, B = np.meshgrid(A, B)
    R = [[0.0 for _ in range(0,len(A))] for _ in range(0,len(A))]
    for i in range (0,len(A)) :
        for j in range (0,len(B)):
             R[i][j] = tauxidentité(signal(t),UnSolve_ChuaNoEffect(solve_ChuaNoEffect(alpha,beta),A[0][i],B[j][0]))
    
    #création de la surface
    ax.plot_wireframe(A, B, R)
    plt.show()        
    return None
    
    
    
"""def efffct1( R1, R2) :
    Pour maximiser l'efficacité du brouilleur, on souhaite que la valeur maximale du taux de corrélation soit la plus forte (et dans une deuxieme fonction que le reste des valeur soient les plus basse).
    De plus, on pose comme référence l'efficacité du filtre pour la fonction x->cos(x) de 1.
    Parametre : R1 (matrix)
                R2 (matrix)
    Résultat : e (int)
    n = len(R)
    Rmax1i = 0
    Rmax1j = 0
    Rmax2i = 0
    Rmax2j = 0
    for i in range (0, n-1) :
        for j in range (0, n-1) :
            if R1[i][j] > R1[Rmax1i][Rmax1j] :
                Rmax1i = i
                Rmax1j = j
            elif R1[i][j] > R2[Rmax2i][Rmax2j] :
                Rmax2i = i
                Rmax2j = j
    return (R1"""
                
    
def carac2(x) :
    return m1*x+0.5*m0*(abs(x+pc1)-abs(x-pc1))
    
def affcarac():
    T = linspace(-13,13,1000)
    X= linspace(-15,15,1000)
    Y = [0 for _ in range (0,1000)]
    C = []
    for k in range (0, len(T)) :
        C.append(carac2(T[k]))
    plt.plot(T, C)
    plt.plot(Y,X)
    plt.plot(X, Y)
    plt.show()
    return None
    
def sigbinaleat(amp) :
    """parametre :amp (float) amplitude du signal binaire voulu
        Résultat : + ou - amp"""
    p = random.random()
    if p >0.5 :
        return amp
    else :
        return -amp    
        
def signalaleat(t,amp) :
    """Signal Input
       Parametre : table temporelle (Array)
                    amplitude u signal désiré (float)
       Précondition : existence de function (function)
       Resultat : table de valeurs de la fonction (Array)"""
    tf = []
    for i in range(0,len(t)) :
        tf.append(sigbinaleat(amp))
    return tf 
    
def solve_Chuabinsig(alpha,beta,amp) :
    """ Parametres : alpha, beta parametres systemes (float)
                    amplitude (float)
        Précondition : on suppose disposer de signal: array -> array correspondant à l'input
        Résultat : tf signal output (array)"""
        
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
    tf = sommetableau(xs,signalaleat(t,amp))
    plt.plot(t, tf)   # Solution numérique
    plt.show()
    return tf
    
    
    
def graphId3Dcreneau(alpha,beta,alphmin,alphmax,betmin,betmax,amp) :
    """Parametres : alpha, beta parametres systemes
       Resultat : aucun
       Effet : affiche le graph 3D correspondant a une variation de (alpha, beta) du décrypteur [x, y] et du coefficient de correlation associé [z]"""
    t = linspace(0,50,1000) #précision
    
    #création des tables de données
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    A = np.arange(alphmin,alphmax,0.1) #précision
    B = np.arange(betmin,betmax, 0.1) #précision
    A, B = np.meshgrid(A, B)
    R = [[0.0 for _ in range(0,len(A))] for _ in range(0,len(A))]
    for i in range (0,len(A)) :
        for j in range (0,len(B)):
             R[i][j] = tauxidentité(signalaleat(t,amp),UnSolve_ChuaNoEffect(solve_Chuabinsig(alpha,beta,amp),A[0][i],B[j][0]))
    
    #création de la surface
    ax.plot_wireframe(A, B, R)
    plt.show()        
    return None
    
def Efficatite(alphmin,alphmax,betmin,betmax,precision) :
    """Paramatres : intervales de alpha et beta étudié (float)
                    precision a laquelle on étudie cette intervale (float)
        Résultat : Efficacité d'un crytage décryptage en fonction de alpha et beta (graph)
        Effet : aucun"""
    
    #création des tables de données
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    A = np.arange(alphmin,alphmax,precision)
    B = np.arange(betmin,betmax, precision)
    A, B = np.meshgrid(A, B)
    R = [[0.0 for _ in range(0,len(A))] for _ in range(0,len(A))]
    for i in range (0,len(A)) :
        for j in range (0,len(B)):
             R[i][j] = calceff(A[0][i],B[j][0])
             
    #création de la surface
    ax.plot_wireframe(A, B, R)
    plt.show()        
    return None
    
def calceff(alpha,beta) :
    """parametres: parametres systemes
        résultat : efficacité de la diode sur 9 points différents"""
    t = linspace(0,50,1000) #précision
    
    #création des tables de données
    A = np.arange((alpha-0.01),(alpha+0.01),0.001) #précision
    B = np.arange((beta-0.01),(beta+0.01), 0.001) #précision
    A, B = np.meshgrid(A, B)
    R = [[0.0 for _ in range(0,len(A))] for _ in range(0,len(A))]
    for i in range (0,len(A)) :
        for j in range (0,len(B)):
             R[i][j] = tauxidentité(signal(t),UnSolve_ChuaNoEffect(solve_ChuaNoEffect(alpha,beta),A[i][0],B[j][0]))
    max = R[6][6]
    min = (R[5][5]+R[5][6]+R[6][5]+R[7][5]+R[5][7]+R[7][6]+R[6][7]+R[7][7])/8
    return (max/min)
    



        
        
def signalaleat2(t,amp) :
    """Signal Input
       Parametre : table temporelle (Array)
                    amplitude u signal désiré (float)
       Précondition : existence de function (function)
       Resultat : table de valeurs de la fonction (Array)"""
    tf = [0]
    for i in range(1,len(t)) :
        p = random.random()
        n = random.randrange(1, 100)
        if p >0.5 :
            res = tf[i-1]+(amp/n)
            if res > amp :
                tf.append(tf[i-1])
            else :
                tf.append(res)
        else :
            res = tf[i-1]-(amp/n)
            if res < -amp :
                tf.append(tf[i-1])
            else :
                tf.append(res)
    return tf 
        
        
def solve_Chuasig2(alpha,beta,amp) :
    """ Parametres : alpha, beta parametres systemes (float)
                    amplitude (float)
        Précondition : on suppose disposer de signal: array -> array correspondant à l'input
        Résultat : tf signal output (array)"""
        
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
    tf = sommetableau(xs,signalaleat2(t,amp))
    plt.plot(t, tf)   # Solution numérique
    plt.show()
    return tf
    