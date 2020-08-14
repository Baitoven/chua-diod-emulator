import matplotlib.pyplot as plt
import math as math

R9 = 100000
R10 = 100000
R11 = 100000
R12 = 100000
R21 = 100000
R22 = 100000
R23 = 100000
R24 = 100000



def crypt(f) :
    """Parametre : signal a crypter (function)
        Resultat : signal crypé par sommation entre s et x (function)"""
    return (1 + (R12/R11)) * ((R9/(R9+R10)) * f() + (R10/(R9+R10)) * x())
    
def x() :
    """Diode de Chua"""
    return #à compléter
    
def s() :
    """Signal Input"""
    return #à choisir
    
    
def décrypt(f) :
    """Parametre : signal a décrypter (function)
        Resultat : singal décrypté (function)"""
    return (1 + (R24/R21)) * ((R23/(R23+R22)) * f() - (R24/(R21+R24)) * x())
    
    
def tauxidentité(f1, f2) :
    """ Parametre : f1 et f2 resp. signaux input et output (function)
        Resultat : pourcentage d'identité entre f1 et f2 (float)"""
    n = ??? #largeur de l'intervalle étudié, à modifier à volonté
    h = 1000 #pas
    T = []
    for t in range (0, h -1) :
        u = ((t/h) * n)
        T.append(abs(f1(u)-f2(u)))
    sum = 0
    for k in range (0,len(T) -1) :
        sum = sum + T[k]
    return (1 - (sum/h))
    
