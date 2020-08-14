def carac(x) :
    if -12.5<=x<=-10.2 :
        return 2.07*x+26.00
    elif -10.2<x<=-1.7 :
        return -0.41*x+0.67
    elif -1.7<x<=1.8 :
        return -0.79*x
    elif 1.8<x<=10.5 :
        return -0.42*x-0.65
    elif 10.5<x<=12.5 :
        return 2.47*x+-31.48
 
(R,C1,C2,L) = (1625,10**(-8),10**(-7),1.9*10**(-2))
    
def equadif(valeur, t) :
    return([-(v[2]/L), ((v[2]-v[1])/(R*C1))+(carac(v[1])/C1), ((v[1]-v[2])/(R*C2))+(v[0]/C2)])
    
def orbite(condinit,n,T) :
    import numpy as np
    import mathplotlib.pyplot as pypl
    t = np.linspace(0,T,n+1)
    values = scipy.integrate.odeint(equadif, condinit,t)
    pypl.plot
    return None


    