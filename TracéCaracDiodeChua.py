import numpy as np




def recup_données(adresse_fichier) :
    #Lecture d'un fichier .txt
    T = np.loadtxt(adresse_fichier, dtype=str)
    return convertstrtofloat(T)
    
def convertstrtofloat(t) :
    #On enleve transforme tous les str du tableau ainsi généré
    n = len(t)
    T2 = []
    for i in range (0,n-1) :
        a = str.split(t[i],sep='"')
        for j in range (0, len(a)-1) :
            T2.append(a[j])
    return T2

def enleveb(t) :
    T3 = []
    for i in range (0, len(t)-1) :
        if T[i] != "b'" :
            f = str.split(T[i], sep=',')
            T3.append(f[0]+'.'+f[1])
    return T3
    
#'C:\\Users\Adrien\Desktop\TIPE\CaracDiodeChua.txt' 