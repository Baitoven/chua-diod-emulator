def min(a,b):
    """Paramètres : (entiers) a et b
    Résultat : (entier) le plus petit entier des deux"""
    if a < b:
        return a
    else :
        return b

def incrementer_compteur(ligne, colonne, plateau, compteur):
    """Paramètres : (entier de 0 à 5) ligne
                    (entier de 0 à 6) colonne
                    (plateau) plateau
    Résultat : (entier entre -4 et 4) l'incrémentation du compteur"""
    joueur = plateau[ligne][colonne]
    print("joueur actu : ", joueur)
    
    if compteur == 0:
        return joueur
    elif compteur * joueur <= 0:
        return joueur
    else :
        return compteur + joueur

def tester_victoire(ligne, colonne, plateau):
    """Paramètres : (entier de 0 à 5) ligne
                    (entier de 0 à 6) colonne
                    (plateau) plateau
                    (les coordonnées du jeton fraichement placé
    Résultat : (booléen) True si un joueur a gagné False sinon"""
    
    joueur = plateau[ligne][colonne]
    compteur = 0
    
    #horizontale
    for k in range(0,7):
        compteur = incrementer_compteur(ligne, k, plateau, compteur)
        if compteur == 4 or compteur == -4:
            return True
    
    compteur = 0
    #vertical
    for k in range(0,6):
        compteur = incrementer_compteur(k, colonne, plateau, compteur)
        if compteur == 4 or compteur == -4:
            return True

    compteur = 0
    #diagonale D1
    a = min(ligne, colonne)
    l1 = ligne - a #point de départ : en haut à gauche de la diagonale avant de descendre (on calcule les coordonnées de ce pt)
    c1 = colonne - a
    b = min(6-l1, 7-c1)
    for k in range(0,b):
        compteur = incrementer_compteur(l1 + k, c1 + k, plateau, compteur)
        if compteur == 4 or compteur == -4:
            return True
            
    compteur = 0
    #diagonale D2
    a = min(ligne, 6-colonne)
    l1 = ligne - a
    c1 = colonne + a
    b = min(6-l1, c1+1)
    for k in range(0,b):
        compteur = incrementer_compteur(l1 + k, c1 - k, plateau, compteur)
        print("l :", l1 + k, ", c :", c1 - k, ", compteur :", compteur)
        if compteur == 4 or compteur == -4:
            return True
    
    
    return False

def placer_jeton(colonne, plateau, joueur):
    """Paramètres : C entier de 0 à 6 (la colonne), P le plateau, J le joueur (1 ou -1)
    Sortie : Le plateau modifié """
    if plateau[0][colonne] != 0 :
        return "impossible"
    ligne = 5
    while plateau[ligne][colonne] != 0 :
        ligne = ligne-1
    plateau[ligne][colonne] = joueur
    print("Joueur ", joueur, " gagne :" ,tester_victoire(ligne, colonne, plateau))
    
def remplir_a_la_main(plateau, nombre): #pour pouvoir remplir comme on veut pour faire des tests
    """Paramètres : (plateau) plateau
                    (naturel) nombre
    Résultat : aucun
    Effet : Permet de placer nombre jetons dans le plateau à la main
    Remarque : taper "stop" permet de stopper la fonction"""
    Txt = ""
    colonne = 0
    joueur = 1
    for i in range(0, nombre):
        joueur *= -1
        Txt = input("colonne : ")
        if Txt == "stop":
            return None
        colonne = int(Txt)
        placer_jeton(colonne,plateau, joueur)
        aff_plateau(plateau)
        
