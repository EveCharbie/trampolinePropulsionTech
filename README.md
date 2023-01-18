# trampolinePropulsionTech

Optimisation des mouvements du sauteur
Codes simplifiés sur 6 deg de liberté et pour un nombre réduit de phase
- modele 1 phase
- modele 2 phases, avec :
	- le code basique
	- avec les actuators en contrainte
	- avec les actuators dans la dynamique
- modele 3 phases, avec :
	- le code basique
	- avec une phase aerienne SANS salto (juste un saut)
	- ajout d'un mapping sur les tau
	
le dossier comprend aussi :
- des tests sur forward dynamics, pour la différence entre F_ext et F_contact
- un modele de cube tres simplifier sur 1 ou 2 phase
- un code pour afficher les videos bioviz et les valeurs des look up tables



Dernier code : Trampo_sauteur_6deg _3phases_sanssalto.py -> avec le bon mapping, mais ne converge pas
