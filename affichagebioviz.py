import bioviz
import casadi as cas
import pickle
import matplotlib.pyplot as plt
import numpy as np

model_path = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_6DoF.bioMod"

path = '/home/mickael/Documents/Jules/propulsion/result/phase012_sanssalto_novinit_mapping5.pkl'

with open(path, 'rb') as file:
    q= pickle.load(file)
    qdot = pickle.load(file)
    u = pickle.load(file)
    t = pickle.load(file)


ygrid = np.linspace(-0.5, 0.5, 100)  # Devant-derriere
zgrid = np.linspace(-1.2, 0, 100)  # Bas

Force_verticale = np.load("/home/mickael/Documents/Jules/propulsion/collecte/Force_verticale_full.npy")
Force_horizontale = np.load("/home/mickael/Documents/Jules/propulsion/collecte/Force_horizontale_full.npy")

Force_verticale *= -1
Force_horizontale *= -1

data_flat_verticale = Force_verticale.ravel(order="F")
lut_verticale = cas.interpolant("name", "bspline", [ygrid, zgrid], data_flat_verticale)

data_flat_horizontale = Force_horizontale.ravel(order="F")
lut_horizontale = cas.interpolant("name", "bspline", [ygrid, zgrid], data_flat_horizontale)

q_y = q[0]
q_z = q[1]

val_lut_vert_appliquee = []
val_lut_horz_appliquee = []

for i in range(153):
    val_lut_vert_appliquee.append(lut_verticale([q_y[i],q_z[i]]).toarray()[0][0])
    val_lut_horz_appliquee.append(lut_horizontale([q_y[i],q_z[i]]).toarray()[0][0])

node = np.linspace(0,152,153)
plt.plot(node,val_lut_vert_appliquee, 'r-', label='force verticale')
plt.plot(node,val_lut_horz_appliquee, 'b-', label ='force horizontale')
plt.axvline(x = 51, color='black', linestyle='dashdot')
plt.axvline(x = 51*2, color='black', linestyle='dashdot')
plt.xlabel('Noeuds', fontsize=15)
plt.ylabel('Force (N)',fontsize=15)
plt.title('Force de la toile sur l\'athlete',fontsize=15)

plt.legend()
plt.show()


b = bioviz.Viz(model_path)
b.load_movement(q)
b.exec()