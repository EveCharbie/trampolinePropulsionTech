"""
Jules Olivié - 21/10/2022
version biorbd : 1.9.0
Test pour comparaison entre f ext et f contact de forwarddynamics


Le cas suivant prend comme modele une masse ponctuelle

Le code renvoie les qddot calculés par forwarddynamics dans 2 cas :
- d'abord dans le cas ou on lui donne une force de contact
-puis dans le cas ou on lui donne une force externe et ou on calcule les moments a la main

/!\ deux biomod sont utilisés : bien vérifier que les masses sont identiques
"""

import numpy as np
import biorbd
import matplotlib.pyplot as plt
from IPython import embed

model = biorbd.Model('/home/lim/Documents/Jules/cubesixdeg.bioMod')
model1 = biorbd.Model('/home/lim/Documents/Jules/cubesixdeg_nocontact.bioMod')


q = np.array([0, 1, 1, 0, 0, 0]) # np.random.rand(3,1)
qdot = np.array([0, 1, 1, 0, 0, 0]) # np.random.rand(3,1)
tau = np.array([0, 1, 1, 0, 0, 0]) # np.random.rand(3,1)

############### f contact  ######################
Force = np.zeros((2, 1))
Force[0] = 10    #contact sur y
Force[1] = 500    #contact sur z

Force = np.reshape(Force, (2,))

count = 0
f_contact_vec = biorbd.VecBiorbdVector()

for ii in range(1):
    n_f_contact = 2
    idx = [i + count for i in range(n_f_contact)]
    f_contact_vec.append(Force[idx])
    count = count + n_f_contact

#########################test f ext###############
force_vector = np.zeros((6, 1))
#FORCE
force_vector[3] = 0
force_vector[4] = 10
force_vector[5] = 500
#MOMENT
force_vector[0] = q[1]*force_vector[5] - q[2]*force_vector[4]
force_vector[1] = q[2]*force_vector[3] - q[0]*force_vector[5]
force_vector[2] = q[0]*force_vector[4] - q[1]*force_vector[3]

f_ext = biorbd.to_spatial_vector(force_vector)



q = np.reshape(q, (6,))
qdot = np.reshape(qdot, (6,))
tau = np.reshape(qdot, (6,))
q = biorbd.GeneralizedCoordinates(q)
qdot = biorbd.GeneralizedVelocity(qdot)
tau = biorbd.GeneralizedTorque(tau)

qddotext = model1.ForwardDynamics(q, qdot, tau, f_ext)
qddotcontact = model.ForwardDynamics(q, qdot, tau, None, f_contact_vec)
qddotext = qddotext.to_array()
qddotcontact = qddotcontact.to_array()
print(qddotcontact)
print(qddotext)