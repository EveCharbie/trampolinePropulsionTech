"""Contact entre sauteur et trampoline, en considérant seulement 6 degrés de liberté
    on se limite ici a la premiere phase : l'enfoncement du sauteur dans la toile,

"""

import numpy as np
import biorbd
import matplotlib.pyplot as plt
from IPython import embed

#import biorbd_casadi as biorbd
import casadi as cas
from bioptim import (
    DynamicsFunctions,
    ConfigureProblem,
    NonLinearProgram,
    OptimalControlProgram,
)

model = biorbd.Model('/home/lim/Documents/Jules/cube.bioMod')
model1 = biorbd.Model('/home/lim/Documents/Jules/cube_nocontact.bioMod')

q = np.array([1, 2, 4]) # np.random.rand(3,1)
qdot = np.array([1, 2, 6]) # np.random.rand(3,1)
tau = np.array([1, 3, 4]) # np.random.rand(3,1)

# q = np.array([0, 0, 0]) # np.random.rand(3,1)
# qdot = np.array([0, 0, 0]) # np.random.rand(3,1)
# tau = np.array([0, 0, 0]) # np.random.rand(3,1)

############### f contact  ######################
Force = np.zeros((2, 1))
Force[0] = 10
Force[1] = 500
Force = np.reshape(Force, (2,))

count = 0
f_contact_vec = biorbd.VecBiorbdVector()

for ii in range(1):
    n_f_contact = 2
    idx = [i + count for i in range(n_f_contact)]
    f_contact_vec.append(Force[idx])
    count = count + n_f_contact

############test f ext#############################
force_vector = np.zeros((6, 1))

force_vector[4] = 10
force_vector[5] = 500
force_vector[0] = q[0]*force_vector[5] - q[1]*force_vector[4]
f_ext = biorbd.to_spatial_vector(force_vector)
###################################################

q = np.reshape(q, (3,))
qdot = np.reshape(qdot, (3,))
tau = np.reshape(qdot, (3,))
q = biorbd.GeneralizedCoordinates(q)
qdot = biorbd.GeneralizedVelocity(qdot)
tau = biorbd.GeneralizedTorque(tau)

qddotext = model1.ForwardDynamics(q, qdot, tau, f_ext)
qddotcontact = model.ForwardDynamics(q, qdot, tau, None, f_contact_vec)
qddotext = qddotext.to_array()
qddotcontact = qddotcontact.to_array()
print(qddotcontact)
print(qddotext)









#avec casadi#
# ygrid = np.linspace(-0.5, 0.5, 100)  # Devant-derriere
# zgrid = np.linspace(-1.2, 0, 100)  # Bas
#
# Force_verticale = np.load("/home/lim/Documents/Jules/code_initiaux_Eve/Force_verticale_full.npy")
# Force_horizontale = np.load("/home/lim/Documents/Jules/code_initiaux_Eve/Force_horizontale_full.npy")
#
# Force_verticale *= -1
# Force_horizontale *= -1
#
# data_flat_verticale = Force_verticale.ravel(order="F")
# lut_verticale = cas.interpolant("name", "bspline", [ygrid, zgrid], data_flat_verticale)
#
# data_flat_horizontale = Force_horizontale.ravel(order="F")
# lut_horizontale = cas.interpolant("name", "bspline", [ygrid, zgrid], data_flat_horizontale)
#
# def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):
#     ConfigureProblem.configure_q(nlp, as_states=True, as_controls=False)
#     ConfigureProblem.configure_qdot(nlp, as_states=True, as_controls=False)
#     ConfigureProblem.configure_tau(nlp, as_states=False, as_controls=True)
#     ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic, expand=False)
#
# def custom_dynamic(states, controls, nlp):
#     q = DynamicsFunctions.get(nlp.states["q"], states)
#     qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
#     tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
#
#     Markers = nlp.model.markers(q)
#     Marker_pied = Markers[0].to_mx()
#
#    # ############### f contact  ######################
#     Force = cas.MX.zeros(3)
#     Force[1] = lut_horizontale(Marker_pied[1:])
#     Force[2] = lut_verticale(Marker_pied[1:])
#
#     count = 0
#     f_contact_vec = biorbd.VecBiorbdVector()
#
#     for ii in range(nlp.model.nbRigidContacts()):
#         n_f_contact = len(nlp.model.rigidContactAxisIdx(ii))
#         idx = [i + count for i in range(n_f_contact)]
#         f_contact_vec.append(Force[idx])
#         count = count + n_f_contact
#
#     qddot1 = nlp.model.ForwardDynamics(q, qdot, tau, None, f_contact_vec).to_mx()
#
#     f_evalcontact = cas.Function("qddot1", [q ,qdot,tau, None, f_contact_vec],[qddot1])
#
#    # #########################test f ext###############
#
#     force_vector = cas.MX.zeros(6)
#     force_vector[4] = lut_horizontale(Marker_pied[1:]) #dans le global
#     force_vector[5] = lut_verticale(Marker_pied[1:])  #dans le global
#
#     force_vector[0] = q[0]*force_vector[5] - q[1]*force_vector[4]
#
#     f_ext = biorbd.VecBiorbdSpatialVector()
#     f_ext.append(biorbd.SpatialVector(force_vector))
#     qddot = nlp.model.ForwardDynamics(q, qdot, tau, f_ext).to_mx()
#
#     f_evalext = cas.Function("qddot", [q ,qdot,tau,f_ext],[qddot])
#
#     return f_evalcontact, f_evalext


