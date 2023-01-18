"""
Jules Olivié - 21/10/2022
version biorbd : 1.9.0
This example is intended to guide you through the implementation of an external force with ForwardDynamics.
It shows 2 different ways to define the external force applied to the contact of the system :
-the first using a force applied at the center of mass, defining all the moment created at the contact point
-the second using directly a force applied at the point of contact.

The two options must return the same results.
"""

import numpy as np
import biorbd

"""Define the parameters used in this example"""

q = np.array([0, 1, 1, 0, 0, 0]) # np.random.rand(3,1)
qdot = np.array([0, 1, 1, 0, 0, 0]) # np.random.rand(3,1)
tau = np.array([0, 1, 1, 0, 0, 0]) # np.random.rand(3,1)

X_Force = 0
Y_Force = 10
Z_Force = 500

def Contact_Force():
    """
    The force defined here is a two-dimensional force applied on a unique contact point
    --------------
    Force_contact : np.array

    :return:
    f_contact : biorbd.VecBiorbdVector

    """
    Force_contact = np.zeros((2, 1))
    Force_contact[0] = Y_Force    # Y contact
    Force_contact[1] = Z_Force    # Z contact

    Force = np.reshape(Force_contact, (2,))
    f_contact = biorbd.VecBiorbdVector()
    count = 0
    for ii in range(1):
        n_f_contact = 2
        idx = [i + count for i in range(n_f_contact)]
        f_contact.append(Force[idx])
        count = count + n_f_contact

    return f_contact

def External_Force():
    """

    :return:
    f_ext : biorbd.VecBiorbdSpacialVector
    """

    force_vector = np.zeros((6, 1))
    #FORCE
    force_vector[3] = X_Force
    force_vector[4] = Y_Force
    force_vector[5] = Z_Force
    #TORQUE
    force_vector[0] = q[1]*force_vector[5] - q[2]*force_vector[4]
    force_vector[1] = q[2]*force_vector[3] - q[0]*force_vector[5]
    force_vector[2] = q[0]*force_vector[4] - q[1]*force_vector[3]

    f_ext = biorbd.to_spatial_vector(force_vector)

    return f_ext

def Prepare_Dynamics(q, qdot, tau):
    """

    :param q:
    :param qdot:
    :param tau:

    :return:
    biorbd.GeneralizedCoordina
    biorbd.GeneralizedVelocitytes
    biorbd.GeneralizedTorque

    """

    q = np.reshape(q, (6,))
    qdot = np.reshape(qdot, (6,))
    tau = np.reshape(tau, (6,))

    q = biorbd.GeneralizedCoordinates(q)
    qdot = biorbd.GeneralizedVelocity(qdot)
    tau = biorbd.GeneralizedTorque(tau)

    return q, qdot, tau

def Qddot(q, qdot, tau):
    """

    :param q:
    :param qdot:
    :param tau:

    :return: Accelerations calculated with Forward.Dynamics
    """

    model_contact = biorbd.Model('/home/lim/Documents/Jules/cubesixdeg.bioMod')
    model_ext = biorbd.Model('/home/lim/Documents/Jules/cubesixdeg_nocontact.bioMod')

    q_f = Prepare_Dynamics(q, qdot, tau)[0]
    qdot_f = Prepare_Dynamics(q, qdot, tau)[1]
    tau_f = Prepare_Dynamics(q, qdot, tau)[2]

    f_ext = External_Force()
    f_contact = Contact_Force()

    qddotext = model_ext.ForwardDynamics(q_f, qdot_f, tau_f, f_ext)
    qddotcontact = model_contact.ForwardDynamics(q_f, qdot_f, tau_f, None, f_contact)

    qddotext = qddotext.to_array()
    qddotcontact = qddotcontact.to_array()

    return qddotext, qddotcontact

def main():
    """
    print the results
    """
    print('Qddot calculed with external force : ' + str(Qddot(q, qdot, tau)[0]))
    print('Qddot calculed with contact force : ' + str(Qddot(q, qdot, tau)[1]))

if __name__ == "__main__":
    main()