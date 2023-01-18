"""Contact entre sauteur et trampoline, en considérant seulement 6 degrés de liberté
    on se limite ici au 3 premiere phase : l'enfoncement du sauteur dans la toile,
                                            la phase de propulsion du sauteur pendant laquelle la toile restitue l'énergie
                                            le phase dans les airs avec réalisation d'un salto
"""

from time import time
import numpy as np
import casadi as cas
import biorbd_casadi as biorbd
import os
import matplotlib.pyplot as plt
from datetime import date
import pickle

# import sys
# sys.path.append("/home/lim/Documents/Jules/bioptim")

from bioptim import (
    Node,
    OptimalControlProgram,
    ConstraintList,
    ConstraintFcn,
    ObjectiveFcn,
    ObjectiveList,
    Dynamics,
    DynamicsList,
    DynamicsFunctions,
    DynamicsEvaluation,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    InitialGuess,
    NoisedInitialGuess,
    Axis,
    PenaltyNode,
    InterpolationType,
    Bounds,
    BiMapping,
    Solver,
    BiorbdInterface,
    NonLinearProgram,
    ConfigureProblem,
    BiMappingList
)

def custom_dynamic(states, controls, parameters, nlp):
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau_activations = DynamicsFunctions.get(nlp.controls["tau"], controls)
    tau = nlp.model.torque(tau_activations, q, qdot).to_mx()

    Markers = nlp.model.markers(q)
    Marker_pied = Markers[0].to_mx()

    Force = cas.MX.zeros(2)
    Force[0] = lut_horizontale(Marker_pied[1:])
    Force[1] = lut_verticale(Marker_pied[1:])
    count = 0
    f_contact_vec = biorbd.VecBiorbdVector()

    for ii in range(nlp.model.nbRigidContacts()):
        n_f_contact = len(nlp.model.rigidContactAxisIdx(ii))
        idx = [i + count for i in range(n_f_contact)]
        f_contact_vec.append(Force[idx])
        count = count + n_f_contact

    qddot = nlp.model.ForwardDynamics(q, qdot, tau, None, f_contact_vec).to_mx()

    return DynamicsEvaluation(dxdt=cas.vertcat(qdot, qddot), defects=None)

def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):

    ConfigureProblem.configure_q(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic, expand=False)


def prepare_ocp_back_back(path_model_cheville, lut_verticale, lut_horizontale, weight, Salto1, Salto2):
    # --- Options --- #
    model_path = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_6DoF.bioMod"
    model_path_massToile = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_6DoF_massToile.bioMod"

    # Model path
    biorbd_model = (
        biorbd.Model(model_path_massToile),
        biorbd.Model(model_path_massToile),
    )

    nb_phases = len(biorbd_model)
    nq = biorbd_model[0].nbQ()
    nqdot = biorbd_model[0].nbQdot()

    number_shooting_points = (
        50,
        50,
    )

    final_time = (
        0.15,
        0.15,
    )

    tau_min, tau_max, tau_init = -1, 1, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", node=Node.END, index=1, weight=1000, phase=0, quadratic=False)

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=1)

    # arriver avec les pieds au centre de la toile
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", phase=0, node=Node.START, index=0, weight=100, target=np.zeros((1, 1)))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", phase=0, node=Node.START, index=1, weight=100, target=np.zeros((1, 1)))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", phase=1, node=Node.END, index=0, weight=100, target=np.zeros((1, 1)))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", phase=1, node=Node.END, index=1, weight=100, target=np.zeros((1, 1)))

    #maximiser la vitesse de remonter
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, phase=1, node=Node.END, weight=-100000, quadratic=False, axes=Axis.Z)

    # # Dynamics
    dynamics = DynamicsList()
    dynamics.add(Dynamics(custom_configure, dynamic_function=custom_dynamic))
    dynamics.add(Dynamics(custom_configure, dynamic_function=custom_dynamic))

    # --- Constraints --- #
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_COM_VELOCITY, node=Node.START, min_bound=-20, max_bound=-5, phase=0, axes=Axis.Z)
    #
    # # Constraint time
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.08, max_bound=0.6, phase=0)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.08, max_bound=0.6, phase=1)

    # Path constraint
    X_bounds = BoundsList()

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[0])) #phase 0
    #q
    X_bounds[0].min[0, :] = [-0.3, -0.3, -0.3]
    X_bounds[0].max[0, :] = [0.3, 0.3, 0.3]
    X_bounds[0].min[1, :] = [0, -1.2, -1.2]
    X_bounds[0].max[1, :] = [0, 0, 0]
    X_bounds[0].min[2, :] = [0, -0.5, -0.5]
    X_bounds[0].max[2, :] = [0, 0.5, 0.5]
    X_bounds[0].min[3, :] = [1.4, -0.5, -0.5]
    X_bounds[0].max[3, :] = [1.4, 2.5, 2.5]
    X_bounds[0].min[4, :] = [1, -1.5, -1.5]
    X_bounds[0].max[4, :] = [1, 1.5, 1.5]
    X_bounds[0].min[5, :] = [1, -0.5, -0.5]
    X_bounds[0].max[5, :] = [1, 3, 3]
    #qdot
    X_bounds[0].min[6, :] = [-50, -50, -50]
    X_bounds[0].max[6, :] = [50, 50, 50]
    X_bounds[0].min[7, :] = [-30, -30, -30]
    X_bounds[0].max[7, :] = [-5, 0, 0]
    X_bounds[0].min[8, :] = [-50, -50, -50]
    X_bounds[0].max[8, :] = [50, 50, 50]
    X_bounds[0].min[9, :] = [-50, -50, -50]
    X_bounds[0].max[9, :] = [50, 50, 50]
    X_bounds[0].min[10, :] = [-50, -50, -50]
    X_bounds[0].max[10, :] = [50, 50, 50]
    X_bounds[0].min[11, :] = [-50, -50, -50]
    X_bounds[0].max[11, :] = [50, 50, 50]


    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[1])) #phase 1
    # q
    X_bounds[1].min[0, :] = [-0.3, -0.3, -0.3]
    X_bounds[1].max[0, :] = [0.3, 0.3, 0.3]
    X_bounds[1].min[1, :] = [-1.2, -1.2, 0]
    X_bounds[1].max[1, :] = [0, 0, 0.2]
    X_bounds[1].min[2, :] = [-0.5, -0.5, -0.5]
    X_bounds[1].max[2, :] = [0.5, 0.5, 0.5]
    X_bounds[1].min[3, :] = [-0.5, -0.5, -0.5]
    X_bounds[1].max[3, :] = [2.5, 2.5, 2.5]
    X_bounds[1].min[4, :] = [-1.5, -1.5, -1.5]
    X_bounds[1].max[4, :] = [1.5, 1.5, 1.5]
    X_bounds[1].min[5, :] = [-0.5, -0.5, -0.5]
    X_bounds[1].max[5, :] = [3, 3, 3]
    # qdot
    X_bounds[1].min[6, :] = [-50, -50, -50]
    X_bounds[1].max[6, :] = [50, 50, 50]
    X_bounds[1].min[7, :] = [-1, 0, 0]
    X_bounds[1].max[7, :] = [30, 30, 30]
    X_bounds[1].min[8, :] = [-50, -50, -50]
    X_bounds[1].max[8, :] = [50, 50, 50]
    X_bounds[1].min[9, :] = [-50, -50, -50]
    X_bounds[1].max[9, :] = [50, 50, 50]
    X_bounds[1].min[10, :] = [-50, -50, -50]
    X_bounds[1].max[10, :] = [50, 50, 50]
    X_bounds[1].min[11, :] = [-50, -50, -50]
    X_bounds[1].max[11, :] = [50, 50, 50]


    u_bounds = BoundsList()
    u_bounds.add(
        bounds=Bounds(
            [-1, -1, -1, -1], [1, 1, 1, 1]
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [-1, -1, -1, -1], [1, 1, 1, 1]
        )
    )

    x_init = InitialGuessList()
    # x_init.add([0]*12)
    # x_init.add([0]*12)
    x_init.add(NoisedInitialGuess(
        [0] * 12,  # (nq + nqdot)
        bounds=X_bounds[0],
        noise_magnitude=0.01,
        n_shooting=number_shooting_points[0],
        bound_push=0.01,
        seed=i_rand,
    )
    )
    x_init.add(NoisedInitialGuess(
        [0] * 12,  # (nq + nqdot)
        bounds=X_bounds[1],
        noise_magnitude=0.01,
        n_shooting=number_shooting_points[1],
        bound_push=0.1,
        seed=i_rand,
    )
    )

    u_init = InitialGuessList()
    # u_init.add([0] * 4)
    # u_init.add([0] * 4)
    u_init.add(NoisedInitialGuess(
        [0] * 4,  # ntorque
        bounds=u_bounds[0],
        noise_magnitude=0.01,
        n_shooting=number_shooting_points[0] - 1,
        bound_push=0.1,
        seed=i_rand,
    ))

    u_init.add(NoisedInitialGuess(
        [0] * 4,
        bounds=u_bounds[1],
        noise_magnitude=0.01,
        n_shooting=number_shooting_points[1] - 1,
        bound_push=0.1,
        seed=i_rand,
    ))



    variable_mappings = BiMappingList()
    variable_mappings.add("tau", to_second=[None, None, 0, 1, 2, 3], to_first=[0, 1, 2, 3])


    ocp = OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init=x_init,
        x_bounds=X_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        n_threads=3,
        variable_mappings=variable_mappings,
    )
    return ocp


if __name__ == "__main__":

    Salto_1 = np.array([1])
    Salto_2 = np.array([1])

    path_model_cheville = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/Cheville.bioMod"

    ygrid = np.linspace(-0.5, 0.5, 100)  # Devant-derriere
    zgrid = np.linspace(-1.2, 0, 100)  # Bas

    Force_verticale = np.load("/home/lim/Documents/Jules/code_initiaux_Eve/Force_verticale_full.npy")
    Force_horizontale = np.load("/home/lim/Documents/Jules/code_initiaux_Eve/Force_horizontale_full.npy")

    Force_verticale *= -1
    Force_horizontale *= -1

    data_flat_verticale = Force_verticale.ravel(order="F")
    lut_verticale = cas.interpolant("name", "bspline", [ygrid, zgrid], data_flat_verticale)

    data_flat_horizontale = Force_horizontale.ravel(order="F")
    lut_horizontale = cas.interpolant("name", "bspline", [ygrid, zgrid], data_flat_horizontale)

    # **************** Masse de la toile au pieds du modele ********************************************************
    # initialisation du parametre temps #######

    Salto1 = 1
    Salto2 = 1
    weight = 1000
    i_rand = 10


    tic = time()
    ocp = prepare_ocp_back_back(path_model_cheville=path_model_cheville,lut_verticale=lut_verticale,lut_horizontale=lut_horizontale,weight=weight,Salto1=Salto1,Salto2=Salto2,)
    # ocp.check_conditioning()

    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(10000)
    solver.set_tol(1e-3)
    solver.set_constr_viol_tol(1e-3)
    solver.set_linear_solver("ma57")
    sol = ocp.solve(solver)

    toc = time() - tic
    print(f"Time to solve weight={weight}, random={i_rand}: {toc}sec")

    q = ocp.nlp[0].variable_mappings["q"].to_second.map(sol.states[0]["q"])
    qdot = ocp.nlp[0].variable_mappings["qdot"].to_second.map(sol.states[0]["qdot"])
    u = ocp.nlp[0].variable_mappings["tau"].to_second.map(sol.controls[0]["tau"])
    t = sol.parameters["time"]
    for i in range(1, len(sol.states)):
        q = np.hstack((q, ocp.nlp[i].variable_mappings["q"].to_second.map(sol.states[i]["q"])))
        qdot = np.hstack((qdot, ocp.nlp[i].variable_mappings["qdot"].to_second.map(sol.states[i]["qdot"])))
        u_act = sol.controls[i]["tau"]
        u_nul = np.zeros((2,51))
        u_ok = np.vstack((u_nul, u_act))
        u = np.hstack((u, ocp.nlp[i].variable_mappings["q"].to_second.map(u_ok)))

#####################################################################################################################
    import bioviz

    model_path = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_6DoF.bioMod"

    path = '/home/lim/Documents/Jules/result_saut/' + 'phase01_sauteur_tauDYN2' + '.pkl'
    with open(path, 'wb') as file:
        pickle.dump(q, file)
        pickle.dump(qdot, file)
        pickle.dump(u, file)
        pickle.dump(t, file)
        pickle.dump(sol.time, file)
        pickle.dump(sol.phase_time, file)

    b = bioviz.Viz(model_path)
    b.load_movement(q)
    b.exec()
