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
    # tau_activations_pos = DynamicsFunctions.get(nlp.controls["tau_pos"], controls)
    # tau_activations_neg = DynamicsFunctions.get(nlp.controls["tau_neg"], controls)
    #
    # tau_pos = nlp.model.torque(tau_activations_pos, q, qdot).to_mx() #de taile 6
    # tau_neg = nlp.model.torque(tau_activations_neg, q, qdot).to_mx() #de taile 6
    #
    # tau = tau_pos + tau_neg
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

    # ConfigureProblem.configure_new_variable('tau_pos',
    #                                         ['Tau_trans_y_pos', 'Tau_trans_z_pos', 'Jambe_rot_x_pos', 'Cuisse_rot_x_pos', 'Pelvis_rot_x_pos', 'Bras_rot_x_pos'],
    #                                         nlp,
    #                                         as_states= False,
    #                                         as_controls=True)
    # ConfigureProblem.configure_new_variable('tau_neg',
    #                                         ['Tau_trans_y_neg', 'Tau_trans_z_neg', 'Jambe_rot_x_neg', 'Cuisse_rot_x_neg', 'Pelvis_rot_x_neg', 'Bras_rot_x_neg'],
    #                                         nlp,
    #                                         as_states= False,
    #                                         as_controls=True)


    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic, expand=False)


def CoM_base_appui(pn: PenaltyNode) -> cas.MX:#centre de masse au dessus de la point de contatc avec la toile, pour rester debout, a ajouter dans contrainte
    val_contrainte = []
    nq = int(pn.nlp.cx / 2)
    q_i = pn.nlp.phase_mapping["q"].to_second.map(pn.x[:nq])  # pn.x[i][:nq]
    CoM = pn.nlp.model.CoM(q_i).to_mx()
    q_pied_y = q_i[0]
    CoM_proj = CoM[1]
    val_contrainte = cas.vertcat(val_contrainte, CoM_proj - q_pied_y)
    return val_contrainte


def Non_trans_toile(pn: PenaltyNode) -> cas.MX:#maraueurs du pied et de la hanche toujours au dessus du pied
    val_contrainte = []
    nq = int(pn.nlp.cx / 2)
    q_i = pn.nlp.phase_mapping["q"].to_second.map(pn.x[:nq])  # pn.x[i][:nq]
    Marker_pied = pn.nlp.model.markers(q_i)[0].to_mx()[2]
    Marker_genou = pn.nlp.model.markers(q_i)[1].to_mx()[2]
    Marker_hanche = pn.nlp.model.markers(q_i)[2].to_mx()[2]

    val_contrainte = cas.vertcat(val_contrainte, Marker_genou - Marker_pied)
    val_contrainte = cas.vertcat(val_contrainte, Marker_hanche - Marker_pied)

    return val_contrainte


def custom_spring_const_post(Q, lut_verticale, lut_horizontale, model_path): #calcul de la force de la toile sur la cheville apres optim pour pouvoir comparer
    m = biorbd.Model(model_path)
    Marker_pied = m.markers(Q)[0].to_mx()
    Force = cas.MX.zeros(3)
    Force[1] = lut_horizontale(Marker_pied[1:])
    Force[2] = lut_verticale(Marker_pied[1:])
    fun = cas.Function("Force_TrampoBed", [Q], [Marker_pied, Force])
    return fun


def q_cheville_func(Q):
    q_mod = cas.fmod(Q, 2 * np.pi)
    q_out = cas.if_else(
        q_mod < 0,
        cas.if_else(q_mod < -np.pi, -(-2 * np.pi - q_mod), -q_mod),
        cas.if_else(q_mod > np.pi, -(2 * np.pi - q_mod), -q_mod),
    )
    fun = cas.Function("q_cheville", [Q], [q_out])
    return fun

# def tau_actuator_constraints_min(pn: PenaltyNode, path_model_cheville: str, minimal_tau: float = None) -> cas.MX:
#     model_cheville = biorbd.Model(path_model_cheville)
#     q_cheville = cas.MX.sym("q_cheville", 1)
#     qdot_cheville = cas.MX.sym("q_dot_cheville", 1)
#
#     nq = int(pn.nlp.states.shape / 2)
#
#     q_mx = pn.nlp.states["q"].mx
#     qdot_mx = pn.nlp.states["qdot"].mx
#     tau_mx = pn.nlp.controls["tau"].mx
#     func_cheville = cas.Function("torqueMax_cheville", [q_cheville, qdot_cheville],
#                                  [model_cheville.torqueMax(q_cheville, qdot_cheville)[1].to_mx()])
#
#     func_q_cheville = q_cheville_func(q_cheville)
#     bound_cheville = func_cheville(func_q_cheville(q_mx[2]), -qdot_mx[2])
#
#     min_bound = []
#     bound = pn.nlp.model.torqueMax(q_mx, qdot_mx)[1].to_mx()
#     min_boundz = cas.if_else(bound[2:3] < minimal_tau, minimal_tau, bound[2:3])
#     min_boundz[0, 0] = 0
#     min_bound.append(min_boundz)
#     # min_bound[0] = cas.vertcat(np.ones((2,)) * -1000000, min_bound[0])
#
#     obj = []
#     obj.append(tau_mx[2:3])
#     obj_star = cas.vertcat(*obj)
#     min_bound = cas.vertcat(*min_bound)
#
#     constraint_min = BiorbdInterface.mx_to_cx("tau_actuator_constraints_min", min_bound -obj_star , pn.nlp.states["q"], pn.nlp.states["qdot"], pn.nlp.controls["tau"])
#
#     return constraint_min
#

# def tau_actuator_constraints_max(pn: PenaltyNode, path_model_cheville: str, minimal_tau: float = None) -> cas.MX:
#     model_cheville = biorbd.Model(path_model_cheville)
#     q_cheville = cas.MX.sym("q_cheville", 1)
#     qdot_cheville = cas.MX.sym("q_dot_cheville", 1)
#
#     nq = int(pn.nlp.states.shape / 2)
#
#     q_mx = pn.nlp.states["q"].mx
#     qdot_mx = pn.nlp.states["qdot"].mx
#     tau_mx = pn.nlp.controls["tau"].mx
#     func_cheville = cas.Function("torqueMax_cheville", [q_cheville, qdot_cheville],
#                                  [model_cheville.torqueMax(q_cheville, qdot_cheville)[0].to_mx()])
#
#     func_q_cheville = q_cheville_func(q_cheville)
#     bound_cheville = func_cheville(func_q_cheville(q_mx[2]), -qdot_mx[2])
#
#     max_bound = []
#     bound = pn.nlp.model.torqueMax(q_mx, qdot_mx)[0].to_mx()
#     max_boundz = cas.if_else(bound[2:3] < minimal_tau, minimal_tau, bound[2:3])
#     max_boundz[0, 0] = cas.if_else(bound_cheville[:, 0] < minimal_tau, minimal_tau,bound_cheville[:, 0])
#     max_bound.append(max_boundz)
#     #max_bound[0] = cas.vertcat(np.ones((2,)) * 1000000, max_bound[0])
#
#     obj = []
#     obj.append(tau_mx[2:3])
#     obj_star = cas.vertcat(*obj)
#     max_bound = cas.vertcat(*max_bound)
#
#     constraint_max = BiorbdInterface.mx_to_cx("tau_actuator_constraints_max", obj_star - max_bound, pn.nlp.states["q"],
#                                               pn.nlp.states["qdot"], pn.nlp.controls["tau"])
#     return constraint_max


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
        0.1,
    )

    tau_min, tau_max, tau_init = -1, 1, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", node=Node.END, index=1, weight=1000, phase=0, quadratic=False)

    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=0)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=1)

    # arriver avec les pieds au centre de la toile
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", phase=0, node=Node.START, index=0, weight=100, target=np.zeros((1, 1)))
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", phase=0, node=Node.START, index=1, weight=100, target=np.zeros((1, 1)))
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", phase=1, node=Node.END, index=0, weight=100, target=np.zeros((1, 1)))
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", phase=1, node=Node.END, index=1, weight=100, target=np.zeros((1, 1)))

    #maximiser la vitesse de remonter
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, phase=1, node=Node.END, weight=-100000, quadratic=False, axes=Axis.Z)

    # # Dynamics
    dynamics = DynamicsList()
    dynamics.add(Dynamics(custom_configure, dynamic_function=custom_dynamic))
    dynamics.add(Dynamics(custom_configure, dynamic_function=custom_dynamic))

    # --- Constraints --- #
    constraints = ConstraintList()
    # constraints.add(ConstraintFcn.TRACK_COM_VELOCITY, node=Node.START, min_bound=-15, max_bound=-10, phase=0, axes=Axis.Z)
    #
    # # Constraint time
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.02, max_bound=0.8, phase=0)
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.1, max_bound=0.9, phase=1)

    #contraintes sur le min
    # constraints.add(
    #     tau_actuator_constraints_min, phase=0, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=-np.inf, max_bound=0
    # )
    # constraints.add(
    #     tau_actuator_constraints_min, phase=1, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=-np.inf, max_bound=0
    # )
    #
    # # #contraintes sur le max
    # constraints.add(
    #     tau_actuator_constraints_max, phase=0, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=-np.inf, max_bound=0
    # )
    # constraints.add(
    #     tau_actuator_constraints_max, phase=1, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=-np.inf, max_bound=0
    # )


    # Path constraint
    X_bounds = BoundsList()

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    # X_bounds[0].min[:1, 1:] = [-0.3]
    # X_bounds[0].max[:1, 1:] = [0.3]

    vitesse_init = -12

    # X_bounds[0].min[:, 0] = [-0.3, 0, -0.4323, 1.4415, -1.5564, 1.02, -10, -30, -1, -1, -1, -1]
    # X_bounds[0].max[:, 0] = [0.3, 0, -0.4323, 1.4415, -1.5564, 1.02, 10, 0, 1, 1, 1, 1]
    # X_bounds[0].min[1:3, 1] = [-1.2, -0.9]
    # X_bounds[0].max[1:3, 1] = [0, 0.1]
    # X_bounds[0].min[1:3, 2] = [-1.2, -0.9]
    # X_bounds[0].max[1:3, 2] = [0, 0.1]
    #
    # X_bounds[0].min[7:8, 1] = [-30]
    # X_bounds[0].max[7:8, 1] = [0]
    # X_bounds[0].min[7:8, 2] = [0]
    # X_bounds[0].max[7:8, 2] = [0]

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[1]))
    # X_bounds[1].min[:1, 1:] = [-0.3]
    # X_bounds[1].max[:1, 1:] = [0.3]
    #
    # X_bounds[1].min[:3, 0] = [-0.3, -1.2, -0.9]
    # X_bounds[1].max[:3, 0] = [0.3, 0, 0.1]
    # X_bounds[1].min[1:3, 1] = [-1.2, -0.9]
    # X_bounds[1].max[1:3, 1] = [0, 0.1]
    # X_bounds[1].min[:3, 2] = [-0.3, 0, -0.9]
    # X_bounds[1].max[:3, 2] = [0.3, 0, 0.1]
    #
    # X_bounds[1].min[7:8, 0] = [0]
    # X_bounds[1].max[7:8, 0] = [30]
    # X_bounds[1].min[7:8, 1] = [0]
    # X_bounds[1].max[7:8, 1] = [30]
    # X_bounds[1].min[7:8, 2] = [0]
    # X_bounds[1].max[7:8, 2] = [30]

    # Define control path constraint
    u_bounds = BoundsList()
    # u_bounds.add(
    #     bounds=Bounds(
    #         [tau_min, tau_min, tau_min, tau_min], [tau_max, tau_max, tau_max, tau_max]
    #     )
    # )
    # u_bounds.add(
    #     bounds=Bounds(
    #         [tau_min, tau_min, tau_min, tau_min], [tau_max, tau_max, tau_max, tau_max]
    #     )
    # )
    u_bounds.add(
        bounds=Bounds(
            [-10000, -10000, -10000, -10000], [10000, 10000, 10000, 10000]
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [-10000, -10000, -10000, -10000], [10000, 10000, 10000, 10000]
        )
    )


    # Initial guess
    x_init = InitialGuessList()
    x_init.add(NoisedInitialGuess(
            [0]*12, # (nq + nqdot)
            bounds=X_bounds[0],
            noise_magnitude=0.2,
            n_shooting=number_shooting_points[0],
            bound_push=0.01,
            seed=i_rand,
        )
    )
    x_init.add(NoisedInitialGuess(
            [0]*12, # (nq + nqdot)
            bounds=X_bounds[1],
            noise_magnitude=0.2,
            n_shooting=number_shooting_points[1],
            bound_push=0.1,
            seed=i_rand,
        )
    )

    # x_init[0].init[1, :] = np.linspace(0, -1.2, 51)
    # x_init[1].init[1, :] = np.linspace(-1.2, -0.1, 51)
    #
    # x_init[0].init[7, :] = np.linspace(-20, 0, 51)
    # x_init[1].init[7, :] = np.linspace(0, 10, 51)

    u_init = InitialGuessList()
    u_init.add(NoisedInitialGuess(
            [0] * 4, # ntorque
            bounds=u_bounds[0],
            noise_magnitude=0.01,
            n_shooting=number_shooting_points[0]-1,
            bound_push=0.1,
            seed=i_rand,
        ))

    u_init.add(NoisedInitialGuess(
            [0] * 4,
            bounds=u_bounds[1],
            noise_magnitude=0.01,
            n_shooting=number_shooting_points[1]-1,
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

    # Date = date.today()
    # Date = Date.strftime("%d-%m-%y")
    # f = open(f"Historique_{Date}.txt", "w+")
    # f.write(" Debut ici \n\n\n")
    # f.close()

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

    # ----------------------------------------------- Sauteur_8 --------------------------------------------------------
    # f = open(f"Historique_{Date}.txt", "a+")
    # f.write(f"\n\n\nSalto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand} : ")
    # f.close()

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

    # path = '/home/lim/Documents/Jules/result_saut/' + 'phase01_sauteur_tauDYN' + '.pkl'
    # with open(path, 'wb') as file:
    #     pickle.dump(q, file)
    #     pickle.dump(qdot, file)
    #     pickle.dump(u, file)
    #     pickle.dump(t, file)
    #     pickle.dump(sol.time, file)
    #     pickle.dump(sol.phase_time, file)

    b = bioviz.Viz(model_path)
    b.load_movement(q)
    b.exec()

    # Q_sym = cas.MX.sym("Q_sym", biorbd.Model(model_path).nbQ(), 1)
    # custom_spring_const_post_func = custom_spring_const_post(Q_sym, lut_verticale, lut_horizontale, model_path)
    #
    # Marker_pied = np.zeros((3, np.shape(q)[1]))
    # Force_toile = np.zeros((3, np.shape(q)[1]))
    # for j in range(np.shape(q)[1]):
    #     Marker_pied_tempo, Force_toile_tempo = custom_spring_const_post_func(q[:, j])
    #     Marker_pied[:, j] = np.reshape(Marker_pied_tempo, (3))
    #     Force_toile[:, j] = np.reshape(Force_toile_tempo, (3))
    #
    # plt.figure()
    # plt.plot(Force_toile[0, :], "-r", label="Force x")
    # plt.plot(Force_toile[1, :], "-g", label="Force y")
    # plt.plot(Force_toile[2, :], "-b", label="Force z")
    # plt.legend()
    #
    # fig, axs = plt.subplots(2, 3)
    # axs = axs.ravel()
    # for iplt in range(biorbd.Model(model_path).nbQ()):
    #     axs[iplt].plot(q[iplt, :], "-b", label="Q")
    #     axs[iplt].plot(qdot[iplt, :], "-r", label="Qdot")
    #     axs[iplt].plot(u[iplt, :] / 100, "-g", label="U/100")
    #     axs[iplt].plot(np.array([50, 50]), np.array([-10, 10]), "--k")
    #     axs[iplt].plot(np.array([100, 100]), np.array([-10, 10]), "--k")
    #     axs[iplt].plot(np.array([150, 150]), np.array([-10, 10]), "--k")
    #     axs[iplt].plot(np.array([200, 200]), np.array([-10, 10]), "--k")
    #     axs[iplt].plot(np.array([250, 250]), np.array([-10, 10]), "--k")
    #     axs[iplt].set_xlabel(biorbd.Model(model_path).nameDof()[iplt].to_string())
    #     axs[iplt].legend()
    #
    # fig, axs = plt.subplots(1, 2)
    # axs = axs.ravel()
    # for iplt in range(2):
    #     axs[iplt].plot(q[iplt, :], "-b", label="Q")
    #     axs[iplt].plot(100, q[iplt, 100], "ob", label="Q")
    #     axs[iplt].plot(150, q[iplt, 150], "ob", label="Q")
    #     axs[iplt].plot(250, q[iplt, 250], "ob", label="Q")
    #     axs[iplt].plot(np.zeros(np.shape(q[iplt, :])), "-r")
    #     axs[iplt].plot(np.array([100, 100]), np.array([-1, 1]), "--k")
    #     axs[iplt].plot(np.array([150, 150]), np.array([-1, 1]), "--k")
    #     axs[iplt].plot(np.array([250, 250]), np.array([-1, 1]), "--k")
    #     axs[iplt].set_xlabel(biorbd.Model(model_path).nameDof()[iplt].to_string())
    #     axs[iplt].set_ylim(-0.15, 0.15)
    #     axs[iplt].legend()
    # plt.show()
    #
    # DirertoryFlies = os.listdir("/home/lim/Documents/Jules/code_initiaux_Eve/Position_massPoints")
    # q_toile = np.zeros((15 * 3, np.shape(q)[1]))
    # for j in range(np.shape(q)[1]):
    #     BestFileIndex = 0
    #     BestFileNum = 1000
    #     for iFile in range(len(DirertoryFlies)):
    #         if "y" in DirertoryFlies[iFile]:
    #             first_ = DirertoryFlies[iFile].index("_")
    #             firsty = DirertoryFlies[iFile].index("y")
    #             yy = float(DirertoryFlies[iFile][first_ + 1 : firsty])
    #             second_ = DirertoryFlies[iFile][first_ + 1 :].index("_")
    #             firstz = DirertoryFlies[iFile].index("z")
    #             zz = float(DirertoryFlies[iFile][first_ + 1 + second_ + 1 : firstz])
    #             if (abs(yy - Marker_pied[1, j]) + abs(zz - Marker_pied[2, j])) < BestFileNum:
    #                 BestFileNum = abs(yy - Marker_pied[1, j]) + abs(zz - Marker_pied[2, j])
    #                 BestFileIndex = iFile
    #                 yy_final = yy
    #                 zz_final = zz
    #
    #         data = np.load(
    #             f"/home/lim/Documents/Jules/code_initiaux_Eve/Position_massPoints/{DirertoryFlies[BestFileIndex]}"
    #         )
    #         q_toile[:, j] = data.T.flatten()
    #
    # b = bioviz.Viz("/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/jumper_sansPieds_rootPied_bioviz.bioMod")
    # b.load_movement(np.vstack((q_toile, q)))
    # b.exec()