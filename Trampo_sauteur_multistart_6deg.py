from time import time
import numpy as np
import casadi as cas
import biorbd_casadi as biorbd
import os
import matplotlib.pyplot as plt
from datetime import date


from bioptim import (
    Node,
    OptimalControlProgram,
    ConstraintList,
    ConstraintFcn,
    ObjectiveFcn,
    ObjectiveList,
    DynamicsList,
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
)


def custom_spring_const(pn: PenaltyNode, lut_verticale, lut_horizontale) -> cas.MX: #ajout de la force de la toile comme etant la force appliquee a la cheville

    # nq = int(pn.nlp.states.shape /2) #nombre de degres de liberté

    val_contrainte = []

    u_i = pn.nlp.variable_mappings["tau"].to_second.map(pn.nlp.controls["tau"].mx)
    q_i = pn.nlp.variable_mappings['tau'].to_second.map(pn.nlp.states['q'].mx)

    Markers = pn.nlp.model.markers(pn.nlp.states["q"].mx)
    Marker_pied = Markers[0].to_mx()

    Force = cas.MX.zeros(3)
    Force[1] = lut_horizontale(Marker_pied[1:])
    Force[2] = lut_verticale(Marker_pied[1:])

    return_value = cas.vertcat(u_i[1] - Force[2], u_i[1] - Force[2])

    val_contrainte = cas.Function("Force", [pn.nlp.states['q'].mx, pn.nlp.controls['tau'].mx], [return_value])(pn.nlp.states['q'].cx, pn.nlp.controls['tau'].cx)

    return val_contrainte


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

def tau_actuator_constraints_min(pn: PenaltyNode, path_model_cheville: str, minimal_tau: float = None) -> cas.MX:
    nq = int(pn.nlp.states.shape / 2)

    q_mx = pn.nlp.states["q"].mx
    qdot_mx = pn.nlp.states["qdot"].mx
    tau_mx = pn.nlp.controls["tau"].mx

    min_bound = []
    bound = pn.nlp.model.torqueMax(q_mx, qdot_mx)[1].to_mx()
    min_boundz = cas.if_else(bound[2:nq] < minimal_tau, minimal_tau, bound[2:nq])
    min_boundz[0, 0] = 0
    min_bound.append(min_boundz)
    min_bound[0] = cas.vertcat(np.ones((2,)) * -100000, min_bound[0])

    obj = []
    obj.append(tau_mx)
    obj_star = cas.vertcat(*obj)
    min_bound = cas.vertcat(*min_bound)

    constraint_min = BiorbdInterface.mx_to_cx("tau_actuator_constraints_min", obj_star - min_bound, pn.nlp.states["q"], pn.nlp.states["qdot"], pn.nlp.controls["tau"])

    # q_mx = pn.nlp.variable_mappings["q"].to_second.map(pn.nlp.states["q"].mx)
    # qdot_mx = pn.nlp.variable_mappings['qdot'].to_second.map(pn.nlp.states['qdot'].mx)
    # tau_mx = pn.nlp.variable_mappings["tau"].to_second.map(pn.nlp.controls["tau"].mx)
    # q_cx = pn.nlp.variable_mappings["q"].to_second.map(pn.nlp.states["q"].cx)
    # qdot_cx = pn.nlp.variable_mappings['qdot'].to_second.map(pn.nlp.states['qdot'].cx)
    # tau_cx = pn.nlp.variable_mappings["tau"].to_second.map(pn.nlp.controls["tau"].cx)
    #
    # # bound = cas.Function("torqueMax", [q_mx, qdot_mx], [pn.nlp.model.torqueMax(q_mx, qdot_mx)[1].to_mx()])(q_mx, qdot_mx)
    # # func_cheville = cas.Function("torqueMax_cheville", [pn.nlp.states['q'].mx, pn.nlp.states['qdot'].mx],
    # #                               [model_cheville.torqueMax(pn.nlp.states['q'].mx, pn.nlp.states['qdot'].mx)[1].to_mx()])(pn.nlp.states['q'].cx, pn.nlp.states['qdot'].cx)
    #
    # # func_q_cheville = q_cheville_func(q_cheville)
    # # bound_cheville = func_cheville(func_q_cheville(q[2]), -qdot[2])
    #
    # # min_boundz = cas.if_else(bound[2:nq] < minimal_tau, minimal_tau, bound[2:nq])
    # # min_boundz[0, 0] = 0
    # # # min_boundz[3, :] = cas.if_else(bound[5, 1] < minimal_tau, minimal_tau, bound[5, 1])
    # # min_bound.append(min_boundz)
    # # min_bound[0] = cas.vertcat(np.ones((2, )) * -np.inf, min_bound[0])
    # obj.append(tau_mx)
    #
    # obj_star = cas.vertcat(*obj)
    # min_bound = cas.vertcat(*min_bound)
    #
    # # return (
    # #     cas.vertcat(np.zeros(min_bound.shape), np.ones(max_bound.shape) * -np.inf),
    # #     cas.vertcat(obj_star + min_bound, obj_star - max_bound),
    # #     cas.vertcat(np.ones(min_bound.shape) * np.inf, np.zeros(max_bound.shape)),
    # # )
    #
    # func_constraint = cas.Function('tau_actuator_constraints_min', [q_mx, qdot_mx], [obj_star])(q_cx, qdot_cx)


    return constraint_min


def tau_actuator_constraints_max(pn: PenaltyNode, path_model_cheville: str, minimal_tau: float = None) -> cas.MX:
    model_cheville = biorbd.Model(path_model_cheville)
    q_cheville = cas.MX.sym("q_cheville", 1)
    qdot_cheville = cas.MX.sym("q_dot_cheville", 1)

    nq = int(pn.nlp.states.shape / 2)

    q_mx = pn.nlp.states["q"].mx
    qdot_mx = pn.nlp.states["qdot"].mx
    tau_mx = pn.nlp.controls["tau"].mx

    # func_cheville = model_cheville.torqueMax(q_cheville, qdot_cheville)
    # func_q_cheville = q_cheville_func(q_cheville)
    # bound_cheville = func_cheville(func_q_cheville(q_mx[2]), -qdot_mx[2])

    max_bound = []
    bound = pn.nlp.model.torqueMax(q_mx, qdot_mx)[1].to_mx()
    max_boundz = cas.if_else(bound[2:nq] < minimal_tau, minimal_tau, bound[2:nq])
    max_boundz[0, 0] = 0 #cas.if_else(bound_cheville[:, 0] < minimal_tau, minimal_tau,bound_cheville[:, 0])
    max_bound.append(max_boundz)
    max_bound[0] = cas.vertcat(np.ones((2,)) * 100000, max_bound[0])

    obj = []
    obj.append(tau_mx)
    obj_star = cas.vertcat(*obj)
    max_bound = cas.vertcat(*max_bound)

    constraint_max = BiorbdInterface.mx_to_cx("tau_actuator_constraints_max", obj_star - max_bound, pn.nlp.states["q"],
                                              pn.nlp.states["qdot"], pn.nlp.controls["tau"])
    return constraint_max


# def tau_actuator_constraints_max(pn: PenaltyNode, path_model_cheville: str, minimal_tau: float = None) -> cas.MX:
#     model_cheville = biorbd.Model(path_model_cheville)
#
#     nq = int(pn.nlp.states.shape /2)
#
#     tau = pn.nlp.variable_mappings["tau"].to_second.map(pn.nlp.controls["tau"].mx)
#     q = pn.nlp.variable_mappings["q"].to_second.map(pn.nlp.states["q"].mx)
#     qdot = pn.nlp.variable_mappings['qdot'].to_second.map(pn.nlp.states['qdot'].mx)
#
#     q_cheville = cas.MX.sym("q_cheville", 1)
#     q_dot_cheville = cas.MX.sym("q_dot_cheville", 1)
#
#     min_bound = []
#     max_bound = []
#     obj = []
#
#     bound = cas.Function("torqueMax", [pn.nlp.states['q'].mx, pn.nlp.states['qdot'].mx],
#                          [pn.nlp.model.torqueMax(pn.nlp.states['q'].mx, pn.nlp.states['qdot'].mx)[0].to_mx()])(pn.nlp.states['q'].cx, pn.nlp.states['qdot'].cx)
#     bound_cheville = cas.Function("torqueMax_cheville", [pn.nlp.states['q'].mx, pn.nlp.states['qdot'].mx],
#                                   [model_cheville.torqueMax(pn.nlp.states['q'].mx, pn.nlp.states['qdot'].mx)[0].to_mx()])(pn.nlp.states['q'].cx, pn.nlp.states['qdot'].cx)
#
#
#     func_q_cheville = q_cheville_func(q_cheville)
#     # bound_cheville = func_cheville(func_q_cheville(q[2]), -qdot[2])
#
#     max_boundz = cas.if_else(bound[2:nq] < minimal_tau, minimal_tau, bound[2:nq])
#     max_boundz[0, 0] = 0
#     # max_boundz[3, :] = cas.if_else(bound[5, 0] < minimal_tau, minimal_tau, bound[5, 0])
#     max_bound.append(max_boundz)
#     obj.append(tau[2:])
#
#     obj_star = cas.vertcat(*obj)
#     max_bound = cas.vertcat(*max_bound)
#
#     # return (
#     #     cas.vertcat(np.zeros(min_bound.shape), np.ones(max_bound.shape) * -np.inf),
#     #     cas.vertcat(obj_star + min_bound, obj_star - max_bound),
#     #     cas.vertcat(np.ones(min_bound.shape) * np.inf, np.zeros(max_bound.shape)),
#     # )
#     return cas.vertcat((max_bound - obj_star) >= 0)


def prepare_ocp_back_back(path_model_cheville, lut_verticale, lut_horizontale, weight, Salto1, Salto2):
    # --- Options --- #
    model_path = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_6DoF.bioMod"
    model_path_massToile = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_6DoF_massToile.bioMod"

    # Model path
    biorbd_model = (
        biorbd.Model(model_path_massToile),
        biorbd.Model(model_path_massToile),
        biorbd.Model(model_path),
        biorbd.Model(model_path_massToile),
        biorbd.Model(model_path_massToile),
        biorbd.Model(model_path),
    )

    nb_phases = len(biorbd_model)
    nq = biorbd_model[0].nbQ()
    nqdot = biorbd_model[0].nbQdot()

    number_shooting_points = (
        50,
        50,
        50,
        50,
        50,
        50,
    )

    final_time = (
        0.3,
        0.3,
        1.5,
        0.3,
        0.3,
        1.5,
    )

    tau_min, tau_max, tau_init = -1000, 1000, 0

    nq = biorbd_model[0].nbQ()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,key="q", node=Node.END, index=1, weight=100, phase=0, quadratic=False)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,key="q", node=Node.END, index=1, weight=100, phase=3, quadratic=False)

    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-100000, phase=1, quadratic=False, axis=Axis.Z)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, weight=-weight, phase=2, quadratic=False, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, weight=-weight, phase=5, quadratic=False, axes=Axis.Z)

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=4)

    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", node=Node.END, index=2, weight=1000, phase=2, target=np.ones((1, 1)) * 2 * np.pi * Salto1,)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", node=Node.END, index=2, weight=1000, phase=5, target=np.ones((1, 1)) * (2 * np.pi * Salto1 + 2 * np.pi * Salto2),)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, phase=0, min_bound=time_min[0], max_bound=time_max[0])

    # arriver avec les pieds au centre de la toile
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,key="q", phase=0, node=Node.START, index=0, weight=100, target=np.zeros((1, 1)))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,key="q", phase=0, node=Node.START, index=1, weight=100, target=np.zeros((1, 1)))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,key="q", phase=1, node=Node.END, index=0, weight=100, target=np.zeros((1, 1)))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,key="q", phase=1, node=Node.END, index=1, weight=100, target=np.zeros((1, 1)))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,key="q", phase=2, node=Node.END, index=0, weight=100, target=np.zeros((1, 1)))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,key="q", phase=2, node=Node.END, index=1, weight=100, target=np.zeros((1, 1)))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,key="q", phase=4, node=Node.END, index=0, weight=100, target=np.zeros((1, 1)))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,key="q", phase=4, node=Node.END, index=1, weight=100, target=np.zeros((1, 1)))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,key="q", phase=5, node=Node.END, index=0, weight=100, target=np.zeros((1, 1)))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,key="q", phase=5, node=Node.END, index=1, weight=100, target=np.zeros((1, 1)))

    # # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # --- Constraints --- #
    constraints = ConstraintList()

    # Constraint arm positivity
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.2, max_bound=3, phase=0)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.2, max_bound=3, phase=1)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.2, max_bound=3, phase=2)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.2, max_bound=3, phase=3)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.2, max_bound=3, phase=4)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.2, max_bound=3, phase=5)

    constraints.add(
        custom_spring_const,
        lut_verticale=lut_verticale,
        lut_horizontale=lut_horizontale,
        node=Node.ALL,
        min_bound=0,
        max_bound=0,
        phase=0,
    )
    constraints.add(
        custom_spring_const,
        lut_verticale=lut_verticale,
        lut_horizontale=lut_horizontale,
        node=Node.ALL,
        min_bound=0,
        max_bound=0,
        phase=1,
    )
    constraints.add(
        custom_spring_const,
        lut_verticale=lut_verticale,
        lut_horizontale=lut_horizontale,
        node=Node.ALL,
        min_bound=0,
        max_bound=0,
        phase=3,
    )
    constraints.add(
        custom_spring_const,
        lut_verticale=lut_verticale,
        lut_horizontale=lut_horizontale,
        node=Node.ALL,
        min_bound=0,
        max_bound=0,
        phase=4,
    )
    #contraintes sur le min
    constraints.add(
        tau_actuator_constraints_min, phase=0, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=0, max_bound=np.inf
    )
    constraints.add(
        tau_actuator_constraints_min, phase=1, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=0, max_bound=np.inf
    )
    constraints.add(
        tau_actuator_constraints_min, phase=2, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=0, max_bound=np.inf
    )
    constraints.add(
        tau_actuator_constraints_min, phase=3, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=0, max_bound=np.inf
    )
    constraints.add(
        tau_actuator_constraints_min, phase=4, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=0, max_bound=np.inf
    )
    constraints.add(
        tau_actuator_constraints_min, phase=5, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=0, max_bound=np.inf
    )
    #contraintes sur le max
    constraints.add(
        tau_actuator_constraints_max, phase=0, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=-np.inf, max_bound=0
    )
    constraints.add(
        tau_actuator_constraints_max, phase=1, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=-np.inf, max_bound=0
    )
    constraints.add(
        tau_actuator_constraints_max, phase=2, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=-np.inf, max_bound=0
    )
    constraints.add(
        tau_actuator_constraints_max, phase=3, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=-np.inf, max_bound=0
    )
    constraints.add(
        tau_actuator_constraints_max, phase=4, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=-np.inf, max_bound=0
    )
    constraints.add(
        tau_actuator_constraints_max, phase=5, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville, min_bound=-np.inf, max_bound=0
    )

    # constraints.add(ConstraintFcn.TORQUE_MAX_FROM_Q_AND_QDOT, phase=0, node=Node.ALL, min_torque=20)#, path_model_cheville=path_model_cheville)
    # constraints.add(ConstraintFcn.TORQUE_MAX_FROM_Q_AND_QDOT, phase=1, node=Node.ALL, min_torque=20)#, path_model_cheville=path_model_cheville)
    # constraints.add(ConstraintFcn.TORQUE_MAX_FROM_Q_AND_QDOT, phase=2, node=Node.ALL, min_torque=20)#, path_model_cheville=path_model_cheville)
    # constraints.add(ConstraintFcn.TORQUE_MAX_FROM_Q_AND_QDOT, phase=3, node=Node.ALL, min_torque=20)#, path_model_cheville=path_model_cheville)
    # constraints.add(ConstraintFcn.TORQUE_MAX_FROM_Q_AND_QDOT, phase=4, node=Node.ALL, min_torque=20)#, path_model_cheville=path_model_cheville)
    # constraints.add(ConstraintFcn.TORQUE_MAX_FROM_Q_AND_QDOT, phase=5, node=Node.ALL, min_torque=20)#, path_model_cheville=path_model_cheville)
    # constraints.add(ConstraintFcn., node=Node.END)

    # Path constraint
    X_bounds = BoundsList()

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    X_bounds[0].min[:, 0] = [0, 0, -0.4323, 1.4415, -1.5564, 1.02, -10, -30, -1, -1, -1, -1]
    X_bounds[0].max[:, 0] = [0, 0, -0.4323, 1.4415, -1.5564, 1.02, 10, 3, 1, 1, 1, 1]
    X_bounds[0].min[1:3, 1] = [-1.2, -0.5]
    X_bounds[0].max[1:3, 1] = [0, 0.5]
    X_bounds[0].min[1:3, 2] = [-1.2, -0.5]
    X_bounds[0].max[1:3, 2] = [0, 0.5]

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[1]))
    X_bounds[1].min[:3, 0] = [-0.5, -1.2, -0.5]
    X_bounds[1].max[:3, 0] = [0.5, 0, 0.5]
    X_bounds[1].min[1:3, 1] = [-1.2, -0.5]
    X_bounds[1].max[1:3, 1] = [0, 0.5]
    X_bounds[1].min[:3, 2] = [-0.5, -0.5, -0.5]  # 0.05
    X_bounds[1].max[:3, 2] = [0.5, 0.5, 0.5]  # 0.05

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[2]))
    X_bounds[2].min[:3, 0] = [-0.5, -0.5, -0.5]
    X_bounds[2].max[:3, 0] = [0.5, 0.5, 0.5]
    X_bounds[2].min[1, 1] = 0
    X_bounds[2].max[1, 1] = 10
    X_bounds[2].min[:3, 2] = [-0.5, -0.5, Salto1 * 2 * np.pi - 0.5]  # 0.05
    X_bounds[2].max[:3, 2] = [0.5, 0.5, Salto1 * 2 * np.pi + 0.5]  # 0.05

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[3]))
    X_bounds[3].min[:3, 0] = [-0.5, -0.5, Salto1 * 2 * np.pi - 0.5]
    X_bounds[3].max[:3, 0] = [0.5, 0.5, Salto1 * 2 * np.pi + 0.5]
    X_bounds[3].min[1:3, 1] = [-1.2, Salto1 * 2 * np.pi - 0.5]
    X_bounds[3].max[1:3, 1] = [0, Salto1 * 2 * np.pi + 0.5]
    X_bounds[3].min[1:3, 2] = [-1.2, Salto1 * 2 * np.pi - 0.5]
    X_bounds[3].max[1:3, 2] = [0, Salto1 * 2 * np.pi + 0.5]

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[4]))
    X_bounds[4].min[:3, 0] = [-0.5, -1.2, Salto1 * 2 * np.pi - 0.5]
    X_bounds[4].max[:3, 0] = [0.5, 0, Salto1 * 2 * np.pi + 0.5]
    X_bounds[4].min[1:3, 1] = [-1.2, Salto1 * 2 * np.pi - 0.5]
    X_bounds[4].max[1:3, 1] = [0, Salto1 * 2 * np.pi + 0.5]
    X_bounds[4].min[:3, 2] = [-0.5, -0.5, Salto1 * 2 * np.pi - 0.5]  # 0.05
    X_bounds[4].max[:3, 2] = [0.5, 0.5, Salto1 * 2 * np.pi + 0.5]  # 0.05

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[5]))
    X_bounds[5].min[:3, 0] = [-0.5, -0.5, Salto1 * 2 * np.pi - 0.5]
    X_bounds[5].max[:3, 0] = [0.5, 0.5, Salto1 * 2 * np.pi + 0.5]
    X_bounds[5].min[1, 1] = 0
    X_bounds[5].max[1, 1] = 10
    X_bounds[5].min[:3, 2] = [-0.5, -0.5, (Salto1 + Salto2) * 2 * np.pi - 0.5]  # 0.05
    X_bounds[5].max[:3, 2] = [0.5, 0.5, (Salto1 + Salto2) * 2 * np.pi + 0.5]  # 0.05

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        bounds=Bounds(
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min], [100000, 100000, tau_max, tau_max, tau_max, tau_max]
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min], [100000, 100000, tau_max, tau_max, tau_max, tau_max]
        )
    )
    u_bounds.add(bounds=Bounds([0, 0, 0, tau_min, tau_min, tau_min], [0, 0, 0, tau_max, tau_max, tau_max]))
    u_bounds.add(
        bounds=Bounds(
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min], [100000, 100000, tau_max, tau_max, tau_max, tau_max]
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min], [100000, 100000, tau_max, tau_max, tau_max, tau_max]
        )
    )
    u_bounds.add(bounds=Bounds([0, 0, 0, tau_min, tau_min, tau_min], [0, 0, 0, tau_max, tau_max, tau_max]))

    # Initial guess
    x_init = InitialGuessList()
    # x_init.add(
    #     np.random.random((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(),
    #                       number_shooting_points[0] + 1)) - 0.5,interpolation = InterpolationType.EACH_FRAME,)
    x_init.add(NoisedInitialGuess(
            [0] * 12, # (nq + nqdot)
            bounds=X_bounds[0],
            noise_magnitude=0.2,
            n_shooting=number_shooting_points[0],
            bound_push=0.01,
            seed=i_rand,
        )
    )
    # x_init.add(
    #     np.random.random((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(),
    #                       number_shooting_points[0] + 1)) - 0.5,interpolation = InterpolationType.EACH_FRAME,)
    x_init.add(NoisedInitialGuess(
            [0] * 12, # (nq + nqdot)
            bounds=X_bounds[1],
            noise_magnitude=0.2,
            n_shooting=number_shooting_points[1],
            bound_push=0.1,
            seed=i_rand,
        )
    )
    # x_init.add(
    #     np.random.random((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(),
    #                       number_shooting_points[0] + 1)) - 0.5,interpolation = InterpolationType.EACH_FRAME,)
    x_init.add(NoisedInitialGuess(
            [0] * 12, # (nq + nqdot)
            bounds=X_bounds[2],
            noise_magnitude=0, # 0.2,
            n_shooting=number_shooting_points[2],
            bound_push=0.1,
            seed=i_rand,
        )
    )
    # x_init.add(
    #     np.random.random((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(),
    #                       number_shooting_points[0] + 1)) - 0.5,interpolation = InterpolationType.EACH_FRAME,)
    x_init.add(NoisedInitialGuess(
            [0] * 12, # (nq + nqdot)
            bounds=X_bounds[3],
            noise_magnitude=0.2,
            n_shooting=number_shooting_points[3],
            bound_push=0.1,
            seed=i_rand,
        )
    )
    # x_init.add(
    #     np.random.random((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(),
    #                       number_shooting_points[0] + 1)) - 0.5,interpolation = InterpolationType.EACH_FRAME,)
    x_init.add(NoisedInitialGuess(
            [0] * 12, # (nq + nqdot)
            bounds=X_bounds[4],
            noise_magnitude=0.2,
            n_shooting=number_shooting_points[4],
            bound_push=0.1,
            seed=i_rand,
        )
    )
    x_init.add(
        np.random.random((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(),
                          number_shooting_points[0] + 1)) - 0.5,interpolation = InterpolationType.EACH_FRAME,)
    x_init.add(NoisedInitialGuess(
            [0] * 12, # (nq + nqdot)
            bounds=X_bounds[5],
            noise_magnitude=0, # 0.2,
            n_shooting=number_shooting_points[5],
            bound_push=0.1,
            seed=i_rand,
        )
    )

    u_init = InitialGuessList()
    u_init.add(NoisedInitialGuess(
            [0] * 6, # ntorque
            bounds=u_bounds[0],
            noise_magnitude=0.01,
            n_shooting=number_shooting_points[0]-1,
            bound_push=0.1,
            seed=i_rand,
        ))

    u_init.add(NoisedInitialGuess(
            [0] * 6,
            bounds=u_bounds[1],
            noise_magnitude=0.01,
            n_shooting=number_shooting_points[1]-1,
            bound_push=0.1,
            seed=i_rand,
        ))

    u_init.add(NoisedInitialGuess(
            [0] * 6,
            bounds=u_bounds[2],
            noise_magnitude=0.01,
            n_shooting=number_shooting_points[2]-1,
            bound_push=0.1,
            seed=i_rand,
        ))

    u_init.add(NoisedInitialGuess(
            [0] * 6,
            bounds=u_bounds[3],
            noise_magnitude=0.01,
            n_shooting=number_shooting_points[3]-1,
            bound_push=0.1,
            seed=i_rand,
        ))

    u_init.add(NoisedInitialGuess(
            [0] * 6,
            bounds=u_bounds[4],
            noise_magnitude=0.01,
            n_shooting=number_shooting_points[4]-1,
            bound_push=0.1,
            seed=i_rand,
        ))

    u_init.add(NoisedInitialGuess(
            [0] * 6,
            bounds=u_bounds[5],
            noise_magnitude=0.01,
            n_shooting=number_shooting_points[5]-1,
            bound_push=0.1,
            seed=i_rand,
        ))

    x_init[0].init[1, :] = np.linspace(-0.1, -1, 51)
    x_init[1].init[1, :] = np.linspace(-1, -0.1, 51)
    # x_init[2].init[1, :] = np.hstack((np.linspace(0.1, 4, 26), np.linspace(4, 0.1, 25)))
    x_init[3].init[1, :] = np.linspace(-0.1, -1, 51)
    x_init[4].init[1, :] = np.linspace(-1, -0.1, 51)
    # x_init[5].init[1, :] = np.hstack((np.linspace(0.1, 4, 26), np.linspace(4, 0.1, 25)))

    # u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    # u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    # u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    # u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    # u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    # u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

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
        n_threads=4,
    )
    return ocp


if __name__ == "__main__":

    Date = date.today()
    Date = Date.strftime("%d-%m-%y")
    f = open(f"Historique_{Date}.txt", "w+")
    f.write(" Debut ici \n\n\n")
    f.close()

    # Salto_1 = np.array([0, 0, 0, 0, 0,  0,  0,  0,  0,
    #                     1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
    #                      1,  1,  1,  1, 2,  2,  2,  2,  3,  3,  3,  3])
    # Salto_2 = np.array([0, 1, 2, 3, 4, -1, -2, -3, -4,
    #                     1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
    #                     -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4])

    Salto_1 = np.array([1])
    Salto_2 = np.array([1])

    Weight_choices = np.array([1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000])

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
    for i_salto in range(len(Salto_1)):
        Salto1 = Salto_1[i_salto]
        Salto2 = Salto_2[i_salto]

        weight = 10000
        i_rand = 1

        # ----------------------------------------------- Sauteur_8 --------------------------------------------------------
        f = open(f"Historique_{Date}.txt", "a+")
        f.write(f"\n\n\nSalto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand} : ")
        f.close()

        tic = time()
        ocp = prepare_ocp_back_back(path_model_cheville=path_model_cheville,lut_verticale=lut_verticale,lut_horizontale=lut_horizontale,weight=weight,Salto1=Salto1,Salto2=Salto2,)

#
        solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
        solver.set_maximum_iterations(100000)
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
            u = np.hstack((u, ocp.nlp[i].variable_mappings["q"].to_second.map(sol.controls[i]["tau"])))

        # if sol.status == 0:
        #     f = open(f"Historique_{Date}.txt", "a+")
        #     f.write(f"Success \n")
        #     f.close()
        #     np.save(
        #         f"Results_MultiStart_Sauteur/Q_Salto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand}",
        #         q,
        #     )
        #     np.save(
        #         f"Results_MultiStart_Sauteur/Qdot_Salto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand}",
        #         qdot,
        #     )
        #     np.save(
        #         f"Results_MultiStart_Sauteur/U_Salto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand}",
        #         u,
        #     )
        #     np.save(
        #         f"Results_MultiStart_Sauteur/T_Salto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand}",
        #         t,
        #     )
        #
        # else:
        #     f = open(f"Historique_{Date}.txt", "a+")
        #     f.write(f"Fail \n")
        #     f.close()

#####################################################################################################################
    import bioviz

    model_path = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_6DoF.bioMod"

    b = bioviz.Viz(model_path)
    b.load_movement(q)
    b.exec()

    Q_sym = cas.MX.sym("Q_sym", biorbd.Model(model_path).nbQ(), 1)
    custom_spring_const_post_func = custom_spring_const_post(Q_sym, lut_verticale, lut_horizontale, model_path)

    Marker_pied = np.zeros((3, np.shape(q)[1]))
    Force_toile = np.zeros((3, np.shape(q)[1]))
    for j in range(np.shape(q)[1]):
        Marker_pied_tempo, Force_toile_tempo = custom_spring_const_post_func(q[:, j])
        Marker_pied[:, j] = np.reshape(Marker_pied_tempo, (3))
        Force_toile[:, j] = np.reshape(Force_toile_tempo, (3))

    # fig, axs = plt.subplots(1, 3)
    # axs = axs.ravel()
    # for iplt in range(3):
    #     axs[iplt].plot(Force_toile[iplt, :], '-k', label='Force')
    #     axs[iplt].legend()
    # axs[0].set_xlabel('x')
    # axs[0].set_xlabel('y')
    # axs[0].set_xlabel('z')
    # plt.show()

    plt.figure()
    plt.plot(Force_toile[0, :], "-r", label="Force x")
    plt.plot(Force_toile[1, :], "-g", label="Force y")
    plt.plot(Force_toile[2, :], "-b", label="Force z")
    plt.legend()
    # plt.show()

    fig, axs = plt.subplots(2, 3)
    axs = axs.ravel()
    for iplt in range(biorbd.Model(model_path).nbQ()):
        axs[iplt].plot(q[iplt, :], "-b", label="Q")
        axs[iplt].plot(qdot[iplt, :], "-r", label="Qdot")
        axs[iplt].plot(u[iplt, :] / 100, "-g", label="U/100")
        axs[iplt].plot(np.array([50, 50]), np.array([-10, 10]), "--k")
        axs[iplt].plot(np.array([100, 100]), np.array([-10, 10]), "--k")
        axs[iplt].plot(np.array([150, 150]), np.array([-10, 10]), "--k")
        axs[iplt].plot(np.array([200, 200]), np.array([-10, 10]), "--k")
        axs[iplt].plot(np.array([250, 250]), np.array([-10, 10]), "--k")
        axs[iplt].set_xlabel(biorbd.Model(model_path).nameDof()[iplt].to_string())
        axs[iplt].legend()
    # plt.show()

    fig, axs = plt.subplots(1, 2)
    axs = axs.ravel()
    for iplt in range(2):
        axs[iplt].plot(q[iplt, :], "-b", label="Q")
        axs[iplt].plot(100, q[iplt, 100], "ob", label="Q")
        axs[iplt].plot(150, q[iplt, 150], "ob", label="Q")
        axs[iplt].plot(250, q[iplt, 250], "ob", label="Q")
        axs[iplt].plot(np.zeros(np.shape(q[iplt, :])), "-r")
        axs[iplt].plot(np.array([100, 100]), np.array([-1, 1]), "--k")
        axs[iplt].plot(np.array([150, 150]), np.array([-1, 1]), "--k")
        axs[iplt].plot(np.array([250, 250]), np.array([-1, 1]), "--k")
        axs[iplt].set_xlabel(biorbd.Model(model_path).nameDof()[iplt].to_string())
        axs[iplt].set_ylim(-0.15, 0.15)
        axs[iplt].legend()
    plt.show()

    DirertoryFlies = os.listdir("/home/lim/Documents/Jules/code_initiaux_Eve/Position_massPoints")
    q_toile = np.zeros((15 * 3, np.shape(q)[1]))
    for j in range(np.shape(q)[1]):
        BestFileIndex = 0
        BestFileNum = 1000
        for iFile in range(len(DirertoryFlies)):
            if "y" in DirertoryFlies[iFile]:
                first_ = DirertoryFlies[iFile].index("_")
                firsty = DirertoryFlies[iFile].index("y")
                yy = float(DirertoryFlies[iFile][first_ + 1 : firsty])
                second_ = DirertoryFlies[iFile][first_ + 1 :].index("_")
                firstz = DirertoryFlies[iFile].index("z")
                zz = float(DirertoryFlies[iFile][first_ + 1 + second_ + 1 : firstz])
                if (abs(yy - Marker_pied[1, j]) + abs(zz - Marker_pied[2, j])) < BestFileNum:
                    BestFileNum = abs(yy - Marker_pied[1, j]) + abs(zz - Marker_pied[2, j])
                    BestFileIndex = iFile
                    yy_final = yy
                    zz_final = zz

            data = np.load(
                f"/home/lim/Documents/Jules/code_initiaux_Eve/Position_massPoints/{DirertoryFlies[BestFileIndex]}"
            )
            q_toile[:, j] = data.T.flatten()

    b = bioviz.Viz("/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/jumper_sansPieds_rootPied_bioviz.bioMod")
    b.load_movement(np.vstack((q_toile, q)))
    b.exec()
