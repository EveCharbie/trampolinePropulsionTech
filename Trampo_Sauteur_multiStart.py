from time import time
import numpy as np
import casadi as cas
import biorbd
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
    Axis,
    PenaltyNode,
    InterpolationType,
    Bounds,
    BiMapping,
)


def custom_spring_const(pn: PenaltyNode, lut_verticale, lut_horizontale) -> cas.MX:

    nq = int(pn.nlp.nx / 2)

    val_contrainte = []

    u_i = pn.nlp.mapping["q"].to_second.map(pn.u)  # pn.u[i]
    q_i = pn.nlp.mapping["q"].to_second.map(pn.x[:nq])  # np.x[i][:nq]

    Marker_pied = pn.nlp.model.markers(q_i)[0].to_mx()

    Force = cas.MX.zeros(3)
    Force[1] = lut_horizontale(Marker_pied[1:])
    Force[2] = lut_verticale(Marker_pied[1:])

    val_contrainte = cas.vertcat(val_contrainte, u_i[0] - Force[1])
    val_contrainte = cas.vertcat(val_contrainte, u_i[1] - Force[2])

    return val_contrainte


def CoM_base_appui(pn: PenaltyNode) -> cas.MX:
    val_contrainte = []
    nq = int(pn.nlp.nx / 2)
    q_i = pn.nlp.mapping["q"].to_second.map(pn.x[:nq])  # pn.x[i][:nq]
    CoM = pn.nlp.model.CoM(q_i).to_mx()
    q_pied_y = q_i[0]
    CoM_proj = CoM[1]
    val_contrainte = cas.vertcat(val_contrainte, CoM_proj - q_pied_y)
    return val_contrainte


def Non_trans_toile(pn: PenaltyNode) -> cas.MX:
    val_contrainte = []
    nq = int(pn.nlp.nx / 2)
    q_i = pn.nlp.mapping["q"].to_second.map(pn.x[:nq])  # pn.x[i][:nq]
    Marker_pied = pn.nlp.model.markers(q_i)[0].to_mx()[2]
    Marker_genou = pn.nlp.model.markers(q_i)[1].to_mx()[2]
    Marker_hanche = pn.nlp.model.markers(q_i)[2].to_mx()[2]

    val_contrainte = cas.vertcat(val_contrainte, Marker_genou - Marker_pied)
    val_contrainte = cas.vertcat(val_contrainte, Marker_hanche - Marker_pied)

    return val_contrainte


def custom_spring_const_post(Q, lut_verticale, lut_horizontale, model_path):
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


def tau_actuator_constraints(pn: PenaltyNode, path_model_cheville: str, minimal_tau: float = None) -> cas.MX:
    model_cheville = biorbd.Model(path_model_cheville)

    nq = pn.nlp.mapping["q"].to_first.len
    q = [pn.nlp.mapping["q"].to_second.map(mx[:nq]) for mx in pn.x]
    q_dot = [pn.nlp.mapping["qdot"].to_second.map(mx[nq:]) for mx in pn.x]
    q_cheville = cas.MX.sym("q_cheville", 1)
    q_dot_cheville = cas.MX.sym("q_dot_cheville", 1)

    min_bound = []
    max_bound = []
    obj = []

    func = biorbd.to_casadi_func("torqueMax", pn.nlp.model.torqueMax, pn.nlp.q, pn.nlp.qdot)
    func_cheville = biorbd.to_casadi_func("torqueMax_cheville", model_cheville.torqueMax, q_cheville, q_dot_cheville)
    func_q_cheville = q_cheville_func(q_cheville)

    bound = func(q, q_dot)
    bound_cheville = func_cheville(func_q_cheville(q[2]), -q_dot[2])

    min_boundz = cas.if_else(bound[2:nq, 1] < minimal_tau, minimal_tau, bound[2:nq, 1])
    min_boundz[0, 0] = 0
    # min_boundz[3, :] = cas.if_else(bound[5, 1] < minimal_tau, minimal_tau, bound[5, 1])
    min_bound.append(min_boundz)
    max_boundz = cas.if_else(bound[2:nq, 0] < minimal_tau, minimal_tau, bound[2:nq, 0])
    max_boundz[0, 0] = cas.if_else(bound_cheville[:, 0] < minimal_tau, minimal_tau, bound_cheville[:, 0])
    # max_boundz[3, :] = cas.if_else(bound[5, 0] < minimal_tau, minimal_tau, bound[5, 0])
    max_bound.append(max_boundz)
    obj.append(pn.u[2:])

    obj_star = cas.vertcat(*obj)
    min_bound = cas.vertcat(*min_bound)
    max_bound = cas.vertcat(*max_bound)

    return (
        cas.vertcat(np.zeros(min_bound.shape), np.ones(max_bound.shape) * -np.inf),
        cas.vertcat(obj_star + min_bound, obj_star - max_bound),
        cas.vertcat(np.ones(min_bound.shape) * np.inf, np.zeros(max_bound.shape)),
    )


def Xinit_ramdom(X_bounds, biorbd_model, number_shooting_points):

    x_init = InitialGuessList()

    for j in range(len(biorbd_model)):
        X_init = np.zeros((biorbd_model[j].nbQ() + biorbd_model[j].nbQdot(), number_shooting_points[j] + 1))
        for i in range(biorbd_model[j].nbQ()):
            if i == 2:
                X_init[i, :] = np.zeros((1, number_shooting_points[j] + 1))
            else:
                X_init[i, 0] = np.random.uniform(low=X_bounds[j].min[i, 0], high=X_bounds[j].max[i, 0], size=(1,))
                X_init[i, 1:-1] = np.random.uniform(
                    low=X_bounds[j].min[i, 1], high=X_bounds[j].max[i, 1], size=(1, number_shooting_points[j] - 1)
                )
                X_init[i, -1] = np.random.uniform(low=X_bounds[j].min[i, 2], high=X_bounds[j].max[i, 2], size=(1,))

        x_init.add(X_init, interpolation=InterpolationType.EACH_FRAME)

    return x_init


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
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, node=Node.END, index=1, weight=100, phase=0, quadratic=False
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, node=Node.END, index=1, weight=100, phase=3, quadratic=False
    )
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-100000, phase=1, quadratic=False, axis=Axis.Z)
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, weight=-weight, phase=2, quadratic=False, axis=Axis.Z
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, weight=-weight, phase=5, quadratic=False, axis=Axis.Z
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=4)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        node=Node.END,
        index=2,
        weight=1000,
        phase=2,
        target=np.ones((1, 1)) * 2 * np.pi * Salto1,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        node=Node.END,
        index=2,
        weight=1000,
        phase=5,
        target=np.ones((1, 1)) * (2 * np.pi * Salto1 + 2 * np.pi * Salto2),
    )
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, phase=0, min_bound=time_min[0], max_bound=time_max[0])

    # arriver avec les pieds au centre de la toile
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=0, node=Node.START, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=0, node=Node.START, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=1, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=1, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=2, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=2, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=4, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=4, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=5, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=5, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # --- Constraints --- #
    constraints = ConstraintList()

    # # Constraint arm positivity
    # constraints.add(ConstraintFcn.TRACK_STATE, phase=1, node=Node.END, index=5, min_bound=0, max_bound=np.inf)
    # constraints.add(ConstraintFcn.TRACK_STATE, phase=4, node=Node.END, index=5, min_bound=0, max_bound=np.inf)

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
    constraints.add(
        tau_actuator_constraints, phase=0, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=1, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=2, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=3, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=4, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=5, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )

    # constraints.add(Non_trans_toile, phase=0, node=Node.ALL, min_bound=0, max_bound=np.inf)  # aucun segment de vedrait etre sous la toile
    # constraints.add(Non_trans_toile, phase=1, node=Node.ALL, min_bound=0, max_bound=np.inf)
    # constraints.add(Non_trans_toile, phase=3, node=Node.ALL, min_bound=0, max_bound=np.inf)
    # constraints.add(Non_trans_toile, phase=4, node=Node.ALL, min_bound=0, max_bound=np.inf)

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

    # x_min_0 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_0 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_0[:, 0] = [0, 0, -0.73, 1.41, -1.70-0.7, 1.02,
    #                  0-10, -30, 0-0.7, 0-0.7, 0-0.7, 0-0.7]
    # x_max_0[:, 0] = [0, 0, -0.73, 1.41, -1.70+0.7, 1.02,
    #                  0+10, -3, 0+0.7, 0+0.7, 0+0.7, 0+0.7]
    # x_min_0[:, 1] = [0-0.5, -1.2, -0.73-0.7, 1.41-0.7, -1.70-0.7, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_0[:, 1] = [0+0.5, 0, -0.73+0.7, 1.41+0.7, -1.70+0.7, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000]
    # x_min_0[:, 2] = x_min_0[:, 1]
    # x_max_0[:, 2] = x_max_0[:, 1]
    # X_bounds.add(bounds=Bounds(x_min_0, x_max_0, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_1 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_1 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_1[:, 0] = x_min_0[:, 1]
    # x_max_1[:, 0] = x_max_0[:, 1]
    # x_min_1[:, 1] = x_min_0[:, 1]
    # x_max_1[:, 1] = x_max_0[:, 1]
    # x_min_1[:, 2] = [0-0.5, 0-0.5, -0.73-0.7, 1.41-0.7, -1.70-0.7, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_1[:, 2] = [0+0.5, 0+0.5, -0.73+0.7, 1.41+0.7, -1.70+0.7, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000]
    # X_bounds.add(bounds=Bounds(x_min_1, x_max_1, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_2 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_2 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_2[:, 0] = x_min_1[:, 2]
    # x_max_2[:, 0] = x_max_1[:, 2]
    # x_min_2[:, 1] = [0-2.7, 0, -4*np.pi, 1.41-2.7, -1.70-2.7, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_2[:, 1] = [0+2.7, 5, 4*np.pi, 1.41+2.7, -1.70+2.7, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000]
    # x_min_2[:, 2] = [0-0.5, 0-0.5, 2*np.pi-0.5, 1.41-0.7, -1.70-0.7, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_2[:, 2] = [0+0.5, 0+0.5, 2*np.pi+0.5, 1.41+0.7, -1.70+0.7, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000]
    # X_bounds.add(bounds=Bounds(x_min_2, x_max_2, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_3 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_3 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_3[:, 0] = [0-0.5, -1.2, 2*np.pi-0.7, 1.41-0.7, -1.70-0.7, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_3[:, 0] = [0+0.5,  0,   2*np.pi+0.7, 1.41+0.7, -1.70+0.7, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000]
    # x_min_3[:, 1] = x_min_3[:, 0]
    # x_max_3[:, 1] = x_max_3[:, 0]
    # x_min_3[:, 2] = x_min_3[:, 0]
    # x_max_3[:, 2] = x_max_3[:, 0]
    # X_bounds.add(bounds=Bounds(x_min_3, x_max_3, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_4 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_4 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_4[:, 0] = x_min_3[:, 1]
    # x_max_4[:, 0] = x_max_3[:, 1]
    # x_min_4[:, 1] = x_min_3[:, 1]
    # x_max_4[:, 1] = x_max_3[:, 1]
    # x_min_4[:, 2] = [0-0.5, 0-0.5, 2*np.pi-0.7, 1.41-0.7, -1.70-0.7, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_4[:, 2] = [0+0.5, 0+0.5, 2*np.pi+0.7, 1.41+0.7, -1.70+0.7, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000]
    # X_bounds.add(bounds=Bounds(x_min_4, x_max_4, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_5 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_5 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_5[:, 0] = x_min_4[:, 2]
    # x_max_5[:, 0] = x_max_4[:, 2]
    # x_min_5[:, 1] = [0-2.7, 0, 2*np.pi-0.7, 1.41-2.7, -1.70-2.7, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_5[:, 1] = [0+2.7, 5, 6*np.pi, 1.41+2.7, -1.70+2.7, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000]
    # x_min_5[:, 2] = [0-0.5, 0-0.5, 4*np.pi-0.5, 1.41-0.7, -1.70-0.7, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_5[:, 2] = [0+0.5, 0+0.5, 4*np.pi+0.5, 1.41+0.7, -1.70+0.7, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000]
    # X_bounds.add(bounds=Bounds(x_min_5, x_max_5, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))

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
    x_init.add(
        np.random.random((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), number_shooting_points[0] + 1)) - 0.5,
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        np.random.random((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), number_shooting_points[0] + 1)) - 0.5,
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        np.random.random((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), number_shooting_points[0] + 1)) - 0.5,
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        np.random.random((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), number_shooting_points[0] + 1)) - 0.5,
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        np.random.random((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), number_shooting_points[0] + 1)) - 0.5,
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        np.random.random((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), number_shooting_points[0] + 1)) - 0.5,
        interpolation=InterpolationType.EACH_FRAME,
    )

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

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
        n_threads=2,
    )
    return ocp


def prepare_ocp_back_back_tronc(path_model_cheville, lut_verticale, lut_horizontale, weight, i_rand, Salto1, Salto2):
    # --- Options --- #
    model_path = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_7DoF.bioMod"
    model_path_massToile = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_7DoF_massToile.bioMod"

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
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, node=Node.END, index=1, weight=100, phase=0, quadratic=False
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, node=Node.END, index=1, weight=100, phase=3, quadratic=False
    )
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-100000, phase=1, quadratic=False, axis=Axis.Z)
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, weight=-weight, phase=2, quadratic=False, axis=Axis.Z
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, weight=-weight, phase=5, quadratic=False, axis=Axis.Z
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=4)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        node=Node.END,
        index=2,
        weight=1000,
        phase=2,
        target=np.ones((1, 1)) * 2 * np.pi * Salto1,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        node=Node.END,
        index=2,
        weight=1000,
        phase=5,
        target=np.ones((1, 1)) * (2 * np.pi * Salto1 + 2 * np.pi * Salto2),
    )
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, phase=0, min_bound=time_min[0], max_bound=time_max[0])

    # arriver avec les pieds au centre de la toile
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=0, node=Node.START, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=0, node=Node.START, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=1, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=1, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=2, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=2, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=4, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=4, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=5, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=5, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # --- Constraints --- #
    constraints = ConstraintList()

    # Custom constraints for positivity of CoM_dot on z axis just before the take-off
    # constraints.add(utils.com_dot_z, phase=0, node=Node.END, min_bound=0, max_bound=np.inf)

    # Constraint arm positivity
    # constraints.add(ConstraintFcn.TRACK_STATE, phase=1, node=Node.END, index=5, min_bound=0, max_bound=np.inf)
    # constraints.add(ConstraintFcn.TRACK_STATE, phase=4, node=Node.END, index=5, min_bound=0, max_bound=np.inf)

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
    constraints.add(
        tau_actuator_constraints, phase=0, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=1, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=2, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=3, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=4, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=5, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )

    # constraints.add(Non_trans_toile, phase=0, node=Node.ALL, min_bound=0, max_bound=np.inf)  # aucun segment de vedrait etre sous la toile
    # constraints.add(Non_trans_toile, phase=1, node=Node.ALL, min_bound=0, max_bound=np.inf)
    # constraints.add(Non_trans_toile, phase=3, node=Node.ALL, min_bound=0, max_bound=np.inf)
    # constraints.add(Non_trans_toile, phase=4, node=Node.ALL, min_bound=0, max_bound=np.inf)

    # # Path constraint
    # X_bounds = BoundsList()
    # x_min_0 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_0 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_0[:, 0] = [0, 0, -0.73, 1.41, -1.70-0.7, -0.5, 1.02,
    #                  0-10, -30, 0-0.7, 0-0.7, 0-0.7, -0.7, 0-0.7]
    # x_max_0[:, 0] = [0, 0, -0.73, 1.41, -1.70+0.7, 0.3, 1.02,
    #                  0+10, -3, 0+0.7, 0+0.7, 0+0.7, 0.7, 0+0.7]
    # x_min_0[:, 1] = [0-0.5, -1.2, -0.73-0.7, 1.41-0.7, -1.70-0.7, -0.5, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_0[:, 1] = [0+0.5, 0, -0.73+0.7, 1.41+0.7, -1.70+0.7, 0.3, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000, 1000]
    # x_min_0[:, 2] = x_min_0[:, 1]
    # x_max_0[:, 2] = x_max_0[:, 1]
    # X_bounds.add(bounds=Bounds(x_min_0, x_max_0, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_1 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_1 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_1[:, 0] = x_min_0[:, 1]
    # x_max_1[:, 0] = x_max_0[:, 1]
    # x_min_1[:, 1] = x_min_0[:, 1]
    # x_max_1[:, 1] = x_max_0[:, 1]
    # x_min_1[:, 2] = [0-0.5, 0-0.5, -0.73-0.7, 1.41-0.7, -1.70-0.7, -0.5, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_1[:, 2] = [0+0.5, 0+0.5, -0.73+0.7, 1.41+0.7, -1.70+0.7, 0.3, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000, 1000]
    # X_bounds.add(bounds=Bounds(x_min_1, x_max_1, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_2 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_2 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_2[:, 0] = x_min_1[:, 2]
    # x_max_2[:, 0] = x_max_1[:, 2]
    # x_min_2[:, 1] = [0-2.7, 0, -4*np.pi, 1.41-2.7, -1.70-2.7, -0.5, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_2[:, 1] = [0+2.7, 5, 4*np.pi, 1.41+2.7, -1.70+2.7, 0.3, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000, 1000]
    # x_min_2[:, 2] = [0-0.5, 0-0.5, 2*np.pi-0.5, 1.41-0.7, -1.70-0.7, -0.5, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_2[:, 2] = [0+0.5, 0+0.5, 2*np.pi+0.5, 1.41+0.7, -1.70+0.7, 0.3, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000, 1000]
    # X_bounds.add(bounds=Bounds(x_min_2, x_max_2, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_3 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_3 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_3[:, 0] = [0-0.5, -1.2, 2*np.pi-0.7, 1.41-0.7, -1.70-0.7, -0.5, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_3[:, 0] = [0+0.5,  0,   2*np.pi+0.7, 1.41+0.7, -1.70+0.7, 0.3, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000, 1000]
    # x_min_3[:, 1] = x_min_3[:, 0]
    # x_max_3[:, 1] = x_max_3[:, 0]
    # x_min_3[:, 2] = x_min_3[:, 0]
    # x_max_3[:, 2] = x_max_3[:, 0]
    # X_bounds.add(bounds=Bounds(x_min_3, x_max_3, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_4 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_4 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_4[:, 0] = x_min_3[:, 1]
    # x_max_4[:, 0] = x_max_3[:, 1]
    # x_min_4[:, 1] = x_min_3[:, 1]
    # x_max_4[:, 1] = x_max_3[:, 1]
    # x_min_4[:, 2] = [0-0.5, 0-0.5, 2*np.pi-0.7, 1.41-0.7, -1.70-0.7, -0.5, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_4[:, 2] = [0+0.5, 0+0.5, 2*np.pi+0.7, 1.41+0.7, -1.70+0.7, 0.3, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000, 1000]
    # X_bounds.add(bounds=Bounds(x_min_4, x_max_4, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_5 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_5 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_5[:, 0] = x_min_4[:, 2]
    # x_max_5[:, 0] = x_max_4[:, 2]
    # x_min_5[:, 1] = [0-2.7, 0, 2*np.pi-0.7, 1.41-2.7, -1.70-2.7, -0.5, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_5[:, 1] = [0+2.7, 5, 6*np.pi, 1.41+2.7, -1.70+2.7, 0.3, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000, 1000]
    # x_min_5[:, 2] = [0-0.5, 0-0.5, 4*np.pi-0.5, 1.41-0.7, -1.70-0.7, -0.5, 1.02-2.7, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_5[:, 2] = [0+0.5, 0+0.5, 4*np.pi+0.5, 1.41+0.7, -1.70+0.7, 0.3, 1.02+2.7,  1000,  1000,  1000,  1000,  1000,  1000, 1000]
    # X_bounds.add(bounds=Bounds(x_min_5, x_max_5, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))

    # Path constraint
    X_bounds = BoundsList()

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    X_bounds[0].min[:, 0] = [0, 0, -0.4323, 1.4415, -1.5564, 0, 1.02, -10, -30, -1, -1, -1, -1, -1]
    X_bounds[0].max[:, 0] = [0, 0, -0.4323, 1.4415, -1.5564, 0, 1.02, 10, 3, 1, 1, 1, 1, 1]
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
    X_bounds[2].min[:2, 0] = [-0.5, -0.5]
    X_bounds[2].max[:2, 0] = [0.5, 0.5]
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
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min, tau_min],
            [100000, 100000, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min, tau_min],
            [100000, 100000, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )
    u_bounds.add(
        bounds=Bounds([0, 0, 0, tau_min, tau_min, tau_min, tau_min], [0, 0, 0, tau_max, tau_max, tau_max, tau_max])
    )
    u_bounds.add(
        bounds=Bounds(
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min, tau_min],
            [100000, 100000, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min, tau_min],
            [100000, 100000, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )
    u_bounds.add(
        bounds=Bounds([0, 0, 0, tau_min, tau_min, tau_min, tau_min], [0, 0, 0, tau_max, tau_max, tau_max, tau_max])
    )

    q = np.load(f"Results_MultiStart_Sauteur/Q_Salto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand}.npy")
    qdot = np.load(
        f"Results_MultiStart_Sauteur/Qdot_Salto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand}.npy"
    )

    q_reshape = np.zeros((nq, np.shape(q)[1]))
    q_reshape[:5, :] = q[:5, :]
    q_reshape[6, :] = q[5, :]

    qdot_reshape = np.zeros((nq, np.shape(qdot)[1]))
    qdot_reshape[:5, :] = qdot[:5, :]
    qdot_reshape[6, :] = qdot[5, :]

    q_old = np.vstack((q_reshape, qdot_reshape))

    nu = biorbd_model[0].nbGeneralizedTorque()
    u = np.load(f"Results_MultiStart_Sauteur/U_Salto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand}.npy")
    u_reshape = np.zeros((nu, np.shape(u)[1]))
    u_reshape[:5, :] = u[:5, :]
    u_reshape[6, :] = u[5, :]
    u_old = u_reshape

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(q_old[:, : number_shooting_points[0] + 1], interpolation=InterpolationType.EACH_FRAME)
    x_init.add(
        q_old[:, number_shooting_points[0] + 1 : number_shooting_points[0] + number_shooting_points[1] + 2],
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        q_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + 2 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + 3,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        q_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + 3 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + 4,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        q_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + 4 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + number_shooting_points[4]
            + 5,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        q_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + number_shooting_points[4]
            + 5 :,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )

    u_init = InitialGuessList()
    u_init.add(u_old[:, : number_shooting_points[0]], interpolation=InterpolationType.EACH_FRAME)
    u_init.add(
        u_old[:, number_shooting_points[0] + 1 : number_shooting_points[0] + number_shooting_points[1] + 1],
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_init.add(
        u_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + 2 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + 2,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_init.add(
        u_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + 3 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + 3,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_init.add(
        u_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + 4 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + number_shooting_points[4]
            + 4,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_init.add(
        u_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + number_shooting_points[4]
            + 5 : -1,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )

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
        n_threads=2,
    )
    return ocp


def prepare_ocp_back_back_tronc_coude(
    path_model_cheville, lut_verticale, lut_horizontale, weight, i_rand, Salto1, Salto2
):
    # --- Options --- #
    model_path = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_8DoF.bioMod"
    model_path_massToile = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_8DoF_massToile.bioMod"

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
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, node=Node.END, index=1, weight=100, phase=0, quadratic=False
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, node=Node.END, index=1, weight=100, phase=3, quadratic=False
    )
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-100000, phase=1, quadratic=False, axis=Axis.Z)
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, weight=-weight, phase=2, quadratic=False, axis=Axis.Z
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, weight=-weight, phase=5, quadratic=False, axis=Axis.Z
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=4)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        node=Node.END,
        index=2,
        weight=1000,
        phase=2,
        target=np.ones((1, 1)) * 2 * np.pi * Salto1,
    )  # quadratic=False,
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        node=Node.END,
        index=2,
        weight=1000,
        phase=5,
        target=np.ones((1, 1)) * (2 * np.pi * Salto1 + 2 * np.pi * Salto2),
    )
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, phase=0, min_bound=time_min[0], max_bound=time_max[0])

    # arriver avec les pieds au centre de la toile
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=0, node=Node.START, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=0, node=Node.START, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=1, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=1, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=2, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=2, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=4, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=4, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=5, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=5, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )

    # Dynamics
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
    # constraints.add(ConstraintFcn.TRACK_STATE, phase=1, node=Node.END, index=5, min_bound=0, max_bound=np.inf)
    # constraints.add(ConstraintFcn.TRACK_STATE, phase=4, node=Node.END, index=5, min_bound=0, max_bound=np.inf)

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
    constraints.add(
        tau_actuator_constraints, phase=0, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=1, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=2, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=3, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=4, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=5, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )

    # constraints.add(Non_trans_toile, phase=0, node=Node.ALL, min_bound=0, max_bound=np.inf)  # aucun segment de vedrait etre sous la toile
    # constraints.add(Non_trans_toile, phase=1, node=Node.ALL, min_bound=0, max_bound=np.inf)
    # constraints.add(Non_trans_toile, phase=3, node=Node.ALL, min_bound=0, max_bound=np.inf)
    # constraints.add(Non_trans_toile, phase=4, node=Node.ALL, min_bound=0, max_bound=np.inf)

    # # Path constraint
    # X_bounds = BoundsList()
    # x_min_0 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_0 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_0[:, 0] = [0, 0, -0.73, 1.41, -1.70-0.7, -0.5, 1.02, 0,
    #                  0-10, -30, 0-0.7, 0-0.7, 0-0.7, -0.7, 0-0.7, 0-0.7]
    # x_max_0[:, 0] = [0, 0, -0.73, 1.41, -1.70+0.7, 0.3, 1.02, 0,
    #                  0+10, -3, 0+0.7, 0+0.7, 0+0.7, 0.7, 0+0.7, 0+0.7]
    # x_min_0[:, 1] = [0-0.5, -1.2, -0.73-0.7, 1.41-0.7, -1.70-0.7, -0.5, 1.02-2.7, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_0[:, 1] = [0+0.5, 0, -0.73+0.7, 1.41+0.7, -1.70+0.7, 0.3, 1.02+2.7, 2.3,  1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000]
    # x_min_0[:, 2] = x_min_0[:, 1]
    # x_max_0[:, 2] = x_max_0[:, 1]
    # X_bounds.add(bounds=Bounds(x_min_0, x_max_0, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_1 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_1 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_1[:, 0] = x_min_0[:, 1]
    # x_max_1[:, 0] = x_max_0[:, 1]
    # x_min_1[:, 1] = x_min_0[:, 1]
    # x_max_1[:, 1] = x_max_0[:, 1]
    # x_min_1[:, 2] = [0-0.5, 0-0.5, -0.73-0.7, 1.41-0.7, -1.70-0.7, -0.5, 1.02-2.7, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_1[:, 2] = [0+0.5, 0+0.5, -0.73+0.7, 1.41+0.7, -1.70+0.7, 0.3, 1.02+2.7, 2.3,  1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000]
    # X_bounds.add(bounds=Bounds(x_min_1, x_max_1, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_2 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_2 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_2[:, 0] = x_min_1[:, 2]
    # x_max_2[:, 0] = x_max_1[:, 2]
    # x_min_2[:, 1] = [0-2.7, 0, -4*np.pi, 1.41-2.7, -1.70-2.7, -0.5, 1.02-2.7, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_2[:, 1] = [0+2.7, 5, 4*np.pi, 1.41+2.7, -1.70+2.7, 0.3, 1.02+2.7, 2.3,  1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000]
    # x_min_2[:, 2] = [0-0.5, 0-0.5, 2*np.pi-0.5, 1.41-0.7, -1.70-0.7, -0.5, 1.02-2.7, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_2[:, 2] = [0+0.5, 0+0.5, 2*np.pi+0.5, 1.41+0.7, -1.70+0.7, 0.3, 1.02+2.7,  2.3, 1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000]
    # X_bounds.add(bounds=Bounds(x_min_2, x_max_2, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_3 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_3 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_3[:, 0] = [0-0.5, -1.2, 2*np.pi-0.7, 1.41-0.7, -1.70-0.7, -0.5, 1.02-2.7, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_3[:, 0] = [0+0.5,  0,   2*np.pi+0.7, 1.41+0.7, -1.70+0.7, 0.3, 1.02+2.7, 2.3, 1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000]
    # x_min_3[:, 1] = x_min_3[:, 0]
    # x_max_3[:, 1] = x_max_3[:, 0]
    # x_min_3[:, 2] = x_min_3[:, 0]
    # x_max_3[:, 2] = x_max_3[:, 0]
    # X_bounds.add(bounds=Bounds(x_min_3, x_max_3, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_4 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_4 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_4[:, 0] = x_min_3[:, 1]
    # x_max_4[:, 0] = x_max_3[:, 1]
    # x_min_4[:, 1] = x_min_3[:, 1]
    # x_max_4[:, 1] = x_max_3[:, 1]
    # x_min_4[:, 2] = [0-0.5, 0-0.5, 2*np.pi-0.7, 1.41-0.7, -1.70-0.7, -0.5, 1.02-2.7, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_4[:, 2] = [0+0.5, 0+0.5, 2*np.pi+0.7, 1.41+0.7, -1.70+0.7, 0.3, 1.02+2.7, 2.3, 1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000]
    # X_bounds.add(bounds=Bounds(x_min_4, x_max_4, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_5 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_max_5 = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), 3))
    # x_min_5[:, 0] = x_min_4[:, 2]
    # x_max_5[:, 0] = x_max_4[:, 2]
    # x_min_5[:, 1] = [0-2.7, 0, 2*np.pi-0.7, 1.41-2.7, -1.70-2.7, -0.5, 1.02-2.7, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_5[:, 1] = [0+2.7, 5, 6*np.pi, 1.41+2.7, -1.70+2.7, 0.3, 1.02+2.7, 2.3,   1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000]
    # x_min_5[:, 2] = [0-0.5, 0-0.5, 4*np.pi-0.5, 1.41-0.7, -1.70-0.7, -0.5, 1.02-2.7, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_5[:, 2] = [0+0.5, 0+0.5, 4*np.pi+0.5, 1.41+0.7, -1.70+0.7, 0.3, 1.02+2.7, 2.3, 1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000]
    # X_bounds.add(bounds=Bounds(x_min_5, x_max_5, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))

    # Path constraint
    X_bounds = BoundsList()

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    X_bounds[0].min[:, 0] = [0, 0, -0.4323, 1.4415, -1.5564, 0, 1.02, 0, -10, -30, -1, -1, -1, -1, -1, -1]
    X_bounds[0].max[:, 0] = [0, 0, -0.4323, 1.4415, -1.5564, 0, 1.02, 0, 10, 3, 1, 1, 1, 1, 1, 1]
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
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min],
            [100000, 100000, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min],
            [100000, 100000, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [0, 0, 0, tau_min, tau_min, tau_min, tau_min, tau_min],
            [0, 0, 0, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min],
            [100000, 100000, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min],
            [100000, 100000, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [0, 0, 0, tau_min, tau_min, tau_min, tau_min, tau_min],
            [0, 0, 0, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )

    q = np.load(f"Results_MultiStart_Sauteur/Q_Salto1{Salto1}_Salto2{Salto2}_DoF7_weight{weight}_random{i_rand}.npy")
    qdot = np.load(
        f"Results_MultiStart_Sauteur/Qdot_Salto1{Salto1}_Salto2{Salto2}_DoF7_weight{weight}_random{i_rand}.npy"
    )

    q_reshape = np.zeros((nq, np.shape(q)[1]))
    q_reshape[:-1, :] = q

    qdot_reshape = np.zeros((nq, np.shape(qdot)[1]))
    qdot_reshape[:-1, :] = qdot

    q_old = np.vstack((q_reshape, qdot_reshape))

    nu = biorbd_model[0].nbGeneralizedTorque()
    u = np.load(f"Results_MultiStart_Sauteur/U_Salto1{Salto1}_Salto2{Salto2}_DoF7_weight{weight}_random{i_rand}.npy")
    u_reshape = np.zeros((nu, np.shape(u)[1]))
    u_reshape[:-1, :] = u
    u_old = u_reshape

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(q_old[:, : number_shooting_points[0] + 1], interpolation=InterpolationType.EACH_FRAME)
    x_init.add(
        q_old[:, number_shooting_points[0] + 1 : number_shooting_points[0] + number_shooting_points[1] + 2],
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        q_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + 2 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + 3,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        q_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + 3 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + 4,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        q_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + 4 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + number_shooting_points[4]
            + 5,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        q_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + number_shooting_points[4]
            + 5 :,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )

    u_init = InitialGuessList()
    u_init.add(u_old[:, : number_shooting_points[0]], interpolation=InterpolationType.EACH_FRAME)
    u_init.add(
        u_old[:, number_shooting_points[0] + 1 : number_shooting_points[0] + number_shooting_points[1] + 1],
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_init.add(
        u_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + 2 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + 2,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_init.add(
        u_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + 3 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + 3,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_init.add(
        u_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + 4 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + number_shooting_points[4]
            + 4,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_init.add(
        u_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + number_shooting_points[4]
            + 5 : -1,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )

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
        n_threads=2,
    )
    return ocp


def prepare_ocp_back_back_tronc_coude2D(
    path_model_cheville, lut_verticale, lut_horizontale, weight, i_rand, Salto1, Salto2
):
    # --- Options --- #
    model_path = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_14DoF.bioMod"
    model_path_massToile = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_14DoF_massToile.bioMod"

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
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, node=Node.END, index=1, weight=100, phase=0, quadratic=False
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, node=Node.END, index=1, weight=100, phase=3, quadratic=False
    )
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-100000, phase=1, quadratic=False, axis=Axis.Z)
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, weight=-1000000, phase=2, quadratic=False, axis=Axis.Z
    )  # -100000
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, weight=-1000000, phase=5, quadratic=False, axis=Axis.Z
    )  # -100000
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1, phase=4)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        node=Node.END,
        index=2,
        weight=1000,
        phase=2,
        target=np.ones((1, 1)) * 2 * np.pi * Salto1,
    )  # quadratic=False,
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        node=Node.END,
        index=2,
        weight=1000,
        phase=5,
        target=np.ones((1, 1)) * (2 * np.pi * Salto1 + 2 * np.pi * Salto2),
    )
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, phase=0, min_bound=time_min[0], max_bound=time_max[0])

    # arriver avec les pieds au centre de la toile
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=0, node=Node.START, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=0, node=Node.START, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=1, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=1, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=2, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=2, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=4, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=4, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=5, node=Node.END, index=0, weight=100, target=np.zeros((1, 1))
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, phase=5, node=Node.END, index=1, weight=100, target=np.zeros((1, 1))
    )

    q_mapping = BiMapping([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -6, 7, -8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    all_mapping = q_mapping, q_mapping, q_mapping, q_mapping, q_mapping, q_mapping

    # Dynamics
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
    # constraints.add(ConstraintFcn.TRACK_STATE, phase=1, node=Node.END, index=5, min_bound=0, max_bound=np.inf)
    # constraints.add(ConstraintFcn.TRACK_STATE, phase=4, node=Node.END, index=5, min_bound=0, max_bound=np.inf)

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
    constraints.add(
        tau_actuator_constraints, phase=0, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=1, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=2, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=3, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=4, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )
    constraints.add(
        tau_actuator_constraints, phase=5, node=Node.ALL, minimal_tau=20, path_model_cheville=path_model_cheville
    )

    # constraints.add(Non_trans_toile, phase=0, node=Node.ALL, min_bound=0, max_bound=np.inf)  # aucun segment de vedrait etre sous la toile
    # constraints.add(Non_trans_toile, phase=1, node=Node.ALL, min_bound=0, max_bound=np.inf)
    # constraints.add(Non_trans_toile, phase=3, node=Node.ALL, min_bound=0, max_bound=np.inf)
    # constraints.add(Non_trans_toile, phase=4, node=Node.ALL, min_bound=0, max_bound=np.inf)

    # # Path constraint
    # nq = q_mapping.to_first.len
    # X_bounds = BoundsList()
    # x_min_0 = np.zeros((nq + nq, 3))
    # x_max_0 = np.zeros((nq + nq, 3))
    # x_min_0[:, 0] = [0, 0, -0.73, 1.41, -1.70-0.7, -0.5, 0, 1.02, 0, 0,
    #                  0-10, -30, -0.7, -0.7, -0.7, -0.7, -0.7, -0.7, -0.7, -0.7]
    # x_max_0[:, 0] = [0, 0, -0.73, 1.41, -1.70+0.7, 0.3, 0, 1.02, 0, 0,
    #                  10, -3, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    # x_min_0[:, 1] = [0-0.5, -1.2, -0.73-0.7, 1.41-0.7, -1.70-0.7, -0.5, -0.5, -1.02-2.7, -0.5, 0,
    #                  -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_0[:, 1] = [0+0.5, 0, -0.73+0.7, 1.41+0.7, -1.70+0.7, 0.3, 0.5, 1.02+2.7, 0.5, 2.3,
    #                   1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000]
    # x_min_0[:, 2] = x_min_0[:, 1]
    # x_max_0[:, 2] = x_max_0[:, 1]
    # X_bounds.add(bounds=Bounds(x_min_0, x_max_0, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_1 = np.zeros((nq + nq, 3))
    # x_max_1 = np.zeros((nq + nq, 3))
    # x_min_1[:, 0] = x_min_0[:, 1]
    # x_max_1[:, 0] = x_max_0[:, 1]
    # x_min_1[:, 1] = x_min_0[:, 1]
    # x_max_1[:, 1] = x_max_0[:, 1]
    # x_min_1[:, 2] = [0-0.5, 0-0.5, -0.73-0.7, 1.41-0.7, -1.70-0.7, -0.5, -0.5, 1.02-2.7, -0.5, 0,
    #                  -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_1[:, 2] = [0+0.5, 0+0.5, -0.73+0.7, 1.41+0.7, -1.70+0.7, 0.3, 0.5, 1.02+2.7, 0.5, 2.3,
    #                   1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000, 1000, 1000]
    # X_bounds.add(bounds=Bounds(x_min_1, x_max_1, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_2 = np.zeros((nq + nq, 3))
    # x_max_2 = np.zeros((nq + nq, 3))
    # x_min_2[:, 0] = x_min_1[:, 2]
    # x_max_2[:, 0] = x_max_1[:, 2]
    # x_min_2[:, 1] = [0-2.7, 0, -4*np.pi, 1.41-2.7, -1.70-2.7, -0.5, -0.5, 1.02-2.7, -0.5, 0,
    #                  -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_2[:, 1] = [0+2.7, 5, 4*np.pi, 1.41+2.7, -1.70+2.7, 0.3, 0.5, 1.02+2.7, 0.5, 2.3,
    #                   1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000, 1000, 1000]
    # x_min_2[:, 2] = [0-0.5, 0-0.5, 2*np.pi-0.5, 1.41-0.7, -1.70-0.7, -0.5, -0.5, 1.02-2.7, -0.5, 0,
    #                  -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_2[:, 2] = [0+0.5, 0+0.5, 2*np.pi+0.5, 1.41+0.7, -1.70+0.7, 0.3, 0.5, 1.02+2.7,  0.5, 2.3,
    #                   1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000, 1000, 1000]
    # X_bounds.add(bounds=Bounds(x_min_2, x_max_2, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_3 = np.zeros((nq + nq, 3))
    # x_max_3 = np.zeros((nq + nq, 3))
    # x_min_3[:, 0] = [0-0.5, -1.2, 2*np.pi-0.7, 1.41-0.7, -1.70-0.7, -0.5, -0.5, 1.02-2.7, -0.5, 0,
    #                  -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_3[:, 0] = [0+0.5,  0,   2*np.pi+0.7, 1.41+0.7, -1.70+0.7, 0.3, 0.5, 1.02+2.7, 0.5, 2.3,
    #                   1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000]
    # x_min_3[:, 1] = x_min_3[:, 0]
    # x_max_3[:, 1] = x_max_3[:, 0]
    # x_min_3[:, 2] = x_min_3[:, 0]
    # x_max_3[:, 2] = x_max_3[:, 0]
    # X_bounds.add(bounds=Bounds(x_min_3, x_max_3, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_4 = np.zeros((nq + nq, 3))
    # x_max_4 = np.zeros((nq + nq, 3))
    # x_min_4[:, 0] = x_min_3[:, 1]
    # x_max_4[:, 0] = x_max_3[:, 1]
    # x_min_4[:, 1] = x_min_3[:, 1]
    # x_max_4[:, 1] = x_max_3[:, 1]
    # x_min_4[:, 2] = [0-0.5, 0-0.5, 2*np.pi-0.7, 1.41-0.7, -1.70-0.7, -0.5, -0.5, 1.02-2.7, -0.5, 0,
    #                  -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_4[:, 2] = [0+0.5, 0+0.5, 2*np.pi+0.7, 1.41+0.7, -1.70+0.7, 0.3, 0.5, 1.02+2.7, 0.5, 2.3,
    #                   1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000, 1000, 1000]
    # X_bounds.add(bounds=Bounds(x_min_4, x_max_4, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))
    #
    # x_min_5 = np.zeros((nq + nq, 3))
    # x_max_5 = np.zeros((nq + nq, 3))
    # x_min_5[:, 0] = x_min_4[:, 2]
    # x_max_5[:, 0] = x_max_4[:, 2]
    # x_min_5[:, 1] = [0-2.7, 0, 2*np.pi-0.7, 1.41-2.7, -1.70-2.7, -0.5, -0.5, 1.02-2.7, -0.5, 0,
    #                  -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_5[:, 1] = [0+2.7, 5, 6*np.pi, 1.41+2.7, -1.70+2.7, 0.3, 0.5, 1.02+2.7, 0.5, 2.3,
    #                   1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000, 1000, 1000]
    # x_min_5[:, 2] = [0-0.5, 0-0.5, 4*np.pi-0.5, 1.41-0.7, -1.70-0.7, -0.5, -0.5, 1.02-2.7, -0.5, 0,
    #                  -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
    # x_max_5[:, 2] = [0+0.5, 0+0.5, 4*np.pi+0.5, 1.41+0.7, -1.70+0.7, 0.3, 0.5, 1.02+2.7, 0.5, 2.3,
    #                   1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000, 1000, 1000]
    # X_bounds.add(bounds=Bounds(x_min_5, x_max_5, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))

    # Path constraint
    X_bounds = BoundsList()

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[0], q_mapping=q_mapping, qdot_mapping=q_mapping))
    X_bounds[0].min[:, 0] = [0, 0, -0.4323, 1.4415, -1.5564, 0, 0, 1.02, 0, 0, -10, -30, -1, -1, -1, -1, -1, -1, -1, -1]
    X_bounds[0].max[:, 0] = [0, 0, -0.4323, 1.4415, -1.5564, 0, 0, 1.02, 0, 0, 10, 3, 1, 1, 1, 1, 1, 1, 1, 1]
    X_bounds[0].min[1:3, 1] = [-1.2, -0.5]
    X_bounds[0].max[1:3, 1] = [0, 0.5]
    X_bounds[0].min[1:3, 2] = [-1.2, -0.5]
    X_bounds[0].max[1:3, 2] = [0, 0.5]

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[1], q_mapping=q_mapping, qdot_mapping=q_mapping))
    X_bounds[1].min[:3, 0] = [-0.5, -1.2, -0.5]
    X_bounds[1].max[:3, 0] = [0.5, 0, 0.5]
    X_bounds[1].min[1:3, 1] = [-1.2, -0.5]
    X_bounds[1].max[1:3, 1] = [0, 0.5]
    X_bounds[1].min[:3, 2] = [-0.5, -0.5, -0.5]  # 0.05
    X_bounds[1].max[:3, 2] = [0.5, 0.5, 0.5]  # 0.05

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[2], q_mapping=q_mapping, qdot_mapping=q_mapping))
    X_bounds[2].min[:3, 0] = [-0.5, -0.5, -0.5]
    X_bounds[2].max[:3, 0] = [0.5, 0.5, 0.5]
    X_bounds[2].min[1, 1] = 0
    X_bounds[2].max[1, 1] = 10
    X_bounds[2].min[:3, 2] = [-0.5, -0.5, Salto1 * 2 * np.pi - 0.5]  # 0.05
    X_bounds[2].max[:3, 2] = [0.5, 0.5, Salto1 * 2 * np.pi + 0.5]  # 0.05

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[3], q_mapping=q_mapping, qdot_mapping=q_mapping))
    X_bounds[3].min[:3, 0] = [-0.5, -0.5, Salto1 * 2 * np.pi - 0.5]
    X_bounds[3].max[:3, 0] = [0.5, 0.5, Salto1 * 2 * np.pi + 0.5]
    X_bounds[3].min[1:3, 1] = [-1.2, Salto1 * 2 * np.pi - 0.5]
    X_bounds[3].max[1:3, 1] = [0, Salto1 * 2 * np.pi + 0.5]
    X_bounds[3].min[1:3, 2] = [-1.2, Salto1 * 2 * np.pi - 0.5]
    X_bounds[3].max[1:3, 2] = [0, Salto1 * 2 * np.pi + 0.5]

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[4], q_mapping=q_mapping, qdot_mapping=q_mapping))
    X_bounds[4].min[:3, 0] = [-0.5, -1.2, Salto1 * 2 * np.pi - 0.5]
    X_bounds[4].max[:3, 0] = [0.5, 0, Salto1 * 2 * np.pi + 0.5]
    X_bounds[4].min[1:3, 1] = [-1.2, Salto1 * 2 * np.pi - 0.5]
    X_bounds[4].max[1:3, 1] = [0, Salto1 * 2 * np.pi + 0.5]
    X_bounds[4].min[:3, 2] = [-0.5, -0.5, Salto1 * 2 * np.pi - 0.5]  # 0.05
    X_bounds[4].max[:3, 2] = [0.5, 0.5, Salto1 * 2 * np.pi + 0.5]  # 0.05

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model[5], q_mapping=q_mapping, qdot_mapping=q_mapping))
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
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min],
            [100000, 100000, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min],
            [100000, 100000, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [0, 0, 0, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min],
            [0, 0, 0, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min],
            [100000, 100000, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [-100000, -100000, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min],
            [100000, 100000, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )
    u_bounds.add(
        bounds=Bounds(
            [0, 0, 0, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min, tau_min],
            [0, 0, 0, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max, tau_max],
        )
    )

    nq = q_mapping.to_first.len
    q = np.load(f"Results_MultiStart_Sauteur/Q_Salto1{Salto1}_Salto2{Salto2}_DoF8_weight{weight}_random{i_rand}.npy")
    qdot = np.load(
        f"Results_MultiStart_Sauteur/Qdot_Salto1{Salto1}_Salto2{Salto2}_DoF8_weight{weight}_random{i_rand}.npy"
    )

    q_reshape = np.zeros((nq, np.shape(q)[1]))
    q_reshape[:6, :] = q[:6, :]
    q_reshape[7, :] = q[6, :]
    q_reshape[9, :] = q[7, :]

    qdot_reshape = np.zeros((nq, np.shape(qdot)[1]))
    qdot_reshape[:6, :] = qdot[:6, :]
    qdot_reshape[7, :] = qdot[6, :]
    qdot_reshape[9, :] = qdot[7, :]

    q_old = np.vstack((q_reshape, qdot_reshape))

    nu = nq
    u = np.load(f"Results_MultiStart_Sauteur/U_Salto1{Salto1}_Salto2{Salto2}_DoF8_weight{weight}_random{i_rand}.npy")
    u_reshape = np.zeros((nu, np.shape(u)[1]))
    u_reshape[:6, :] = u[:6, :]
    u_reshape[7, :] = u[6, :]
    u_reshape[9, :] = u[7, :]
    u_old = u_reshape

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(q_old[:, : number_shooting_points[0] + 1], interpolation=InterpolationType.EACH_FRAME)
    x_init.add(
        q_old[:, number_shooting_points[0] + 1 : number_shooting_points[0] + number_shooting_points[1] + 2],
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        q_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + 2 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + 3,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        q_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + 3 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + 4,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        q_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + 4 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + number_shooting_points[4]
            + 5,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        q_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + number_shooting_points[4]
            + 5 :,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )

    u_init = InitialGuessList()
    u_init.add(u_old[:, : number_shooting_points[0]], interpolation=InterpolationType.EACH_FRAME)
    u_init.add(
        u_old[:, number_shooting_points[0] + 1 : number_shooting_points[0] + number_shooting_points[1] + 1],
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_init.add(
        u_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + 2 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + 2,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_init.add(
        u_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + 3 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + 3,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_init.add(
        u_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + 4 : number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + number_shooting_points[4]
            + 4,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_init.add(
        u_old[
            :,
            number_shooting_points[0]
            + number_shooting_points[1]
            + number_shooting_points[2]
            + number_shooting_points[3]
            + number_shooting_points[4]
            + 5 : -1,
        ],
        interpolation=InterpolationType.EACH_FRAME,
    )

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
        q_mapping=all_mapping,
        qdot_mapping=all_mapping,
        tau_mapping=all_mapping,
        n_threads=2,
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

        for weight in Weight_choices:
            for i_rand in range(5):

                # ----------------------------------------------- Sauteur_8 --------------------------------------------------------
                f = open(f"Historique_{Date}.txt", "a+")
                f.write(f"\n\n\nSalto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand} : ")
                f.close()

                tic = time()
                ocp = prepare_ocp_back_back(
                    path_model_cheville=path_model_cheville,
                    lut_verticale=lut_verticale,
                    lut_horizontale=lut_horizontale,
                    weight=weight,
                    Salto1=Salto1,
                    Salto2=Salto2,
                )
                sol = ocp.solve(
                    show_online_optim=False,
                    solver_options={
                        "hessian_approximation": "limited-memory",
                        "max_iter": 5000,
                        "ipopt.tol": 1e-5,
                    },
                )
                toc = time() - tic
                print(f"Time to solve weight={weight}, random={i_rand}: {toc}sec")

                q = ocp.nlp[0].mapping["q"].to_second.map(sol.states[0]["q"])
                qdot = ocp.nlp[0].mapping["qdot"].to_second.map(sol.states[0]["qdot"])
                u = ocp.nlp[0].mapping["tau"].to_second.map(sol.controls[0]["tau"])
                t = sol.parameters["time"]
                for i in range(1, len(sol.states)):
                    q = np.hstack((q, ocp.nlp[i].mapping["q"].to_second.map(sol.states[i]["q"])))
                    qdot = np.hstack((qdot, ocp.nlp[i].mapping["qdot"].to_second.map(sol.states[i]["qdot"])))
                    u = np.hstack((u, ocp.nlp[i].mapping["q"].to_second.map(sol.controls[i]["tau"])))

                if sol.status == 0:
                    f = open(f"Historique_{Date}.txt", "a+")
                    f.write(f"Success \n")
                    f.close()
                    np.save(
                        f"Results_MultiStart_Sauteur/Q_Salto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand}",
                        q,
                    )
                    np.save(
                        f"Results_MultiStart_Sauteur/Qdot_Salto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand}",
                        qdot,
                    )
                    np.save(
                        f"Results_MultiStart_Sauteur/U_Salto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand}",
                        u,
                    )
                    np.save(
                        f"Results_MultiStart_Sauteur/T_Salto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand}",
                        t,
                    )

                    # ------------------------------------------ + Tronc ---------------------------------------------------
                    f = open(f"Historique_{Date}.txt", "a+")
                    f.write(f"Salto1{Salto1}_Salto2{Salto2}_DoF7_weight{weight}_random{i_rand} : ")
                    f.close()

                    tic = time()
                    ocp = prepare_ocp_back_back_tronc(
                        path_model_cheville=path_model_cheville,
                        lut_verticale=lut_verticale,
                        lut_horizontale=lut_horizontale,
                        weight=weight,
                        i_rand=i_rand,
                        Salto1=Salto1,
                        Salto2=Salto2,
                    )
                    sol = ocp.solve(
                        show_online_optim=False,
                        solver_options={
                            "hessian_approximation": "limited-memory",
                            "max_iter": 5000,
                            "ipopt.tol": 1e-5,
                        },
                    )
                    toc = time() - tic
                    print(f"Time to solve weight={weight}, random={i_rand}: {toc}sec")

                    q = ocp.nlp[0].mapping["q"].to_second.map(sol.states[0]["q"])
                    qdot = ocp.nlp[0].mapping["qdot"].to_second.map(sol.states[0]["qdot"])
                    u = ocp.nlp[0].mapping["tau"].to_second.map(sol.controls[0]["tau"])
                    t = sol.parameters["time"]
                    for i in range(1, len(sol.states)):
                        q = np.hstack((q, ocp.nlp[i].mapping["q"].to_second.map(sol.states[i]["q"])))
                        qdot = np.hstack((qdot, ocp.nlp[i].mapping["qdot"].to_second.map(sol.states[i]["qdot"])))
                        u = np.hstack((u, ocp.nlp[i].mapping["q"].to_second.map(sol.controls[i]["tau"])))

                    if sol.status == 0:
                        f = open(f"Historique_{Date}.txt", "a+")
                        f.write(f"Success \n")
                        f.close()
                        np.save(
                            f"Results_MultiStart_Sauteur/Q_Salto1{Salto1}_Salto2{Salto2}_DoF7_weight{weight}_random{i_rand}",
                            q,
                        )
                        np.save(
                            f"Results_MultiStart_Sauteur/Qdot_Salto1{Salto1}_Salto2{Salto2}_DoF7_weight{weight}_random{i_rand}",
                            qdot,
                        )
                        np.save(
                            f"Results_MultiStart_Sauteur/U_Salto1{Salto1}_Salto2{Salto2}_DoF7_weight{weight}_random{i_rand}",
                            u,
                        )
                        np.save(
                            f"Results_MultiStart_Sauteur/T_Salto1{Salto1}_Salto2{Salto2}_DoF7_weight{weight}_random{i_rand}",
                            t,
                        )

                        # ------------------------------------ + Tronc + coude ---------------------------------------------
                        f = open(f"Historique_{Date}.txt", "a+")
                        f.write(f"Salto1{Salto1}_Salto2{Salto2}_DoF8_weight{weight}_random{i_rand} : ")
                        f.close()

                        tic = time()
                        ocp = prepare_ocp_back_back_tronc_coude(
                            path_model_cheville=path_model_cheville,
                            lut_verticale=lut_verticale,
                            lut_horizontale=lut_horizontale,
                            weight=weight,
                            i_rand=i_rand,
                            Salto1=Salto1,
                            Salto2=Salto2,
                        )
                        sol = ocp.solve(
                            show_online_optim=False,
                            solver_options={
                                "hessian_approximation": "limited-memory",
                                "max_iter": 5000,
                                "ipopt.tol": 1e-5,
                            },
                        )
                        toc = time() - tic
                        print(f"Time to solve weight={weight}, random={i_rand}: {toc}sec")

                        q = ocp.nlp[0].mapping["q"].to_second.map(sol.states[0]["q"])
                        qdot = ocp.nlp[0].mapping["qdot"].to_second.map(sol.states[0]["qdot"])
                        u = ocp.nlp[0].mapping["tau"].to_second.map(sol.controls[0]["tau"])
                        t = sol.parameters["time"]
                        for i in range(1, len(sol.states)):
                            q = np.hstack((q, ocp.nlp[i].mapping["q"].to_second.map(sol.states[i]["q"])))
                            qdot = np.hstack((qdot, ocp.nlp[i].mapping["qdot"].to_second.map(sol.states[i]["qdot"])))
                            u = np.hstack((u, ocp.nlp[i].mapping["q"].to_second.map(sol.controls[i]["tau"])))

                        if sol.status == 0:
                            f = open(f"Historique_{Date}.txt", "a+")
                            f.write(f"Success \n")
                            f.close()
                            np.save(
                                f"Results_MultiStart_Sauteur/Q_Salto1{Salto1}_Salto2{Salto2}_DoF8_weight{weight}_random{i_rand}",
                                q,
                            )
                            np.save(
                                f"Results_MultiStart_Sauteur/Qdot_Salto1{Salto1}_Salto2{Salto2}_DoF8_weight{weight}_random{i_rand}",
                                qdot,
                            )
                            np.save(
                                f"Results_MultiStart_Sauteur/U_Salto1{Salto1}_Salto2{Salto2}_DoF8_weight{weight}_random{i_rand}",
                                u,
                            )
                            np.save(
                                f"Results_MultiStart_Sauteur/T_Salto1{Salto1}_Salto2{Salto2}_DoF8_weight{weight}_random{i_rand}",
                                t,
                            )

                            # ---------------------------------- + Tronc + coude2D -------------------------------------------
                            f = open(f"Historique_{Date}.txt", "a+")
                            f.write(f"Salto1{Salto1}_Salto2{Salto2}_DoF14_weight{weight}_random{i_rand} : ")
                            f.close()

                            tic = time()
                            ocp = prepare_ocp_back_back_tronc_coude2D(
                                path_model_cheville=path_model_cheville,
                                lut_verticale=lut_verticale,
                                lut_horizontale=lut_horizontale,
                                weight=weight,
                                i_rand=i_rand,
                                Salto1=Salto1,
                                Salto2=Salto2,
                            )
                            sol = ocp.solve(
                                show_online_optim=False,
                                solver_options={
                                    "hessian_approximation": "limited-memory",
                                    "max_iter": 5000,
                                    "ipopt.tol": 1e-5,
                                },
                            )
                            toc = time() - tic
                            print(f"Time to solve weight={weight}, random={i_rand}: {toc}sec")

                            q = ocp.nlp[0].mapping["q"].to_second.map(sol.states[0]["q"])
                            qdot = ocp.nlp[0].mapping["qdot"].to_second.map(sol.states[0]["qdot"])
                            u = ocp.nlp[0].mapping["tau"].to_second.map(sol.controls[0]["tau"])
                            t = sol.parameters["time"]
                            for i in range(1, len(sol.states)):
                                q = np.hstack((q, ocp.nlp[i].mapping["q"].to_second.map(sol.states[i]["q"])))
                                qdot = np.hstack(
                                    (qdot, ocp.nlp[i].mapping["qdot"].to_second.map(sol.states[i]["qdot"]))
                                )
                                u = np.hstack((u, ocp.nlp[i].mapping["q"].to_second.map(sol.controls[i]["tau"])))

                            if sol.status == 0:
                                f = open(f"Historique_{Date}.txt", "a+")
                                f.write(f"Success \n")
                                f.close()
                                np.save(
                                    f"Results_MultiStart_Sauteur/Q_Salto1{Salto1}_Salto2{Salto2}_DoF14_weight{weight}_random{i_rand}",
                                    q,
                                )
                                np.save(
                                    f"Results_MultiStart_Sauteur/Qdot_Salto1{Salto1}_Salto2{Salto2}_DoF14_weight{weight}_random{i_rand}",
                                    qdot,
                                )
                                np.save(
                                    f"Results_MultiStart_Sauteur/U_Salto1{Salto1}_Salto2{Salto2}_DoF14_weight{weight}_random{i_rand}",
                                    u,
                                )
                                np.save(
                                    f"Results_MultiStart_Sauteur/T_Salto1{Salto1}_Salto2{Salto2}_DoF14_weight{weight}_random{i_rand}",
                                    t,
                                )

                            else:
                                f = open(f"Historique_{Date}.txt", "a+")
                                f.write(f"Fail \n")
                                f.close()
                        else:
                            f = open(f"Historique_{Date}.txt", "a+")
                            f.write(f"Fail \n")
                            f.close()
                    else:
                        f = open(f"Historique_{Date}.txt", "a+")
                        f.write(f"Fail \n")
                        f.close()
                else:
                    f = open(f"Historique_{Date}.txt", "a+")
                    f.write(f"Fail \n")
                    f.close()

    #####################################################################################################################
    import bioviz

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
    plt.show()

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
    plt.show()

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

    DirertoryFlies = os.listdir("/home/user/Documents/Programmation/Eve/Toile/Position_massPoints")
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
                f"/home/user/Documents/Programmation/Eve/Toile/Position_massPoints/{DirertoryFlies[BestFileIndex]}"
            )
            q_toile[:, j] = data.T.flatten()

    b = bioviz.Viz("/home/user/Documents/Programmation/Eve/Modeles/jumper_sansPieds_rootPied_bioviz.bioMod")
    b.load_movement(np.vstack((q_toile, q)))
    b.exec()
