"""Contact entre sauteur et trampoline, en considérant seulement 6 degrés de liberté
    on se limite ici a la premiere phase : l'enfoncement du sauteur dans la toile,

"""

from time import time
import numpy as np
import casadi as cas
import biorbd_casadi as biorbd
import os
import matplotlib.pyplot as plt
from datetime import date
from IPython import embed
import pickle

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
)

def custom_dynamic(states, controls, parameters, nlp):

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    Markers = nlp.model.markers(q)
    Marker_pied = Markers[0].to_mx()

############### f contact  ######################
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

    qddot1 = nlp.model.ForwardDynamics(q, qdot, tau, None, f_contact_vec).to_mx()
##################################################
#########################test f ext###############
    #attention : la force est dans le repere globale, il faut donc la mettre dans le repere locale de la masse ponctuelle en ajoutant le moment associé

    force_vector = cas.MX.zeros(6)
    force_vector[4] = lut_horizontale(Marker_pied[1:]) #dans le global
    force_vector[5] = lut_verticale(Marker_pied[1:])  #dans le global

    force_vector[0] = q[0]*force_vector[5] - q[1]*force_vector[4]

    f_ext = biorbd.VecBiorbdSpatialVector()
    f_ext.append(biorbd.SpatialVector(force_vector))
    qddot = nlp.model.ForwardDynamics(q, qdot, tau, f_ext).to_mx()
##################################################
    # f_evalext = cas.Function("qddot", [q, qdot, tau, f_ext], [qddot])
    # f_evalcontact = cas.Function("qddot1", [q, qdot, tau, None, f_contact_vec], [qddot1])
    # ext = cas.evalf(f_evalext)
    # cont = cas.evalf(f_evalcontact)

    return DynamicsEvaluation(dxdt=cas.vertcat(qdot, qddot), defects=None)

def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):

    ConfigureProblem.configure_q(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic, expand=False)

def custom_spring_const(pn: PenaltyNode, lut_verticale, lut_horizontale) -> cas.MX: #ajout de la force de la toile comme etant la force appliquee a la cheville

    val_contrainte = []

    u_i = pn.nlp.variable_mappings["tau"].to_second.map(pn.nlp.controls["tau"].mx)
    q_i = pn.nlp.variable_mappings['tau'].to_second.map(pn.nlp.states['q'].mx)

    Markers = pn.nlp.model.markers(pn.nlp.states["q"].mx)
    Marker_pied = Markers[0].to_mx()

    Force = cas.MX.zeros(3)
    Force[1] = lut_horizontale(Marker_pied[1:])
    Force[2] = lut_verticale(Marker_pied[1:])

    return_value = cas.vertcat(u_i[0] - Force[1], u_i[1] - Force[2])

    val_contrainte = cas.Function("Force", [pn.nlp.states['q'].mx, pn.nlp.controls['tau'].mx], [return_value])(pn.nlp.states['q'].cx, pn.nlp.controls['tau'].cx)

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
    model_cheville = biorbd.Model(path_model_cheville)
    q_cheville = cas.MX.sym("q_cheville", 1)
    qdot_cheville = cas.MX.sym("q_dot_cheville", 1)

    nq = int(pn.nlp.states.shape / 2)

    q_mx = pn.nlp.states["q"].mx
    qdot_mx = pn.nlp.states["qdot"].mx
    tau_mx = pn.nlp.controls["tau"].mx
    func_cheville = cas.Function("torqueMax_cheville", [q_cheville, qdot_cheville],
                                 [model_cheville.torqueMax(q_cheville, qdot_cheville)[1].to_mx()])

    func_q_cheville = q_cheville_func(q_cheville)
    bound_cheville = func_cheville(func_q_cheville(q_mx[2]), -qdot_mx[2])

    min_bound = []
    bound = pn.nlp.model.torqueMax(q_mx, qdot_mx)[1].to_mx()
    min_boundz = cas.if_else(bound[2:nq] < minimal_tau, minimal_tau, bound[2:nq])
    min_boundz[0, 0] = 0
    min_bound.append(min_boundz)
    min_bound[0] = cas.vertcat(np.ones((2,)) * -1000000, min_bound[0])

    obj = []
    obj.append(tau_mx)
    obj_star = cas.vertcat(*obj)
    min_bound = cas.vertcat(*min_bound)

    constraint_min = BiorbdInterface.mx_to_cx("tau_actuator_constraints_min", min_bound -obj_star , pn.nlp.states["q"], pn.nlp.states["qdot"], pn.nlp.controls["tau"])

    return constraint_min


def tau_actuator_constraints_max(pn: PenaltyNode, path_model_cheville: str, minimal_tau: float = None) -> cas.MX:
    model_cheville = biorbd.Model(path_model_cheville)
    q_cheville = cas.MX.sym("q_cheville", 1)
    qdot_cheville = cas.MX.sym("q_dot_cheville", 1)

    nq = int(pn.nlp.states.shape / 2)

    q_mx = pn.nlp.states["q"].mx
    qdot_mx = pn.nlp.states["qdot"].mx
    tau_mx = pn.nlp.controls["tau"].mx
    func_cheville = cas.Function("torqueMax_cheville", [q_cheville, qdot_cheville],
                                 [model_cheville.torqueMax(q_cheville, qdot_cheville)[0].to_mx()])

    func_q_cheville = q_cheville_func(q_cheville)
    bound_cheville = func_cheville(func_q_cheville(q_mx[2]), -qdot_mx[2])

    max_bound = []
    bound = pn.nlp.model.torqueMax(q_mx, qdot_mx)[0].to_mx()
    max_boundz = cas.if_else(bound[2:nq] < minimal_tau, minimal_tau, bound[2:nq])
    max_boundz[0, 0] = cas.if_else(bound_cheville[:, 0] < minimal_tau, minimal_tau,bound_cheville[:, 0])
    max_bound.append(max_boundz)
    max_bound[0] = cas.vertcat(np.ones((2,)) * 1000000, max_bound[0])

    obj = []
    obj.append(tau_mx)
    obj_star = cas.vertcat(*obj)
    max_bound = cas.vertcat(*max_bound)

    constraint_max = BiorbdInterface.mx_to_cx("tau_actuator_constraints_max", obj_star - max_bound, pn.nlp.states["q"],
                                              pn.nlp.states["qdot"], pn.nlp.controls["tau"])
    return constraint_max

def prepare_ocp_back_back(path_model_cheville, lut_verticale, lut_horizontale, weight, Salto1, Salto2):
    # --- Options --- #
    model_path = "/home/lim/Documents/Jules/bioptim/bioptim/examples/getting_started/models/cube.bioMod"
    model_path_massToile = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_6DoF_massToile.bioMod"

    # Model path
    biorbd_model = (
        biorbd.Model(model_path)
    )

    nb_phases = 1

    number_shooting_points = (
        50,
    )

    final_time = (
        0.1,
    )

    tau_min, tau_max, tau_init = -10000, 10000, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", node=Node.END, index=1, weight=1000, phase=0, quadratic=False)  # etre le plus bas a la fin de la phase 0
    #objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=Node.START, index=1, weight=1, phase=0, quadratic=False)  #augmenter la norme de la vitesse initale

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=0)

    # arriver avec les pieds au centre de la toile
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", phase=0, node=Node.START, index=0, weight=100,
                            target=np.zeros((1, 1)))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", phase=0, node=Node.START, index=1, weight=100,
                            target=np.zeros((1, 1)))

    # # Dynamics
    dynamics = DynamicsList()
    # dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics = Dynamics(custom_configure, dynamic_function=custom_dynamic)

    # --- Constraints --- #
    constraints = ConstraintList()
    # constraints.add(
    #     tau_actuator_constraints_min, phase=0, node=Node.ALL, minimal_tau=50, path_model_cheville=path_model_cheville,
    #     min_bound=-np.inf, max_bound=0)
    # constraints.add(
    #     tau_actuator_constraints_max, phase=0, node=Node.ALL, minimal_tau=50, path_model_cheville=path_model_cheville,
    #     min_bound=-np.inf, max_bound=0)

    # constraints.add(custom_spring_const, lut_verticale=lut_verticale, lut_horizontale=lut_horizontale, node=Node.ALL,
    #                 min_bound=-2, max_bound=2, phase=0)

    # Constraint
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.05, max_bound=0.3, phase=0)

    # Path constraint
    X_bounds = BoundsList()

    vitesse_de_depart = -20 #np.sqrt(2 * 9.81 * 8)  # 2*g*hauteur_saut

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    X_bounds[0].min[:1, 1:] = [-0.3]
    X_bounds[0].max[:1, 1:] = [0.3]

    X_bounds[0].min[:, 0] = [-0.3, 0, -0.5,  -1, vitesse_de_depart, -1]
    X_bounds[0].max[:, 0] = [0.3, 0, 0.5, 1, 0, 1]
    X_bounds[0].min[1:3, 1] = [-1.2, -0.5]
    X_bounds[0].max[1:3, 1] = [0, 0.5]
    X_bounds[0].min[1:3 ,2] = [-1.2, -0.5]
    X_bounds[0].max[1:3, 2] = [0, 0.5]

    X_bounds[0].min[4:5, 1] = [vitesse_de_depart]
    X_bounds[0].max[4:5, 1] = [0]
    X_bounds[0].min[4:5, 2] = [0]
    X_bounds[0].max[4:5, 2] = [0]

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        bounds=Bounds(
            [-100000, -100000, tau_min], [100000, 100000, tau_max]
        )
    )

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(NoisedInitialGuess(
        [0,0,0,0,0,0],  # (nq + nqdot)
        bounds=X_bounds[0],
        noise_magnitude=0.2,
        n_shooting=number_shooting_points[0],
        bound_push=0.01,
        seed=i_rand,
    )
    )
    #position
    x_init[0].init[1, :] = np.linspace(0, -1.2, 51)
    # vitesse jambe z
    x_init[0].init[4:5, :] = np.linspace(-vitesse_de_depart, 0, 51)

    u_init = InitialGuessList()
    u_init.add(NoisedInitialGuess(
        [0] * 3,  # ntorque
        bounds=u_bounds[0],
        noise_magnitude=0.01,
        n_shooting=number_shooting_points[0] - 1,
        bound_push=0.1,
        seed=i_rand,
    ))

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
        # use_sx=False,
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

    # ----------------------------------------------- Sauteur_8 --------------------------------------------------------


    tic = time()
    ocp = prepare_ocp_back_back(path_model_cheville=path_model_cheville, lut_verticale=lut_verticale,
                                lut_horizontale=lut_horizontale, weight=weight, Salto1=Salto1, Salto2=Salto2, )

    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(100000)
    solver.set_tol(1e-3)
    solver.set_constr_viol_tol(1e-3)
    solver.set_linear_solver("ma57")
    sol = ocp.solve(solver)

    toc = time() - tic
    print(f"Time to solve (weight={weight}, random={i_rand}): {toc} sec")

    q = ocp.nlp[0].variable_mappings["q"].to_second.map(sol.states["q"])
    qdot = ocp.nlp[0].variable_mappings["qdot"].to_second.map(sol.states["qdot"])
    u = ocp.nlp[0].variable_mappings["tau"].to_second.map(sol.controls["tau"])
    t = sol.parameters["time"]
    # for i in range(1, len(sol.states)):
    q = np.hstack((q, ocp.nlp[0].variable_mappings["q"].to_second.map(sol.states["q"])))
    qdot = np.hstack((qdot, ocp.nlp[0].variable_mappings["qdot"].to_second.map(sol.states["qdot"])))
    u = np.hstack((u, ocp.nlp[0].variable_mappings["q"].to_second.map(sol.controls["tau"])))

    ##############################################################################################################################
    import bioviz

    model_path = "/home/lim/Documents/Jules/bioptim/bioptim/examples/getting_started/models/cube.bioMod"

    # path = '/home/lim/Documents/Jules/result_saut/' + 'cube_phase0_fcontact' + '.pkl'
    # with open(path, 'wb') as file:
    #     pickle.dump(q, file)
    #     pickle.dump(qdot, file)
    #     pickle.dump(u, file)
    #     pickle.dump(t, file)

    b = bioviz.Viz(model_path)
    b.load_movement(q)
    b.exec()


    #avec toile
    #
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
    # DirertoryFlies = os.listdir("/home/lim/Documents/Jules/code_initiaux_Eve/Position_massPoints")
    # q_toile = np.zeros((15 * 3, np.shape(q)[1]))
    # for j in range(np.shape(q)[1]):
    #     BestFileIndex = 0
    #     BestFileNum = 1000
    #     for iFile in range(len(DirertoryFlies)):
    #         if "y" in DirertoryFlies[iFile]:
    #             first_ = DirertoryFlies[iFile].index("_")
    #             firsty = DirertoryFlies[iFile].index("y")
    #             yy = float(DirertoryFlies[iFile][first_ + 1: firsty])
    #             second_ = DirertoryFlies[iFile][first_ + 1:].index("_")
    #             firstz = DirertoryFlies[iFile].index("z")
    #             zz = float(DirertoryFlies[iFile][first_ + 1 + second_ + 1: firstz])
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