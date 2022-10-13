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


def custom_spring_const(pn: PenaltyNode, lut_verticale,lut_horizontale) -> cas.MX:  # ajout de la force de la toile comme etant la force appliquee a la cheville

    # nq = int(pn.nlp.states.shape /2) #nombre de degres de liberté

    val_contrainte = []

    u_i = pn.nlp.variable_mappings["tau"].to_second.map(pn.nlp.controls["tau"].mx)
    q_i = pn.nlp.variable_mappings['q'].to_second.map(pn.nlp.states['q'].mx)

    Markers = pn.nlp.model.markers(q_i)
    Marker_pied = Markers[0].to_mx()

    Force = cas.MX.zeros(3)
    Force[1] = lut_horizontale(Marker_pied[1:])
    Force[2] = lut_verticale(Marker_pied[1:])

    return_value = cas.vertcat((u_i[0] - Force[1]), (u_i[1] - Force[2]))

    val_contrainte = cas.Function("Force", [pn.nlp.states['q'].mx, pn.nlp.controls['tau'].mx], [return_value])(
        pn.nlp.states['q'].cx, pn.nlp.controls['tau'].cx)

    force_ressort = cas.MX.zeros(3)
    force_ressort[2] = 10 * 1.2 ** 2
    r_v = cas.vertcat(u_i[1]-force_ressort[2])
    v_c = cas.Function("Force", [pn.nlp.states['q'].mx, pn.nlp.controls['tau'].mx], [r_v])(
        pn.nlp.states['q'].cx, pn.nlp.controls['tau'].cx)

    return val_contrainte

def custom_spring_const_post(Q, lut_verticale, lut_horizontale,
                             model_path):  # calcul de la force de la toile sur la cheville apres optim pour pouvoir comparer
    m = biorbd.Model(model_path)
    Marker_pied = m.markers(Q)[0].to_mx()
    Force = cas.MX.zeros(3)
    Force[1] = lut_horizontale(Marker_pied[1:])
    Force[2] = lut_verticale(Marker_pied[1:])
    fun = cas.Function("Force_TrampoBed", [Q], [Marker_pied, Force])
    return fun

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

    tau_min, tau_max, tau_init = -100000, 100000, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", node=Node.END, index=1, weight=10000, phase=0, quadratic=False)  # etre le plus bas a la fin de la phase 0

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=0)

    # arriver avec les pieds au centre de la toile
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", phase=0, node=Node.START, index=0, weight=100,
                            target=np.zeros((1, 1)))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", phase=0, node=Node.START, index=1, weight=100,
                            target=np.zeros((1, 1)))

    #objectif sur le temps
    #objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, node=Node.END)

    # # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # --- Constraints --- #
    constraints = ConstraintList()

    # Constraint arm positivity
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.05, max_bound=0.6, phase=0)

    constraints.add(
        custom_spring_const,
        lut_verticale=lut_verticale,
        lut_horizontale=lut_horizontale,
        node=Node.ALL,
        min_bound=0,
        max_bound=0,
        phase=0,
    )

    # Path constraint
    X_bounds = BoundsList()

    X_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    X_bounds[0].min[:1, 1:] = [-0.3]
    X_bounds[0].max[:1, 1:] = [0.3]

    X_bounds[0].min[:, 0] = [-0.3, 0, -0.5,  -1, -30, -1]
    X_bounds[0].max[:, 0] = [0.3, 0, 0.5, 1, 12, 1]
    X_bounds[0].min[1:3, 1] = [-1.2, -0.5]
    X_bounds[0].max[1:3, 1] = [0, 0.5]
    X_bounds[0].min[1:3 ,2] = [-1.2, -0.5]
    X_bounds[0].max[1:3, 2] = [0, 0.5]

    # X_bounds[0].min[4:5, 2] = [0]
    # X_bounds[0].max[4:5, 2] = [0]

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

    x_init[0].init[1, :] = np.linspace(0, -1.2, 51)

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
    )
    return ocp


if __name__ == "__main__":

    Date = date.today()
    Date = Date.strftime("%d-%m-%y")
    f = open(f"Historique_{Date}.txt", "w+")
    f.write(" Debut ici \n\n\n")
    f.close()

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
    weight = 1000000
    i_rand = 10

    # ----------------------------------------------- Sauteur_8 --------------------------------------------------------
    f = open(f"Historique_{Date}.txt", "a+")
    f.write(f"\n\n\nSalto1{Salto1}_Salto2{Salto2}_DoF6_weight{weight}_random{i_rand} : ")
    f.close()

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
    print(f"Time to solve weight={weight}, random={i_rand}: {toc}sec")

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

    b = bioviz.Viz(model_path)
    b.load_movement(q)
    b.exec()
