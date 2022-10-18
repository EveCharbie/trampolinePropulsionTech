import bioviz
import pickle

model_path = "/home/lim/Documents/Jules/code_initiaux_Eve/collectesaut/SylvainMan_Sauteur_6DoF.bioMod"

path = '/home/lim/Documents/Jules/result_saut/phase0_force*100.pkl'
with open(path, 'rb') as file:
    q= pickle.load(file)
    qdot = pickle.load(file)
    u = pickle.load(file)
    t = pickle.load(file)

b = bioviz.Viz(model_path)
b.load_movement(q)
b.exec()