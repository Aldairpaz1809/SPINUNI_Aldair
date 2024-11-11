import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import pandas as pd

# Definimos la energía libre como función de angulo (phi_M)
def energia_libre(angulo, parametros):
    # Obtenemos los parámetros
    M = parametros["M"]
    Ku = parametros["Hu"]*M/2
    Kc = parametros["Hc"]*M/2
    H = parametros["H"]
    phi_H = np.radians(parametros["phi_H"])
    phi_u = np.radians(parametros["phi_u"])
    # Parámetro a ajustar
    phi_M = angulo[0]  # Aseguramos que sea un escalar
    e_u=np.array([np.cos(phi_u),np.sin(phi_u),0])
    # Componentes del vector de campo H
    H_vec = H * np.array([np.cos(phi_H), np.sin(phi_H), 0])

    # Componentes del vector de magnetización M
    alpha_1 = np.cos(phi_M)
    alpha_2 = np.sin(phi_M)
    alpha_3 = 0
    M_vec = M*np.array([alpha_1, alpha_2, alpha_3])

    # Cálculo de términos de energía libre
    # Término de Zeeman: -H·M
    zeeman = -np.dot(H_vec, M_vec)
    
    # Término de anisotropía magnética cúbica
    ani_mag_cris = - Kc * (alpha_1**2 * alpha_2**2+alpha_2**2*alpha_3**2+alpha_3**2*alpha_1**2) # Término simplificado
     
    # Término de anisotropía uniaxial
    ani_uniaxial = -Ku * (np.dot(e_u, M_vec))**2/np.dot(M_vec,M_vec)

    return zeeman + ani_mag_cris + ani_uniaxial

def angulo(parametros,paso=1e-3):
    H = parametros["H"]
    H_vals = H * np.concatenate((np.arange(-1, 1, paso), np.arange(1, -1,-paso)))
    
    phi_M_vals = [np.radians(parametros["phi_H"])]  # Valor inicial para phi_M
    for h in H_vals:
        parametros["H"] = h
        res = minimize(energia_libre, x0=[phi_M_vals[-1]], args=(parametros,))
        phi_M_vals.append(res.x[0])        
    phi_M_vals.pop(0)
    return np.array(phi_M_vals, float), H_vals


def is_numeric(value):
    return pd.to_numeric(value, errors='coerce')

def leer_archivo(path):
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = ['Rxx (ohm)', 'Rxy (ohm)', 'EM Current (percentage)', 'Vxx (mV)',
                  'Vxy (mV)', 'Magnetic Field (T)', 'Time stamp', 'T sample (K)',
                  'T valve (k)', 'DC current (mA)', 'angle (°)']

    # Convertir a numérico y eliminar filas no numéricas
    df_numeric = df.map(is_numeric)
    df_cleaned = df_numeric.dropna()
    return df_cleaned


def angulo2(parametros,paso=1e-3):
    H = parametros["H"]
    #H_vals = H * np.concatenate((np.arange(-1, 1, paso), np.arange(1, -1,-paso))) 
    phi_M_vals = [np.radians(parametros["phi_H"])]  # Valor inicial para phi_M
    res = minimize(energia_libre, x0=[phi_M_vals[-1]], args=(parametros,))
    phi_M_vals.append(res.x[0])        
    phi_M_vals.pop(0)
    return np.array(phi_M_vals, float)
