import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def generate_naca_aerofoil(c,camber, thickness, n_points): 
    # Provides equal spaces along the chord
    base = 0
    x = np.linspace(0, c, n_points) 

    # Calculating the camber (curve of the aerofoil)
    camber_line = camber * (2*(x/c)-(x/c)**2 + 0.5 * (x/c))

    # Calculating the thickness distribution along the aerofoil
    thickness_dist = 5 * thickness * (0.2969 * np.sqrt(x/c) -
                                      0.1260 * (x/c) -
                                      0.3516 * (x/c)**2 +
                                      0.2843 * (x/c)**3 -
                                      0.1015 * (x/c)**4)

    # Assigning upper and lower surfaces of aerofoil
    upper_surface = camber_line + thickness_dist
    lower_surface = camber_line - thickness_dist

    return x, upper_surface, lower_surface

def generate_streamlines(aerofoil_x, aerofoil_y_upper, aerofoil_y_lower, aoa, grid_size):
    X, Y = np.meshgrid(np.linspace(-1, 2, grid_size),
                       np.linspace(-1, 1, grid_size))
    
    u_freestream = 1.0
    aoa_rad = np.radians(aoa)
    
    # Velocity components
    U_inf_x = u_freestream * np.cos(aoa_rad)
    U_inf_y = u_freestream * np.sin(aoa_rad)
    
    # Creating velocity field
    u = U_inf_x * np.ones_like(X)
    v = U_inf_y * np.ones_like(Y)
    
    strength = u_freestream * 2 * np.pi
    
    for xi, yi_upper, yi_lower in zip(aerofoil_x, aerofoil_y_upper, aerofoil_y_lower):
        r_upper = np.sqrt((X - xi)**2 + (Y - yi_upper)**2)
        theta_upper = np.arctan2(Y - yi_upper, X - xi)
        
        r_lower = np.sqrt((X - xi)**2 + (Y - yi_lower)**2)
        theta_lower = np.arctan2(Y - yi_lower, X - xi)
        
        u += (strength / (2 * np.pi * r_upper)) * np.cos(theta_upper)
        v += (strength / (2 * np.pi * r_upper)) * np.sin(theta_upper)
        
        u += (strength / (2 * np.pi * r_lower)) * np.cos(theta_lower)
        v += (strength / (2 * np.pi * r_lower)) * np.sin(theta_lower)
        
    # Streaming function
    psi = U_inf_x * Y - U_inf_y * X
    
    return X, Y, psi

# Function to input aerofoil parameters and store in cache
cache = {}
cache_order = []
cache_limit = 3

def delete_and_cache(parameters, current_aerofoil_key):
    cache_key = "aerofoil" + str(len(cache_order) + 1)
    cache[cache_key] = parameters[current_aerofoil_key]
    
    # Tracking the order of the aerofoils in cache
    cache_order.append(cache_key)
    
    # Removing current aerofoil from parameters
    parameters.pop(current_aerofoil_key)
    
    # Checking if the cache size exceeds the limit
    if len(cache) > cache_limit:
        fifo_key = cache_order.pop(0)
        cache.pop(fifo_key)

def overwrite_parameters(parameters, current_aerofoil_key):
    parameters[current_aerofoil_key] = {
        "camber": float(input("Enter camber: ")),
        "thickness": float(input("Enter thickness: ")),
        "aoa": float(input("Enter angle of attack: "))
    }

def retrieve_from_cache(parameters, current_aerofoil_key, key):
    # Retrieving back data from cache
    if key in cache:
        parameters[current_aerofoil_key] = cache.pop(key)
        cache_order.remove(key)
        
def create_panels(x, y_upper, y_lower):
    panels = []
    
    for i in range(len(x)-1):
        panels.append({
            'start': (x[i+1], y_lower[i+1]),
            'end': (x[i], y_lower[i])
        })
    return panels

def calculate_lift_drag(panels, aoa_deg, chord_length, span=1.0, airflow_speed =10.0, air_density=1.225):
    aoa_rad = np.radians(aoa_deg)
    lift_force = 0.0
    drag_force = 0.0
    
    for panel in panels:
        Cp = -0.5 if panel['start'][1] > 0 else 0.5
        
        dx = panel['end'][0] - panel['start'][0]
        dy = panel['end'][1] - panel['start'][1]
        length = np.sqrt(dx**2 + dy**2)
        
        force_perpendicular = -Cp * length
        force_parallel = Cp * length
        
        lift_force += (force_perpendicular * np.cos(aoa_rad)-
                       force_parallel * np.sin(aoa_rad))
        drag_force += (force_perpendicular * np.sin(aoa_rad)+
                       force_parallel * np.cos(aoa_rad))
        
    dynamic_pressure = 0.5 * air_density * airflow_speed**2
    lift_coefficient = (lift_force / (dynamic_pressure * chord_length))
    drag_coefficient = (drag_force / (dynamic_pressure * chord_length))
    
    # Converting into actual forces
    ref_area = chord_length * span
    lift_N = lift_coefficient * dynamic_pressure * ref_area
    drag_N = drag_coefficient * dynamic_pressure * ref_area
    
    # Converting forces into pound force
    lift_lbf = round(lift_N * 0.224809)
    drag_lbf = round(drag_N * 0.224809)
    
    return lift_coefficient, drag_coefficient

def calculate_influence(panel_j, panel_i):
    # placeholder
    # planning to calculate the influence coefficient of panel_j on panel_i
    return 0

def calculate_pressure_distribution(aerofoil, airflow_properties, airflow_speed, angle_of_attack):
    # Calculating pressure distribution along the aerofoil
    pressure_coefficients = {}
    num_panels = len(aerofoil.panels)
    gamma = np.zeros(num_panels)
    A = np.zeros((num_panels,num_panels))
    B = np.zeros(num_panels)

    # Populating the matrix and right-hand side vector
    for i in range(num_panels):
        for j in range(num_panels):
            A[i, j] = calculate_influence(aerofoil.panels[j], aerofoil.panels[i])
        B[i] = -airflow_speed * np.cos(angle_of_attack)

    # hopefully solving for circulation strengths using the Gauss-Jordan algorithm
    gamma = gauss_jordan(A,B)

    # calculating pressure coefficients for each panel
    for i in range(num_panels):
        Cp = 1 - (gamma[i] / airflow_speed) ** 2
        pressure_coefficients[aerofoil.panels[i]] = Cp

    return pressure_coefficients

def gauss_jordan(matrix, vector):
    """
    using the gauss-jordan elimination method solving the linear system should simple
    """
    n = len(matrix)
    matrix = matrix.astype(float)
    vector = vector.astype(float)

    # ELIMINATING TIME
    for i in range(n):
        divisor = matrix[i, i]
        if divisor != 0:
            matrix[i] /= divisor
            vector[i] /= divisor

        for k in range(i + 1, n):
            factor = matrix[k, i]
            matrix[k] -= factor * matrix[i]
            vector[k] -= factor * vector[i]

    # Now substitution
    for i in range(n -1, -1, -1):
        for k in range(i - 1, -1, -1):
            factor = matrix[k, i]
            matrix[k] -= factor * matrix[i]
            vector[k] -= factor * vector[i]

    return vector

def create_gui():
    # Dictionary to store the parameters and canvas/plot
    state = {
        "chord": 1.0,
        "camber": 0.00,
        "thickness": 0.12,
        "aoa": 0,
        "n_points": 100,
        "grid_size": 100,
        "fig": None,
        "fig2": None,
        "ax": None,
        "bar": None,
        "canvas1": None,
        "canvas2": None,
    }

    # Creating the main window
    root = tk.Tk()
    root.title("Aerofoil Simulator")

    # Creating the control panel on the left side of main window
    control_panel = tk.Frame(root, padx=10, pady=10)
    control_panel.pack(side=tk.LEFT, fill=tk.Y)

    # Camber slider (adjusts the camber of the aerofoil)
    tk.Label(control_panel, text="Camber (%)").pack()
    camber_slider = tk.Scale(control_panel, from_=-20, to=20, resolution=0.1, orient=tk.HORIZONTAL, command=lambda e: update_plot(state, camber_slider, thickness_slider, aoa_slider))
    camber_slider.set(state["camber"] * 100)
    camber_slider.pack()
    
    # Camber slider (adjusts the thickness of the aerofoil)
    tk.Label(control_panel, text="Thickness (%)").pack()
    thickness_slider = tk.Scale(control_panel, from_=1, to=20, resolution=0.1, orient=tk.HORIZONTAL, command=lambda e: update_plot(state, camber_slider, thickness_slider, aoa_slider))
    thickness_slider.set(state["thickness"] * 100)
    thickness_slider.pack()
    
    # Angle of Attack slider (adjusts the angle of attack of the simulation)
    tk.Label(control_panel, text="Angle of Attack (°)").pack()
    aoa_slider = tk.Scale(control_panel, from_=-15, to=15, resolution=1, orient=tk.HORIZONTAL, command=lambda e: update_plot(state, camber_slider, thickness_slider, aoa_slider))
    aoa_slider.set(state["aoa"])
    aoa_slider.pack()
    
    # Recent Aerofoils list
    tk.Label(control_panel, text="Recent Aerofoils").pack()
    recent_aerofoils_list = tk.Listbox(control_panel, height=3)
    recent_aerofoils_list.pack()
    
    def update_recent_aerofoils_list():
        recent_aerofoils_list.delete(0, tk.END)
        for key in cache_order:
            recent_aerofoils_list.insert(tk.END, key)
            
    def insert_aerofoil():
        camber = simpledialog.askfloat("Camber", "Enter camber (%):", minvalue=-20, maxvalue=20)
        thickness = simpledialog.askfloat("Camber", "Enter thickness (%):", minvalue=1, maxvalue=20)
        aoa = simpledialog.askfloat("Angle of Attack", "Enter angle of attack (°):", minvalue=-15, maxvalue=15)
        
        if camber is None or thickness is None or aoa is None:
            return
        
        camber = max(-20, min(20, camber))
        thickness = max(-20, min(20, thickness))
        aoa = max(-20, min(20, aoa))
        
        camber_slider.set(camber)
        thickness_slider.set(thickness)
        aoa_slider.set(aoa)
        state["camber"] = camber / 100
        state["thickness"] = thickness / 100
        state["aoa"] = aoa
        
        cache_key = "aerofoil" + str(len(cache_order) + 1)
        cache[cache_key] = {"camber": camber, "thickness": thickness, "aoa": aoa}
        cache_order.append(cache_key)
        
        if len(cache) > cache_limit:
            fifo_key = cache_order.pop(0)
            cache.pop(fifo_key)
            
        update_recent_aerofoils_list()
        update_plot(state, camber_slider, thickness_slider, aoa_slider)
        

    def on_aerofoil_select(event):
        selected_index = recent_aerofoils_list.curselection()
        if selected_index:
            selected_key = recent_aerofoils_list.get(selected_index)
            selected_aerofoil = cache[selected_key]

            camber_slider.set(selected_aerofoil["camber"])
            thickness_slider.set(selected_aerofoil["thickness"])
            aoa_slider.set(selected_aerofoil["aoa"])
            state["camber"] = selected_aerofoil["camber"] / 100
            state["thickness"] = selected_aerofoil["thickness"] / 100
            state["aoa"] = selected_aerofoil["aoa"]

            cache_order.remove(selected_key)
            cache_order.append(selected_key)

            update_plot(state, camber_slider, thickness_slider, aoa_slider)

        recent_aerofoils_list.bind('<<ListboxSelect>>', on_aerofoil_select)

    insert_button = tk.Button(control_panel, text="Insert Aerofoil", command=insert_aerofoil)
    insert_button.pack(pady=10)
        
    # Creating the plotting area on the right side of the main window
    state["fig"], state["ax"] = plt.subplots(figsize=(6, 4))
    state["canvas1"] = FigureCanvasTkAgg(state["fig"], master=root)
    state["canvas1"].get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Creating the plotting area on the right side of the main window
    state["fig2"], state["bar"] = plt.subplots(figsize=(6, 4))
    state["canvas2"] = FigureCanvasTkAgg(state["fig2"], master=root)
    state["canvas2"].get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Drawing the default plot
    update_plot(state, camber_slider, thickness_slider, aoa_slider)
    
    # Allowing the main window to stay open
    root.mainloop()

# Function to update the parameters
def update_plot(state, camber_slider, thickness_slider, aoa_slider):
    state["camber"] = camber_slider.get() / 100
    state["thickness"] = thickness_slider.get() / 100
    state["aoa"] = aoa_slider.get()

    aerofoil_x, aerofoil_y_upper, aerofoil_y_lower = generate_naca_aerofoil(state["chord"], state["camber"], state["thickness"], state["n_points"])

    panels = create_panels(aerofoil_x, aerofoil_y_upper, aerofoil_y_lower)
    lift_coefficient, drag_coefficient = calculate_lift_drag(panels, state["aoa"], state["chord"])

    X, Y, psi = generate_streamlines(aerofoil_x, aerofoil_y_upper, aerofoil_y_lower, state["aoa"], state["grid_size"])
    
    # Clears the previous plot and redraws the canvas
    state["ax"].clear()
    state["ax"].plot(aerofoil_x, aerofoil_y_upper, "b-", label="Upper Surface")
    state["ax"].plot(aerofoil_x, aerofoil_y_lower, "r-", label="Lower Surface")
    state["ax"].contour(X, Y, psi, levels=50, colors='green', alpha=0.7, linewidths=0.5)
    state["ax"].set_title("Aerofoil Simulator")
    state["ax"].axis("equal")
    state["ax"].legend()
    
    state["bar"].clear()
    forces = ["Lift (Cl)", "Drag (Cd)"]
    bars = state["bar"].bar(["Lift (Cl)", "Drag (Cd)"],
                            [lift_coefficient, drag_coefficient],
                            color="violet",
                            width=0.6)
    coefficients = [lift_coefficient, drag_coefficient]
    state["bar"].bar(forces, coefficients, color='violet')
    max_value = max(coefficients)
    state["bar"].set_title("Lift vs Drag")
    
    for bar in bars:
        height = bar.get_height()
        state["bar"].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha = "center",
            va = "bottom"
        )    
    
    state["canvas1"].draw()
    state["canvas2"].draw()
    
if __name__ == "__main__":
    create_gui()
    
#fix

# Preset_aerofoils = {
#     "Plate": {"chord_length": 1.0, "camber": 0.00, "thickness": 0.00},
#     "Ball": {"chord_length": 1.0, "camber": 0.00, "thickness": 0.20},
#     "Symmetric Aerofoil": {"chord_length": 1.0, "camber": 0.00, "thickness": 0.12},
#     # more presets can be added into this dictionary
# }

# def get_preset_data(preset_name, presets):
#     return presets.get(preset_name, None)

# def update_fields(entries, preset_data):
#     if preset_data:
#         return {key: str(value) for key, value in preset_data.items()}
#     return {}

# #testing
# selected_preset = "Ball"
# preset_value = get_preset_data(selected_preset, Preset_aerofoils)
# updated_entries = update_fields(["chord_length", "camber", "thickness"], preset_value)

# print(updated_entries)


# import json
# import os

# SAVE_FILE = "saved_results.json"

# def load_saved_results():
#     if os.path.exists(SAVE_FILE):
#         with open(SAVE_FILE, "r") as file:
#             return json.load(file)
#     return {}

# def save_result(name, values):
#     saved_data = load_saved_results()
#     saved_data[name] = values
#     with open(SAVE_FILE, "w") as file:
#         json.dump(saved_data, file, indent=4)
#     update_saved_results_menu()
    
# def update_saved_results_menu():
#     saved_data = load_saved_results()
#     saved_results_menu["values"] = list(saved_data.keys())
    
# def load_saved_result(event):
#     selected = saved_result_var.get()
#     saved_data = load_saved_results().get(selected, {})
    
#     for key, entry in entry_fields.items():
#         entry.delete(0, tk.END)
#         entry.insert(0, str(saved_data.get(key, "")))

# saved_results_var = tk.StringVar()
# saved_results_menu = ttk.Combobox(textvariable=saved_results_var)
# saved_results_menu.pack()
# saved_results_menu.bind("<<ComboboxSelected>>", load_saved_result)

# entry_fields = {
#     "chord_length": tk.Entry(),
#     "camber": tk.Entry(),
#     "thickness":tk.Entry(),
# }

# def save_current_result():
#     name = tk.simpledialog.askstring("Save Result", "Enter name for saved result:")
#     if name:
#         values = {key: entry.get() for key, entry in entry_fields.items()}
#         save_result(name, values)

# save_button = tk.Button(root, text="Save Result", command=save_current_result)
# save_button.pack()