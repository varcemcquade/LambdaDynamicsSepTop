import numpy as np
import matplotlib.pyplot as plt

def calculate_fpls(data_list: list[list[list[int]]]) -> list[list[float]]:
    cutoff = 0.95
    frames = len(data_list[0])
    fpl_list = []

    for data in data_list:
        physical_frames = [0, 0, 0]
        for lig_idx in range(3):
            lambda_column = data[:, lig_idx]
            physical_frames[lig_idx] = np.sum(lambda_column > cutoff)
            print(f"Max lambda of lig_idx {lig_idx} is {max(lambda_column)}")
        fpl = [round(float(x / frames), 3) for x in physical_frames]
        print("-------------")
        fpl_list.append(fpl)

    return fpl_list

def plot_lambdas(filename: str):
    lambdas = np.loadtxt(filename)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lambdas[:, 0], label='λ₀ (Lig 0)', alpha=0.8)
    ax.plot(lambdas[:, 1], label='λ₁ (Lig 1)', alpha=0.8)
    ax.plot(lambdas[:, 2], label='λ₂ (Lig 2)', alpha=0.8)

    ax.axhline(y=0.99, color='r', linestyle='--', label='Old cutoff (0.99)')
    ax.axhline(y=0.95, color='g', linestyle='--', label='New cutoff (0.95)')

    ax.set_xlabel("Frame")
    ax.set_ylabel("λ value")
    ax.set_title("Lambda Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
    #plt.savefig('lambda_trajectory_a0.01_rep4.png', dpi=150)


"""
for a in ["0.005", "0.007", "0.012", "0.017"]:
    folder_name = f"TYK2_complex_run500_Lambdas_{a}"
    data_list = [np.loadtxt(f"{folder_name}/Lambda1.dat"),
                 np.loadtxt(f"{folder_name}/Lambda2.dat"),
                 np.loadtxt(f"{folder_name}/Lambda3.dat"),
                 np.loadtxt(f"{folder_name}/Lambda4.dat"),
                 np.loadtxt(f"{folder_name}/Lambda5.dat")]
    print(f"alpha value: {a}")
    print(calculate_fpls(data_list))
"""



plot_lambdas("TYK2_complex_run500_Lambdas_0.017/Lambda3.dat")