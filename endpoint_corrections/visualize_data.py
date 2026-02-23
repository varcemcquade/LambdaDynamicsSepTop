import numpy as np
import matplotlib.pyplot as plt

def CalculateFPLs(data_list: list[list[list[int]]]) -> list[list[float]]:
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

def PlotLambdaTrajectory(filename: str):
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

def PlotLambdasTwoDimensional(filename: str):
    lambdas = np.loadtxt(filename)
    fig, ax = plt.subplots()
    ax.plot(lambdas[:, 0], lambdas[:, 1])
    #ax.plot(lambdas[:, 1], lambdas[:, 2])
    #ax.plot(lambdas[:, 2], lambdas[:, 0])
    plt.show()


for folder in ["strongK_Lambdas210", "strongK_intralig_Lambdas210"]:
    data_list = [np.loadtxt(f"{folder}/Lambda1.dat"),
                 np.loadtxt(f"{folder}/Lambda2.dat"),
                 np.loadtxt(f"{folder}/Lambda3.dat"),
                 np.loadtxt(f"{folder}/Lambda4.dat"),
                 np.loadtxt(f"{folder}/Lambda5.dat")]
    print(f"test: {folder}")
    print(CalculateFPLs(data_list))




"""
data_list = [np.loadtxt("Lambda1.dat"),
             np.loadtxt("Lambda2.dat"),
             np.loadtxt("Lambda4.dat"),
             np.loadtxt("Lambda4.dat"),
             np.loadtxt("Lambda5.dat")]

print(CalculateFPLs(data_list))
#PlotLambdaTrajectory("Lambda1.dat")
#PlotLambdasTwoDimensional("Lambda4.dat")
"""