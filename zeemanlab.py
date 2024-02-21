import matplotlib.pyplot as plt
import numpy as np

FPI_diameter: float = 3e-3   # [m]
conversion_constant: float = 58e-4   # [T/A]


def magnetic_field(conversion_constant, current):
    return conversion_constant * current  # [T]


def splitting_size(d, meas, meas_without):
    D_a: float = meas[0] * 2e-6  # [m]
    D_b: float = meas[1] * 2e-6  # [m]
    D_1: float = meas_without[0] * 2e-6  # [m]
    D_2: float = meas_without[1] * 2e-6  # [m]
    return (1 / (2 * d) * (D_a ** 2 - D_b ** 2) / (D_2 ** 2 - D_1 ** 2)) / 100  # [cm^-1]


def plot(magnetic_field, splitting_size, color):
    plt.scatter(magnetic_field, splitting_size, color=color)
    splitting_size_fit: np.poly1d = np.poly1d(np.polyfit(magnetic_field, splitting_size, 1))
    plt.plot(magnetic_field, splitting_size_fit(magnetic_field), '--', label=f'{color} linear fit: slope={splitting_size_fit[0] * 10:.2f}', color=color)


def process_and_plot(meas_data, reference_meas, color):
    splitting_sizes = [splitting_size(FPI_diameter, meas, reference_meas) for meas in meas_data]
    magnetic_fields = [magnetic_field(conversion_constant, meas[2]) for meas in meas_data]
    plot(magnetic_fields, splitting_sizes, color)


blue_meas_data: list = [
    [1385.425, 1951.263, 0],
    [1434.173, 1317.834, 1],
    [1444.891, 1278.880, 1.5],
    [1499.952, 1258.944, 2],
    [1531.370, 1225.272, 2.5],
    [1545.225, 1204.965, 3],
    [1563.868, 1178.646, 3.5],
    [1592.980, 1139.581, 4]
]

red_meas_data: list = [
    [17493.047, 23973.471, 0],
    [18009.921, 17294.295, 1],
    [18129.192, 17095, 1.5],
    [18447.248, 16976.239, 2],
    [18606.276, 16777.643, 2.5],
    [18725.547, 16697.941, 3],
    [19083.362, 16658.183, 3.5],
    [19162.874, 16499.155, 4]
]

process_and_plot(blue_meas_data[1:], blue_meas_data[0], 'Blue')
process_and_plot(red_meas_data[1:], red_meas_data[0], 'Red')

plt.xlabel(r'$Magnetic \ field \ [T]$', fontsize=14)
plt.ylabel(r'$Splitting \ size \ [cm^{-1}]$', fontsize=14)
plt.title('Magnetic field vs Splitting size')
plt.legend()
plt.savefig('Plot.png')
plt.show()
