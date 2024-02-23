import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(['science', 'grid', 'notebook'])


def magnetic_field(conversion_factor: float, current: float) -> float:
    return conversion_factor * current  # [T]


def splitting_size_exp(d: float, meas: list, meas_without: list) -> float:
    D_a: float = meas[0]  # [m]
    D_b: float = meas[1]  # [m]
    D_1: float = meas_without[0]  # [m]
    D_2: float = meas_without[1]  # [m]
    return (1 / (2 * d) * (D_a ** 2 - D_b ** 2) / (D_2 ** 2 - D_1 ** 2)) / 100  # [cm^-1]


def plot_exp(magnetic_fields: list, splitting_size: list, color: str) -> None:
    global blue_slope_exp, red_slope_exp
    plt.scatter(magnetic_fields, splitting_size, color=color)
    coefficients = np.polyfit(magnetic_fields, splitting_size, 1)
    slope = coefficients[0]
    intersect = coefficients[1]
    plt.plot(magnetic_fields, slope * np.array(magnetic_fields) + intersect, label=f'{color} experimental: slope={slope:.2f}',color=color)
    if color == 'Red':
        red_slope_exp = slope
    elif color == 'Blue':
        blue_slope_exp = slope
        ratio = red_slope_exp / blue_slope_exp
        plt.text(0.19, 0.68, f'Experimental ratio of slopes: {ratio:.4}', ha='center', va='center',
                         transform=plt.gca().transAxes, fontsize=12)


def plot_theory(magnetic_fields: np.ndarray, slope: float, color: str, sigma: str, marker: str) -> None:
    global blue_slope_theory, red_slope_theory
    plt.plot(magnetic_fields, slope * np.array(magnetic_fields), marker, label=rf'${color} \ theoretical \ \sigma^{sigma}$, : slope={slope:.2f}',color=color)
    if color == 'Red':
        red_slope_theory = slope
    elif color == 'Blue':
        blue_slope_theory = slope
        if red_slope_theory == 0:
            ratio = 0
        else:
            ratio = red_slope_theory / blue_slope_theory
        plt.text(0.16, 0.63, f'Theoretical ratio of slopes: {ratio:.4}', ha='center', va='center',
                 transform=plt.gca().transAxes, fontsize=12)

def process_and_plot_exp(meas_data: list, currents, color: str, FPI_diameter: float, conversion_factor: float) -> None:
    magnetic_fields = [magnetic_field(conversion_factor, current) for current in currents[1:]]
    splitting_sizes = [splitting_size_exp(FPI_diameter, meas, meas_data[0]) for meas in meas_data[1:]]
    plot_exp(magnetic_fields, splitting_sizes, color)


def process_and_plot_theory(magnetic_fields: np.ndarray, g_J: int, delta_M_J: int, sigma: str,
                            marker: str, color: str, mu_B: float, h: float, c: float) -> None:
    slope: float = ((g_J * mu_B * delta_M_J) / (h * c)) / 100  # [cm^-1]
    plot_theory(magnetic_fields, slope, color, sigma, marker)


def main() -> None:
    mu_B: float = 9.2741e-24  # [J T^-1]
    h: float = 6.62618e-34  # [Js]
    c: float = 299792458  # [m/s]
    conversion_factor: float = 0.1  # [T/A]
    FPI_diameter: float = 0.003  # [m]

    currents: list = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4]  # [A]
    magnetic_fields: np.ndarray = np.linspace(0.1, 0.4, 100)  # [T]
    delta_M_J_values: list = [1, -1]

    sigma_sign: list = ['+', '-']
    markers: list = ['--', '-.']

    red_slope_exp: int = 0
    red_slope_theory: int = 0
    red_g_J: int = 1
    red_meas_data: list = [
        [731.864, 1749.304, 0],
        [827.952, 671.872, 1],
        [857.266, 619.443, 1.5],
        [883.587, 604.073, 2],
        [926.431, 558.214, 2.5],
        [946.878, 527.846, 3],
        [989.733, 480.210, 3.5],
        [1005.797, 429.807, 4]
    ]

    blue_slope_exp: int = 0
    blue_slope_theory: int = 0
    blue_g_J: int = 2
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


    plt.figure(figsize=(12,8))
    process_and_plot_exp(red_meas_data, currents, 'Red', FPI_diameter, conversion_factor)
    process_and_plot_exp(blue_meas_data, currents,'Blue', FPI_diameter, conversion_factor)
    for delta_M_J, sigma, marker in zip(delta_M_J_values, sigma_sign, markers):
        process_and_plot_theory(magnetic_fields, red_g_J, delta_M_J, sigma, marker, 'Red', mu_B, h, c)
        process_and_plot_theory(magnetic_fields, blue_g_J, delta_M_J, sigma, marker, 'Blue', mu_B, h, c)

    plt.xlabel(r'$B \ [T]$', fontsize=14)
    plt.ylabel(r'$\Delta \nu \ [\text{cm}^{-1}]$', fontsize=14)
    plt.title('Magnetic field vs Splitting size')

    plt.legend(fontsize=12)
    plt.savefig('Plot1.png')
    plt.show()

    magnetic_fields: list = [magnetic_field(current, conversion_factor) for current in currents]
    plt.plot(currents, magnetic_fields)
    plt.xlabel(r'$I \ [A]$', fontsize=14)
    plt.ylabel(r'$B \ [T]$', fontsize=14)
    plt.title('Magnetic current vs Magnetic field')
    plt.savefig('Plot2.png')
    plt.show()

if __name__ == '__main__':
    main()
