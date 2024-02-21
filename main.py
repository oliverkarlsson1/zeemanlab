import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(['science', 'grid', 'notebook'])

def magnetic_field(conversion_constant: float, current: float) -> float:
    return conversion_constant * current  # [T]


def splitting_size_exp(d: float, meas: list, meas_without: list) -> float:
    D_a: float = meas[0] * 2e-6  # [m]
    D_b: float = meas[1] * 2e-6  # [m]
    D_1: float = meas_without[0] * 2e-6  # [m]
    D_2: float = meas_without[1] * 2e-6  # [m]
    exp = (1 / (2 * d) * (D_a ** 2 - D_b ** 2) / (D_2 ** 2 - D_1 ** 2)) / 1000
    return exp  # [cm^-1]


def plot_exp(magnetic_field: list, splitting_size: list, color: str) -> None:
    global blue_slope_exp, red_slope_exp
    plt.scatter(magnetic_field, splitting_size, color=color)
    coefficients = np.polyfit(magnetic_field, splitting_size, 1)
    slope = coefficients[0]
    intersect = coefficients[1]
    plt.plot(magnetic_field, slope * np.array(magnetic_field) + intersect, '--', label=f'{color} experimental: slope={slope:.2f}',color=color)
    if color == 'Red':
        red_slope_exp = slope
    elif color == 'Blue':
        blue_slope_exp = slope
        ratio = red_slope_exp / blue_slope_exp
        plt.text(0.2, 0.5, f'Experimental ratio of slopes: {ratio:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=17)


def plot_theory(magnetic_fields, slope, color) -> None:
    global blue_slope_theory, red_slope_theory
    plt.plot(magnetic_fields, slope * np.array(magnetic_fields), label=f'{color} theoretical: slope={slope:.2f}', color=color)
    if color == 'Red':
        red_slope_theory = slope
    elif color == 'Blue':
        blue_slope_exp = slope
        ratio = red_slope_theory / blue_slope_exp
        plt.text(0.2, 0.6, f'Theoretical ratio of slopes: {ratio:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=17)


def process_and_plot_exp(meas_data: list, reference_meas: list, color: str, FPI_diameter: float, conversion_constant: float) -> None:
    magnetic_fields: list[float] = [magnetic_field(conversion_constant, meas[2]) for meas in meas_data]
    splitting_sizes: list[float] = [splitting_size_exp(FPI_diameter, meas, reference_meas) for meas in meas_data]
    plot_exp(magnetic_fields, splitting_sizes, color)


def process_and_plot_theory(meas_data, g_J, delta_M_J, color, conversion_constant) -> None:
    mu_B = 3.2741e-24  # [J T^-1]
    h = 6.62618e-34  # [Js]
    c = 299792458  # [m/s]
    magnetic_fields = [magnetic_field(conversion_constant, meas[2]) for meas in meas_data]
    slope = (g_J * mu_B * delta_M_J) / (100 * h * c)
    plot_theory(magnetic_fields, slope, color)


def main() -> None:
    FPI_diameter: float = 3e-3  # [m]
    conversion_constant: float = 1e-1  # [T/A]

    red_slope_exp = 0
    red_slope_theory = 0
    red_g_J = 1
    red_delta_M_J = 1
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

    blue_slope_exp = 0
    blue_slope_theory = 0
    blue_g_J = 2
    blue_delta_M_J = 1
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

    process_and_plot_exp(red_meas_data[1:], red_meas_data[0], 'Red', FPI_diameter, conversion_constant)
    process_and_plot_exp(blue_meas_data[1:], blue_meas_data[0], 'Blue', FPI_diameter, conversion_constant)
    process_and_plot_theory(red_meas_data[1:], red_g_J, red_delta_M_J, 'Red', conversion_constant)
    process_and_plot_theory(blue_meas_data[1:], blue_g_J, blue_delta_M_J, 'Blue', conversion_constant)

    plt.xlabel(r'$B \ [T]$', fontsize=14)
    plt.ylabel(r'$\Delta \nu \ [\text{cm}^{-1}]$', fontsize=14)
    plt.title('Magnetic field vs Splitting size')
    plt.legend()
    plt.savefig('Plot.png')
    plt.show()


if __name__ == '__main__':
    main()
