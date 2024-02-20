import matplotlib.pyplot as plt
import numpy as np

FPI_diameter = 3e-3   # [m]
conversion_constant = 0.058   # [T/A]


class Measurement:
    def __init__(self, d, c, meas_with, meas_without):
        self.d = d
        self.c = c
        self.D_a = meas_with[1]*2e-6
        self.D_b = meas_with[0]*2e-6
        self.D_1 = meas_without[1]*2e-6
        self.D_2 = meas_without[0]*2e-6
        self.current = meas_with[2]

    def splitting_size(self):
        return (1/(2*self.d) * (self.D_a**2 - self.D_b**2)/(self.D_2**2 - self.D_1**2))/100  # [cm^-1]

    def magnetic_field(self):
        return self.current*self.c  # [T]


def plot(magnetic_field, splitting_size, color):
    plt.scatter(magnetic_field, splitting_size, color=color)
    splitting_size_fit = np.polyfit(magnetic_field, splitting_size, 1)
    splitting_size_fit = np.poly1d(splitting_size_fit)
    plt.plot(magnetic_field, splitting_size_fit(magnetic_field), '--', label=f'{color} linear fit: slope={splitting_size_fit[0] * 10:.2f}', color=color)

#[R_1, R_2, I]
blue_meas_without = [1385.425, 1951.263, 0]
blue_meas_with = [
                    [1434.173, 1317.834, 1],
                    [1444.891, 1278.880, 1.5],
                    [1499.952, 1258.944, 2],
                    [1531.370, 1225.272, 2.5],
                    [1545.225, 1204.965, 3],
                    [1563.868, 1178.646, 3.5],
                    [1592.980, 1139.581, 4]
                ]

red_meas_without = [17493.047, 23973.471, 0]
red_meas_with = [
                    [18009.921, 17294.295, 1],
                    [18129.192, 17095, 1.5],
                    [18447.248, 16976.239, 2],
                    [18606.276, 16777.643, 2.5],
                    [18725.547, 16697.941, 3],
                    [19083.362, 16658.183, 3.5],
                    [19162.874, 16499.155, 4]


blue = [Measurement(FPI_diameter, conversion_constant, i, blue_meas_without) for i in blue_meas_with]
blue_splitting_size = blue.splitting_size()
blue_magnetic_field = blue.magnetic_field()
plot(blue_magnetic_field, blue_splitting_size, 'Blue')

red = [Measurement(FPI_diameter, conversion_constant, i, red_meas_without) for i in red_meas_with]
red_splitting_size = red.splitting_size()
red_magnetic_field = red.magnetic_field()
plot(red_magnetic_field, red_splitting_size, 'red')

plt.xlabel(r'$Magnetic \ field \ [T]$', fontsize=14)
plt.ylabel(r'$Splitting \ size \ [cm^{-1}]$', fontsize=14)
plt.title('Magnetic field vs Splitting size')
plt.legend()
plt.savefig('Plot.png')
plt.show()
