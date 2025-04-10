import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import imageio.v2 as imageio

class Point():
    def __init__(self, x, y, z, fixed, index = 0, applied_forces = set(), mass = 10):
        self.coords = np.array([x, y, z], dtype = float)

        self.fixed = fixed # True = fixed, False = free

        self.index = index

        self.forces = applied_forces

        self.mass = mass

        self.velocity = np.array([0., 0., 0.])

        self.acceleration = np.array([0., 0., 0.])

        self.connected = set()

    def dir(self, other):
        d = [other.coords[i] - self.coords[i] for i in range(3)]
        u_d = d / np.linalg.norm(d)

        return u_d

class LineElement():
    def __init__(self, start, end, k, sigma_y, area):
        self.E = k
        self.sigma_y = sigma_y
        self.area = area # cross-sectional area in square meters

        self.start = start
        self.end = end

        self.start.connected.add(self.end)
        self.end.connected.add(self.start)

        self.rest_length = self.len()
        self.ORIG_REST = self.len()

        self.y = False # if yielded already

    def dir(self):
        return self.start.dir(self.end)
    
    def len(self):
        delta = [self.end.coords[i] - self.start.coords[i] for i in range(3)]
        return float(np.linalg.norm(delta))
    
    def strain(self):
        return self.len() / self.rest_length - 1
    
    def yielded(self):
        if self.y:
            return True
        else:
            self.y = self.E * self.strain() >= self.sigma_y
            return self.y
    
class Structure():
    def __init__(self, elements, PIDs, sensors):
        self.elements = elements
        self.points = {elt.start for elt in elements} | {elt.end for elt in elements}
        self.sensors = sensors
        self.PIDs = PIDs

        self.C = self.create_connectivity_mat()

        self.t = 0

    def create_connectivity_mat(self):
        max_ind = max(p.index for p in self.points)
        C = np.zeros((max_ind + 1, max_ind + 1))

        for elt in self.elements:
            C[elt.start.index, elt.end.index] = 1
            C[elt.end.index, elt.start.index] = 1

        return C
    
    def timestep(self, dt):
        self.t += dt

        for controller in self.PIDs:
            controller.timestep(dt)

        for point in self.points:
            if not point.fixed:
                point.coords += dt * point.velocity

        # for controller in self.PIDs:
        #     controller.timestep(dt)

        self.applied_forces, self.structural_forces = self.update_velocities(dt)
    
    def update_velocities(self, dt, damping = 0.3):
        applied_forces = {p.index: np.array([0., 0., 0.]) for p in self.points}

        structural_forces = {p.index: np.array([0., 0., 0.]) for p in self.points}

        for elt in self.elements:
            hook_stress = (elt.len() - elt.rest_length) * elt.E * elt.dir()
            
            structural_forces[elt.start.index] += 1 * hook_stress

            structural_forces[elt.end.index] += -1 * hook_stress

        for pt in self.points:
            applied_forces[pt.index] += sum(force(self.t) for force in pt.forces)

        for point in self.points:
            if not point.fixed:
                point.acceleration = (applied_forces[point.index] + structural_forces[point.index]) / point.mass - (damping/point.mass) * point.velocity
                point.velocity += dt * point.acceleration

        for sensor in self.sensors: 
            sensor.observe() # update the acceleration that each sensor reads
        
        return applied_forces, structural_forces
    
    def get_pt_by_ind(self, ind):
        for point in self.points:
            if point.index == ind:
                return point
            
        return None
    
    def plot(self, i = 0, fig = None, ax = None):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        containers = []

        for elt in self.elements:
            a = ax.plot(
                [elt.start.coords[0], elt.end.coords[0]],
                [elt.start.coords[1], elt.end.coords[1]],
                [elt.start.coords[2], elt.end.coords[2]],
                color = "k" if not elt.yielded() else "r"
            )

            b = ax.scatter(
                [elt.start.coords[0], elt.end.coords[0]],
                [elt.start.coords[1], elt.end.coords[1]],
                [elt.start.coords[2], elt.end.coords[2]],
                color = "blue"
            )

            containers.extend([a, b])

        for pt in self.points:
            if not pt.fixed:
                for force in pt.forces:
                    c = ax.quiver(
                        pt.coords[0],
                        pt.coords[1],
                        pt.coords[2],
                        force(self.t)[0],
                        force(self.t)[1],
                        force(self.t)[2],
                        color = "red",
                        length = 0.05
                    )

                    containers.append(c)

        for pt_ind, struct_force in self.structural_forces.items():
            pt = self.get_pt_by_ind(pt_ind)
            if not pt.fixed:

                d = ax.quiver(
                        pt.coords[0],
                        pt.coords[1],
                        pt.coords[2],
                        struct_force[0],
                        struct_force[1],
                        struct_force[2],
                        color = "green",
                        length = 0.05
                    )
                
                containers.append(d)
            
        ax.set_box_aspect(
            (np.ptp([pt.coords[0] for pt in self.points]) if np.ptp([pt.coords[0] for pt in self.points]) > 0 else 1,
            np.ptp([pt.coords[1] for pt in self.points]) if np.ptp([pt.coords[1] for pt in self.points]) > 0 else 1,
            np.ptp([pt.coords[2] for pt in self.points]) if np.ptp([pt.coords[2] for pt in self.points]) > 0 else 1,)
        )

        mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'
            
        ax.set(xticklabels=[],
            yticklabels=[],
            zticklabels=[])

        # plt.show()

        plt.savefig(f"img/{i}.png")

        return containers
    
    def step_and_plot(self, frame, fig, ax, dt):
        self.timestep(dt)
        return self.plot(fig, ax)

class AppliedForce():
    def __init__(self, tstart, tend, magnitude = lambda t: 0, dir = [0, 0, -1]):
        # magnitude is a function of time
        # dir is a unit (or unitized) vector

        self.tstart = tstart
        self.tend = tend
        self.magnitude = magnitude
        self.dir = dir / np.linalg.norm(dir)

    def __call__(self, t):
        if t < self.tstart or t > self.tend:
            return np.array([0., 0., 0.])
        else:
            return self.magnitude(t) * self.dir
        

class Sensor:
    def __init__(self, parent = None, alpha = 0.1, noise_std = 0.05, add_noise = True):
        
        self.parent = parent
        self.alpha = alpha
        self.noise_std = noise_std
        self.add_noise = add_noise

        self.accels_vec = []
        self.smoothed_accels_vec = []

        self.smoothed_accel = 0.0

        # self.smoothed_force = 0.0
        # self.raw_forces = []
        # self.smoothed_forces = []

    def observe(self): # called by the struct update timestep function
        
        accel = self.parent.acceleration

        if self.add_noise:
            accel += np.random.normal(0, self.noise_std)

        self.accels_vec.append(accel)

        # EMA filter
        if len(self.smoothed_accels_vec) == 0:
            smoothed = accel
        else:
            smoothed = self.alpha * accel + (1 - self.alpha) * self.smoothed_accel

        self.smoothed_accel = smoothed
        self.smoothed_accels_vec.append(smoothed)



class PID():
    def __init__(self, Kp, Ki, Kd, elt, error_func = None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.elt = elt

        self.error_func = error_func if error_func is not None else lambda: self.elt.ORIG_REST - self.elt.len()
        self.last_e = self.error_func()
        self.e_deriv = 0
        self.e_int = 0

        self.evec = []

    def timestep(self, dt):
        self.apply_response()

        error = self.error_func()

        self.evec.append(error)

        self.e_deriv = (error - self.last_e)/dt
        # print(f"({error:.5f}, {self.e_deriv:.5f}, {self.e_int:.5f})")

        self.e_int += error * dt

        self.last_e = error

    def response(self):
        return (self.Kp * self.error_func() + self.Ki * self.e_int + self.Kd * self.e_deriv)
    
    def apply_response(self):
        n = self.elt.dir()
        u = self.response()

        # self.elt.start.coords += n * u/2

        # self.elt.end.coords -= n * u/2

        self.elt.rest_length = self.elt.rest_length - u

gravity = AppliedForce(0, 100, lambda t: 9.8)

wind1 = AppliedForce(0, 100, lambda t: 2 * np.sin(2 * 2 * np.pi * t), [1, 0, 0])

wind2 = AppliedForce(0, 100, lambda t: 4 * np.cos(4 * 2 * np.pi * t), [0, 1, 0])

forces = {gravity, wind1, wind2}

SC = 3.5 / 10e3 # * 10e6 # concrete yield strength
SS = 100 / 10e3 # * 10e6 # steel yield strength

CE = 40 # * 10e9
SE = 200 # * 10e9

points = {
    0: Point(-6, -1, 0, True, 0, forces),
    1: Point(-2, -1, 0, False, 1, forces),
    2: Point(2, -1, 0, False, 2, forces),
    3: Point(6, -1, 0, True, 3, forces),
    4: Point(-6, 1, 0, True, 4, forces),
    5: Point(-2, 1, 0, False, 5, forces),
    6: Point(2, 1, 0, False, 6, forces),
    7: Point(6, 1, 0, True, 7, forces),

    8: Point(0, -1, -8, True, 8, forces),
    9: Point(0, 1, -8, True, 9, forces),
    10: Point(0, -1, 4, False, 10, forces),
    11: Point(0, 1, 4, False, 11, forces),
}

elts = [
    LineElement(points[0], points[1], CE, SC, 1), # concrete
    LineElement(points[1], points[2], CE, SC, 1),
    LineElement(points[2], points[3], CE, SC, 1),
    LineElement(points[4], points[5], CE, SC, 1),
    LineElement(points[5], points[6], CE, SC, 1),
    LineElement(points[6], points[7], CE, SC, 1),
    LineElement(points[0], points[4], CE, SC, 1),
    LineElement(points[1], points[5], CE, SC, 1),
    LineElement(points[2], points[6], CE, SC, 1),
    LineElement(points[3], points[7], CE, SC, 1),

    LineElement(points[8], points[10], 1000, SS, 0.25), # pillars
    LineElement(points[9], points[11], 1000, SS, 0.25),

    LineElement(points[10], points[1], SE, SS, 0.01), # steel
    LineElement(points[10], points[2], SE, SS, 0.01),
    LineElement(points[11], points[5], SE, SS, 0.01),
    LineElement(points[11], points[6], SE, SS, 0.01),
]

# Kp = 0.015
# Ki = 0.005
# Kd = 0.025

# PIDs = {
#     PID(Kp, Ki, Kd, elts[12]),
#     PID(Kp, Ki, Kd, elts[13]),
#     PID(Kp, Ki, Kd, elts[14]),
#     PID(Kp, Ki, Kd, elts[15]),
# }

# PIDs = {
#                 PID(Kp, Ki, Kd, elts[12], lambda: (points[1].velocity - [-2, -1, 0]) @ elts[12].dir()),
#                 PID(Kp, Ki, Kd, elts[13], lambda: (points[2].velocity - [2, -1, 0]) @ elts[13].dir()),
#                 PID(Kp, Ki, Kd, elts[14], lambda: (points[5].velocity - [-2, 1, 0]) @ elts[14].dir()),
#                 PID(Kp, Ki, Kd, elts[15], lambda: (points[6].velocity - [2, 1, 0]) @ elts[15].dir()),
#             }

Kp = 0
Ki = 0
Kd = 0

sensors = {
    1: Sensor(parent = points[1], alpha = 0.5, noise_std = 0.1, add_noise = True),
    2: Sensor(parent = points[2], alpha = 0.5, noise_std = 0.1, add_noise = True),
    5: Sensor(parent = points[5], alpha = 0.5, noise_std = 0.1, add_noise = True),
    6: Sensor(parent = points[6], alpha = 0.5, noise_std = 0.1, add_noise = True)
}

# PIDs = {
#                 PID(Kp, Ki, Kd, elts[12], lambda: (points[1].velocity) @ elts[12].dir()),
#                 PID(Kp, Ki, Kd, elts[13], lambda: (points[2].velocity) @ elts[13].dir()),
#                 PID(Kp, Ki, Kd, elts[14], lambda: (points[5].velocity) @ elts[14].dir()),
#                 PID(Kp, Ki, Kd, elts[15], lambda: (points[6].velocity) @ elts[15].dir()),
#             }

PIDs = {
                PID(Kp, Ki, Kd, elts[12], lambda: sensors[1].smoothed_accel * points[1].dir @ elts[12].dir()),
                PID(Kp, Ki, Kd, elts[13], lambda: sensors[2].smoothed_accel * points[2].dir @ elts[13].dir()),
                PID(Kp, Ki, Kd, elts[14], lambda: sensors[5].smoothed_accel * points[5].dir @ elts[14].dir()),
                PID(Kp, Ki, Kd, elts[15], lambda: sensors[6].smoothed_accel * points[6].dir @ elts[15].dir()),
            }

struct = Structure(elts, PIDs, sensors)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# ani = animation.FuncAnimation(fig=fig, func=struct.step_and_plot, frames=40, fargs = (fig, ax, 0.1,), interval=30)
# plt.show()

vid_len = 600
dt = 0.05

best_PID = None
best_squerror = None

for P in np.arange(0, 0.055, 0.005):
    for I in np.arange(0, 0.005, 0.001):
        for D in np.arange(0, 0.055, 0.005):
            points = {
                0: Point(-6, -1, 0, True, 0, forces),
                1: Point(-2, -1, 0, False, 1, forces), #
                2: Point(2, -1, 0, False, 2, forces), #
                3: Point(6, -1, 0, True, 3, forces),
                4: Point(-6, 1, 0, True, 4, forces),
                5: Point(-2, 1, 0, False, 5, forces), #
                6: Point(2, 1, 0, False, 6, forces), #
                7: Point(6, 1, 0, True, 7, forces),

                8: Point(0, -1, -8, True, 8, forces),
                9: Point(0, 1, -8, True, 9, forces),
                10: Point(0, -1, 4, False, 10, forces),
                11: Point(0, 1, 4, False, 11, forces),
            }

            elts = [
                LineElement(points[0], points[1], CE, SC, 1), # concrete
                LineElement(points[1], points[2], CE, SC, 1),
                LineElement(points[2], points[3], CE, SC, 1),
                LineElement(points[4], points[5], CE, SC, 1),
                LineElement(points[5], points[6], CE, SC, 1),
                LineElement(points[6], points[7], CE, SC, 1),
                LineElement(points[0], points[4], CE, SC, 1),
                LineElement(points[1], points[5], CE, SC, 1),
                LineElement(points[2], points[6], CE, SC, 1),
                LineElement(points[3], points[7], CE, SC, 1),

                LineElement(points[8], points[10], 1000, SS, 0.25), # pillars
                LineElement(points[9], points[11], 1000, SS, 0.25),

                LineElement(points[10], points[1], SE, SS, 0.01), # steel
                LineElement(points[10], points[2], SE, SS, 0.01),
                LineElement(points[11], points[5], SE, SS, 0.01),
                LineElement(points[11], points[6], SE, SS, 0.01),
            ]

            sensors = {
                1: Sensor(parent = points[1], alpha = 0.5, noise_std = 0.1, add_noise = True),
                2: Sensor(parent = points[2], alpha = 0.5, noise_std = 0.1, add_noise = True),
                5: Sensor(parent = points[5], alpha = 0.5, noise_std = 0.1, add_noise = True),
                6: Sensor(parent = points[6], alpha = 0.5, noise_std = 0.1, add_noise = True)
            }

            PIDs = {
                PID(Kp, Ki, Kd, elts[12], lambda: sensors[1].smoothed_accel * points[1].dir @ elts[12].dir()),
                PID(Kp, Ki, Kd, elts[13], lambda: sensors[2].smoothed_accel * points[2].dir @ elts[13].dir()),
                PID(Kp, Ki, Kd, elts[14], lambda: sensors[5].smoothed_accel * points[5].dir @ elts[14].dir()),
                PID(Kp, Ki, Kd, elts[15], lambda: sensors[6].smoothed_accel * points[6].dir @ elts[15].dir()),
            }

            struct = Structure(elts, PIDs, sensors)

            for i in range(vid_len):
                struct.timestep(dt)

            sqerror = sum(sum(dt * e ** 2 for e in controller.evec) for controller in PIDs) * 10

            if not best_PID or sqerror < best_squerror:
                best_PID = (P, I, D)
                best_squerror = sqerror

print(best_PID)
print(best_squerror)


with imageio.get_writer(f'{(Kp, Ki, Kd)}.gif', mode='I', fps = math.floor(1/dt)) as writer:
    for i in range(vid_len):
        struct.timestep(dt)
        struct.plot(i = i)

        image = imageio.imread(f"img/{i}.png")
        writer.append_data(image)

        plt.close()

sqerror = sum(sum(dt * e ** 2 for e in controller.evec) for controller in PIDs) * 10

fig, ax = plt.subplots()

ax.set_xlabel("timestep")
ax.set_ylabel("error")
ax.set_title(f"Error versus Timestep, {(Kp, Ki, Kd)}, s. sq. er. = {sqerror:0.3f}")

for controller in PIDs:
    ax.plot(
        controller.evec
    )
plt.savefig(f"{(Kp, Ki, Kd)}.png", dpi=300)
plt.savefig(f"{(Kp, Ki, Kd)}.pdf")
