import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import imageio.v2 as imageio

import csv
from itertools import zip_longest

run_parameter_search = False
make_animation = True


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

class AppliedDisplacement():
    def __init__(self, tstart, tend, magnitude = lambda t: 0, dir = [0, 1, 0]):
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

class Point():
    def __init__(self, x, y, z, fixed, index = 0, applied_forces = set(), applied_displacements = None, mass = 0.1):
        self.initial_position = np.array([x, y, z], dtype = float)

        self.coords = np.array([x, y, z], dtype = float)
        self.fixed = fixed # True = fixed, False = free
        self.index = index
        self.forces = applied_forces
        self.displacements = applied_displacements
        self.mass = mass
        self.velocity = np.array([0., 0., 0.])
        self.acceleration = np.array([0., 0., 0.])
        self.connected = set()
    

    def dir(self, other): # vector pointing from this point to another point
        d = [other.coords[i] - self.coords[i] for i in range(3)]
        u_d = d / np.linalg.norm(d)

        return u_d

    def reset(self):
        self.coords = self.initial_position
        self.velocity = np.array([0., 0., 0.])
        self.acceleration = np.array([0., 0., 0.])

class LineElement():
    def __init__(self, start, end, k, sigma_y, area, unstretched_length = 0.127):
        self.E = k
        self.sigma_y = sigma_y
        self.area = area # cross-sectional area in square meters

        self.start = start
        self.end = end

        self.start.connected.add(self.end)
        self.end.connected.add(self.start)

        self.unstretched_length = unstretched_length

        self.rest_length = unstretched_length
        self.ORIG_REST = unstretched_length

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
        
    def reset(self):
        self.y = False
        self.rest_length = self.ORIG_REST
        return None
        

class Structure():
    def __init__(self, elements, PIDs, sensors):
        self.elements = elements
        self.points = {elt.start for _, elt in elements.items()} | {elt.end for _, elt in elements.items()}
        self.sensors = sensors
        self.PIDs = PIDs

        self.C = self.create_connectivity_mat()

        self.t = 0

    def reset(self):
        for point in self.points:
            point.reset()

        for pid in self.PIDs:
            pid.reset()

        for number, sensor in self.sensors.items():
            sensor.reset()

        self.t = 0

    def create_connectivity_mat(self):
        max_ind = max(p.index for p in self.points)
        C = np.zeros((max_ind + 1, max_ind + 1))

        for _, elt in self.elements.items():
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

        self.update_displacements(dt)
        self.applied_forces, self.structural_forces = self.update_velocities(dt)

        # print(f"on point 6 applied forces are {np.linalg.norm(self.applied_forces[6])}, structural_forces are {np.linalg.norm(self.structural_forces[6])}")

    def update_displacements(self, dt):
        for p in self.points:
            if p.displacements is not None:
                for displacement in p.displacements:
                    p.coords = p.initial_position + displacement.magnitude(self.t)*dt
        return None

    def update_velocities(self, dt, damping=0.1):
        applied_forces = {p.index: np.array([0., 0., 0.]) for p in self.points}

        structural_forces = {p.index: np.array([0., 0., 0.]) for p in self.points}

        for _, elt in self.elements.items():
            hook_stress = (elt.len() - elt.rest_length) * elt.E * elt.dir()

            structural_forces[elt.start.index] += 1 * hook_stress

            structural_forces[elt.end.index] += -1 * hook_stress
            # if i == 15:
            #     print(f"hook stress is {np.linalg.norm(hook_stress)} and the other ones are {structural_forces[elt.start.index]} and {structural_forces[elt.end.index]}")




        for pt in self.points:
            applied_forces[pt.index] += sum(force(self.t) for force in pt.forces)

        for point in self.points:
            if not point.fixed:
                point.acceleration = (applied_forces[point.index] + structural_forces[point.index]) / point.mass - (damping/point.mass) * point.velocity + np.array([0, 0, -9.8])
                # point.acceleration = (applied_forces[point.index] + structural_forces[point.index]) / point.mass 
                point.velocity += dt * point.acceleration
                # point.velocity += dt * np.array([0, 0, -9.8])

        for number, sensor in self.sensors.items():
            sensor.observe(dt) # update the acceleration that each sensor reads

        return applied_forces, structural_forces

    def get_pt_by_ind(self, ind):
        for point in self.points:
            if point.index == ind:
                return point

        return None

    def plot(self, i = 0, fig = None, ax = None):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        containers = []

        for _, elt in self.elements.items():
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
                        length = 0.01
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
                        length = 0.01
                    )

                containers.append(d)

        ax.set_box_aspect(
            (np.ptp([pt.coords[0] for pt in self.points]) if np.ptp([pt.coords[0] for pt in self.points]) > 0 else 1,
            np.ptp([pt.coords[1] for pt in self.points]) if np.ptp([pt.coords[1] for pt in self.points]) > 0 else 1,
            np.ptp([pt.coords[2] for pt in self.points]) if np.ptp([pt.coords[2] for pt in self.points]) > 0 else 1,)
        )

        # mpl.rcParams['axes3d.mouserotationstyle'] = 'azel' # this is how you interact wiht the plot -- add back in when we want to see

        ax.set(xticklabels=[],
            yticklabels=[],
            zticklabels=[])

        # plt.show()

        plt.savefig(f"img/{i}.png")

        return containers

    def step_and_plot(self, frame, fig, ax, dt):
        self.timestep(dt)
        return self.plot(fig, ax)

class Sensor():
    def __init__(self, parent = None, alpha = 0.1, noise_std = 0.05, add_noise = True):

        self.parent = parent
        self.alpha = alpha
        self.noise_std = noise_std
        self.add_noise = add_noise

        self.accels_vec = []
        self.smoothed_accels_vec = []

        self.velocities_vec = []

        self.smoothed_accel = 0.0
        self.velocity = 0.0

        # self.smoothed_force = 0.0
        # self.raw_forces = []
        # self.smoothed_forces = []

    def observe(self, dt): # called by the struct update timestep function

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

        self.velocity += self.smoothed_accel * dt
        self.velocities_vec.append(self.smoothed_accel * dt)

    def reset(self):
        self.accels_vec = []
        self.smoothed_accels_vec = []

        self.velocities_vec = []

        self.smoothed_accel = 0.0
        self.velocity = 0.0

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
        # print(f"response is {self.Kp * self.error_func() + self.Ki * self.e_int + self.Kd * self.e_deriv} with erf value {self.error_func()}")
        if np.abs(self.error_func()) < 0.03:
            return 0
        else:
            return (self.Kp * self.error_func() + self.Ki * self.e_int + self.Kd * self.e_deriv)

    def apply_response(self):
        n = self.elt.dir()
        u = self.response()

        # self.elt.start.coords += n * u/2

        # self.elt.end.coords -= n * u/2

        # SAFE GUARD
        if u > 0:
            u = min(u, 0.06545/in_to_m)
        else:
            u = max(u, -0.06545/in_to_m)
        # if self.elt.rest_length >= 0.9*(self.elt.ORIG_REST):
        self.elt.rest_length = self.elt.rest_length - u
            # print(f"new length is {self.elt.len()} with new rest length {self.elt.rest_length}")
        # else:
        #     self.elt.rest_length = self.elt.rest_length

    def reset(self):
        self.last_e = self.error_func()
        self.e_deriv = 0
        self.e_int = 0
        self.evec = []

shake = AppliedDisplacement(1, 2, lambda t: 4 * np.sin(2 * np.pi * t))

gravity = AppliedForce(0, 10000, lambda t: 9.8)
wind1 = AppliedForce(0, 100, lambda t: 2 * np.sin(2 * 2 * np.pi * t), [1, 0, 0])
wind2 = AppliedForce(0, 100, lambda t: 4 * np.cos(4 * 2 * np.pi * t), [0, 1, 0])

#forces = {gravity, wind1, wind2}
# forces = {gravity}
forces = set()
displacements = None
# displacements = {shake}

# delay = 10 # num timesteps to look back


#something wrong in error function
def error_function(point, sensor, elt, delay = 0):
    # if np.linalg.norm(point.velocity) > 0:
    # return lambda: np.linalg.norm(sensor.smoothed_accel) * point.velocity/np.linalg.norm(point.velocity) @ elt.dir() if np.linalg.norm(point.velocity) > 0 else 0
    # return lambda: np.linalg.norm(point.velocity) + np.linalg.norm(point.coords-point.initial_position)
    # return lambda: np.linalg.norm(point.coords-point.initial_position)
    # return lambda: np.linalg.norm(sensor.velocities_vec[-delay]) if len(sensor.velocities_vec) >= delay + 1 else np.linalg.norm([0,0,0])        
    return lambda: -np.linalg.norm(sensor.smoothed_accels_vec[-delay][2])*np.sign(sensor.smoothed_accels_vec[-delay][2])/9.81 if len(sensor.smoothed_accels_vec)>delay else 0

SC = 3.5 / 10e3 # * 10e6 # concrete yield strength
SS = 100 / 10e3 # * 10e6 # steel yield strength

CE = 40 # * 10e9
SE = 200 # * 10e9


# CONSTANTS TO MATCH PHYSICAL MODEL

# 1 lbf = 4.45 N
# 39.37 inch = 1 meter
# k_deck = 0.4 lb in^-1
# k_cable = 6 * k_deck

in_to_m = 39.37
lbf_to_N = 4.45

k_deck = 0.4*lbf_to_N*in_to_m
k_cable =  k_deck 
k_tower = 1000 * k_cable

# uncomment below to make interactive view
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ani = animation.FuncAnimation(fig=fig, func=struct.step_and_plot, frames=40, fargs = (fig, ax, 0.1,), interval=30)
# plt.show()

vid_len = 700
dt = 0.05

# create the elements of the structure that are consistent through the whole model

delay = 10

tower_top = 19.625/in_to_m
tower_bottom = 0/in_to_m

front_plane = 3.5/in_to_m
back_plane = -3.5/in_to_m

# front_plane_mid = 3/in_to_m
# back_plane_mid = -3/in_to_m
front_plane_mid = 2.4375/in_to_m
back_plane_mid = -2.4375/in_to_m

front_plane_far = 4.25/in_to_m
back_plane_far = -4.25/in_to_m

mid_deck = 3.6875/in_to_m
end_deck = 12/in_to_m

end_deck_height = 5.25/in_to_m
# mid_deck_height = 8/in_to_m
mid_deck_height = 4.375/in_to_m

positive_tower = 2.125/in_to_m
negative_tower = -2.125/in_to_m

point_masses = 0.01

points = {
    #"left" side
    0: Point(-end_deck, back_plane, end_deck_height, True, 0, forces, displacements, mass = point_masses),
    1: Point(-mid_deck, back_plane_mid, mid_deck_height, False, 1, forces, mass = 0.918),
    2: Point(mid_deck, back_plane_mid, mid_deck_height, False, 2, forces, mass = 0.918),
    3: Point(end_deck, back_plane, end_deck_height, True, 3, forces, displacements, mass = point_masses),

    #"right" side
    4: Point(-end_deck, front_plane, end_deck_height, True, 4, forces, displacements, mass = point_masses),
    5: Point(-mid_deck, front_plane_mid, mid_deck_height, False, 5, forces, mass = 0.918),
    6: Point(mid_deck, front_plane_mid, mid_deck_height-0.05, False, 6, forces, mass = 0.918),
    7: Point(end_deck, front_plane, end_deck_height, True, 7, forces, displacements, mass = point_masses),

    #towers
    #positive tower
    8: Point(positive_tower, back_plane_far, tower_bottom, True, 8, forces, displacements, mass = point_masses),
    9: Point(positive_tower, front_plane_far, tower_bottom, True, 9, forces, displacements, mass = point_masses),
    10: Point(positive_tower, back_plane_far, tower_top, True, 10, forces, displacements, mass = point_masses),
    11: Point(positive_tower, front_plane_far, tower_top, True, 11, forces, displacements, mass = point_masses),

    #negative tower
    12: Point(negative_tower, back_plane_far, tower_bottom, True, 8, forces, displacements, mass = point_masses),
    13: Point(negative_tower, front_plane_far, tower_bottom, True, 9, forces, displacements, mass = point_masses),
    14: Point(negative_tower, back_plane_far, tower_top, True, 10, forces, displacements, mass = point_masses),
    15: Point(negative_tower, front_plane_far, tower_top, True, 11, forces, displacements, mass = point_masses),
}

elts = {

    # CHANGE YIELD VALUES

    #deck
    0: LineElement(points[0], points[1], k_deck, SC, 1), # concrete
    1: LineElement(points[1], points[2], k_deck, SC, 1),
    2: LineElement(points[2], points[3], k_deck, SC, 1),
    3: LineElement(points[4], points[5], k_deck, SC, 1),
    4: LineElement(points[5], points[6], k_deck, SC, 1),
    5: LineElement(points[6], points[7], k_deck, SC, 1),
    6: LineElement(points[0], points[4], k_deck, SC, 1),

    7: LineElement(points[1], points[5], k_deck, SC, 1),
    8: LineElement(points[2], points[6], k_deck, SC, 1),

    9: LineElement(points[3], points[7], k_deck, SC, 1),

    #towers
    10: LineElement(points[8], points[10], k_tower, SS, 0.25, unstretched_length=tower_top), # pillars
    11: LineElement(points[9], points[11], k_tower, SS, 0.25, unstretched_length=tower_top),

    12: LineElement(points[12], points[14], k_tower, SS, 0.25, unstretched_length=tower_top), # pillars
    13: LineElement(points[13], points[15], k_tower, SS, 0.25, unstretched_length=tower_top),

    # cables
    14: LineElement(points[14], points[1], k_cable, SS, 0.01, unstretched_length = 11.45/in_to_m), # steel
    15: LineElement(points[10], points[2], k_cable, SS, 0.01, unstretched_length = 11.45/in_to_m),
    16: LineElement(points[15], points[5], k_cable, SS, 0.01, unstretched_length = 11.45/in_to_m),
    17: LineElement(points[11], points[6], k_cable, SS, 0.01, unstretched_length = 11.45/in_to_m),
}

sensors = {
    1: Sensor(parent = points[1], alpha = 0.5, noise_std = 0.001, add_noise = False),
    2: Sensor(parent = points[2], alpha = 0.5, noise_std = 0.001, add_noise = False),
    5: Sensor(parent = points[5], alpha = 0.5, noise_std = 0.001, add_noise = False),
    6: Sensor(parent = points[6], alpha = 0.5, noise_std = 0.001, add_noise = False)
}
erf_1 = error_function(points[1], sensors[1], elts[14], delay = delay)
erf_2 = error_function(points[2], sensors[2], elts[15], delay = delay)
erf_5 = error_function(points[5], sensors[5], elts[16], delay = delay)
erf_6 = error_function(points[6], sensors[6], elts[17], delay = delay)


if run_parameter_search:

    best_PID = None
    best_squerror = None
    for P in np.arange(0.0, 1.0, 0.005):
        for I in np.arange(0, 0.01, 0.005):
            for D in np.arange(0, 0.055, 0.005):
    # for P in np.arange(0, 10, 0.1):
    #     for I in np.arange(0, 5, 5):
    #         for D in np.arange(0, 10, 0.05):


                points = {
                    #"left" side
                    0: Point(-end_deck, back_plane, end_deck_height, True, 0, forces, displacements, mass = point_masses),
                    1: Point(-mid_deck, back_plane_mid, mid_deck_height, False, 1, forces, mass = 0.918),
                    2: Point(mid_deck, back_plane_mid, mid_deck_height, False, 2, forces, mass = 0.918),
                    3: Point(end_deck, back_plane, end_deck_height, True, 3, forces, displacements, mass = point_masses),

                    #"right" side
                    4: Point(-end_deck, front_plane, end_deck_height, True, 4, forces, displacements, mass = point_masses),
                    5: Point(-mid_deck, front_plane_mid, mid_deck_height, False, 5, forces, mass = 0.918),
                    6: Point(mid_deck, front_plane_mid, mid_deck_height-0.05, False, 6, forces, mass = 0.918),
                    7: Point(end_deck, front_plane, end_deck_height, True, 7, forces, displacements, mass = point_masses),

                    #towers
                    #positive tower
                    8: Point(positive_tower, back_plane_far, tower_bottom, True, 8, forces, displacements, mass = point_masses),
                    9: Point(positive_tower, front_plane_far, tower_bottom, True, 9, forces, displacements, mass = point_masses),
                    10: Point(positive_tower, back_plane_far, tower_top, True, 10, forces, displacements, mass = point_masses),
                    11: Point(positive_tower, front_plane_far, tower_top, True, 11, forces, displacements, mass = point_masses),

                    #negative tower
                    12: Point(negative_tower, back_plane_far, tower_bottom, True, 8, forces, displacements, mass = point_masses),
                    13: Point(negative_tower, front_plane_far, tower_bottom, True, 9, forces, displacements, mass = point_masses),
                    14: Point(negative_tower, back_plane_far, tower_top, True, 10, forces, displacements, mass = point_masses),
                    15: Point(negative_tower, front_plane_far, tower_top, True, 11, forces, displacements, mass = point_masses),
                }

                elts = {

                    # CHANGE YIELD VALUES

                    #deck
                    0: LineElement(points[0], points[1], k_deck, SC, 1), # concrete
                    1: LineElement(points[1], points[2], k_deck, SC, 1),
                    2: LineElement(points[2], points[3], k_deck, SC, 1),
                    3: LineElement(points[4], points[5], k_deck, SC, 1),
                    4: LineElement(points[5], points[6], k_deck, SC, 1),
                    5: LineElement(points[6], points[7], k_deck, SC, 1),
                    6: LineElement(points[0], points[4], k_deck, SC, 1),

                    7: LineElement(points[1], points[5], k_deck, SC, 1),
                    8: LineElement(points[2], points[6], k_deck, SC, 1),

                    9: LineElement(points[3], points[7], k_deck, SC, 1),

                    #towers
                    10: LineElement(points[8], points[10], k_tower, SS, 0.25, unstretched_length=tower_top), # pillars
                    11: LineElement(points[9], points[11], k_tower, SS, 0.25, unstretched_length=tower_top),

                    12: LineElement(points[12], points[14], k_tower, SS, 0.25, unstretched_length=tower_top), # pillars
                    13: LineElement(points[13], points[15], k_tower, SS, 0.25, unstretched_length=tower_top),

                    # cables
                    14: LineElement(points[14], points[1], k_cable, SS, 0.01, unstretched_length = 11.45/in_to_m), # steel
                    15: LineElement(points[10], points[2], k_cable, SS, 0.01, unstretched_length = 11.45/in_to_m),
                    16: LineElement(points[15], points[5], k_cable, SS, 0.01, unstretched_length = 11.45/in_to_m),
                    17: LineElement(points[11], points[6], k_cable, SS, 0.01, unstretched_length = 11.45/in_to_m),
                }

                sensors = {
                    1: Sensor(parent = points[1], alpha = 0.5, noise_std = 0.001, add_noise = False),
                    2: Sensor(parent = points[2], alpha = 0.5, noise_std = 0.001, add_noise = False),
                    5: Sensor(parent = points[5], alpha = 0.5, noise_std = 0.001, add_noise = False),
                    6: Sensor(parent = points[6], alpha = 0.5, noise_std = 0.001, add_noise = False)
                }
                erf_1 = error_function(points[1], sensors[1], elts[14], delay = delay)
                erf_2 = error_function(points[2], sensors[2], elts[15], delay = delay)
                erf_5 = error_function(points[5], sensors[5], elts[16], delay = delay)
                erf_6 = error_function(points[6], sensors[6], elts[17], delay = delay)

            

                PIDs_search = [
                    PID(P, I, D, elts[14], erf_1),
                    # PID(P, I, D, elts[15], erf_2),
                    # PID(P, I, D, elts[16], erf_5),
                    PID(P, I, D, elts[17], erf_6),
                ]

                struct_search = Structure(elements = elts, PIDs = PIDs_search, sensors = sensors)

                for i in range(vid_len):
                    struct_search.timestep(dt)

                sqerror_search = sum(sum(dt * e ** 2 for e in controller.evec) for controller in PIDs_search)

                if not best_PID or sqerror_search < best_squerror:
                    best_PID = (P, I, D)
                    best_squerror = sqerror_search

                for  _, point in points.items():
                    point.reset()
                for _, elt in elts.items():
                    elt.reset()
                for _, sensor in sensors.items():
                    sensor.reset()
                for pid in PIDs_search:
                    pid.reset()
                    

                    

    print(best_PID)
    print(best_squerror)

if make_animation:

    # parameters to make animation
    Kp = 1.5
    Ki = 0.0
    Kd = 0.5

    PIDs_animate = [
        PID(Kp, Ki, Kd, elts[14], erf_1),
        # PID(Kp, Ki, Kd, elts[13], erf_2),
        # PID(Kp, Ki, Kd, elts[14], erf_5),
        PID(Kp, Ki, Kd, elts[17], erf_6),
    ]

    struct_animate = Structure(elements = elts, PIDs = PIDs_animate, sensors = sensors)

    with imageio.get_writer(f'{(Kp, Ki, Kd)}.gif', mode='I', fps = math.floor(1/dt)) as writer:
        for i in range(vid_len):
            struct_animate.timestep(dt)
            struct_animate.plot(i = i)

            image = imageio.imread(f"img/{i}.png")
            writer.append_data(image)

            plt.close()

    sqerror = sum(sum(dt * e ** 2 for e in controller.evec) for controller in PIDs_animate)

    fig, ax = plt.subplots(figsize = (24, 8))

    ax.set_xlabel("timestep")
    ax.set_ylabel("error")
    ax.set_title(f"Error versus Timestep, {(Kp, Ki, Kd)}, s. sq. er. = {sqerror:0.3f}")

    for controller in PIDs_animate:
        ax.plot(
            controller.evec, lw=0.1
        )
    plt.savefig(f"{(Kp, Ki, Kd)}.png", dpi=300)
    plt.savefig(f"{(Kp, Ki, Kd)}.pdf")

    all_evecs = [controller.evec for controller in PIDs_animate]
    max_len = max(len(evec) for evec in all_evecs)

    time_stamps = [round(dt * i, 3) for i in range(max_len)]  

    all_data = [time_stamps] + all_evecs

    with open('controllers_evecs.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['t_s'] + [f'err{i+1}' for i in range(len(all_evecs))]
        writer.writerow(header)
        for row in zip_longest(*all_data, fillvalue=''):
            writer.writerow(row)

