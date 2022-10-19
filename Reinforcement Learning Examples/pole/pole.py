# pylint: disable=invalid-name
import numpy as np
import math

class Pole():
    """
    ### Description
        main : controls simulation iterations and implements the learning system

        cart_and_pole: the cart and pole dynamics; given action and current state,
        estimates the next state

        get_box: cart-pole's state space divided into 162 boxes
    """
    def __init__(self):
        self.N_BOXES = 162
        self.ALPHA = 1000
        self.BETA = 0.5
        self.GAMMA = 0.95
        self.LAMBDAw = 0.9
        self.LAMBDAv = 0.8
        self.MAX_FAILURES = 100
        self.MAX_STEPS = 100000

        self.state = np.float32([0, 0, 0, 0]) # x, x_dot, theta, theta_dot
        self.w = np.zeros(self.N_BOXES)
        self.v = np.zeros(self.N_BOXES)
        self.e = np.zeros(self.N_BOXES)
        self.xbar = np.zeros(self.N_BOXES)

        # parameters required for simulation
        self.GRAVITY = 9.8
        self.MASSCART = 1.0
        self.MASSPOLE = 0.1
        self.TOTAL_MASS = self.MASSCART + self.MASSPOLE
        self.LENGTH = 0.5
        self.POLEMASS_LENGTH = self.MASSPOLE * self.LENGTH
        self.FORCE_MAG = 10.0
        self.TAU = 0.02
        self.FOURTHIRDS = 1.3333333333333

        # parameters required for getting box from current state
        self.one_degree = 0.0174532
        self.six_degrees = 0.1047192
        self.twelve_degrees = 0.2094384
        self.fifty_degrees = 0.87266

    def prob_push_right(self, s):
        return (1.0 / (1.0 + math.exp(-1 * max(-50.0, min(s, 50.0)))))

    def get_box(self):
        box = 0
        x = self.state[0]
        x_dot = self.state[1]
        theta = self.state[2]
        theta_dot = self.state[3]

        if (x < -2.4 or x > 2.4  or theta < -self.twelve_degrees or theta > self.twelve_degrees):
                return (-1) # failure

        if(x < -0.8):
            box = 0
        elif(x < 0.8):
            box = 1
        else:
            box = 2


        if(x_dot < -0.5):
            pass
        elif(x < 0.5):
            box += 3
        else:
            box += 6

        if (theta < -self.six_degrees):
            pass
        elif(theta < -self.one_degree):
            box += 9
        elif (theta < 0):
            box += 18
        elif (theta < self.one_degree):
            box += 27
        elif (theta < self.six_degrees):
            box += 36
        else:
            box += 45

        if (theta_dot < -self.fifty_degrees):
            pass
        elif(theta_dot < self.fifty_degrees):
            box += 54
        else:
            box += 108
        return box

    def cart_pole(self, action):
        force = self.FORCE_MAG if (action > 0) else (-1 * self.FORCE_MAG)
        costheta = np.cos(self.state[2])
        sintheta = np.sin(self.state[2])

        temp = (force + self.POLEMASS_LENGTH * self.state[3] * self.state[3] * sintheta) / self.TOTAL_MASS
        thetaacc = (self.GRAVITY * sintheta - costheta * temp) / (self.LENGTH * (self.FOURTHIRDS - self.MASSPOLE * costheta * costheta / self.TOTAL_MASS))
        xacc = temp - self.POLEMASS_LENGTH * thetaacc * costheta / self.TOTAL_MASS

        self.state[0] += self.TAU * self.state[1]
        self.state[1] += self.TAU * xacc
        self.state[2] += self.TAU * self.state[3]
        self.state[3] += self.TAU * thetaacc

    def main(self):
        steps = 0
        failures = 0

        box = self.get_box()

        while (steps < self.MAX_STEPS and failures < self.MAX_FAILURES):
            steps = steps + 1
            y = (np.random.rand() < self.prob_push_right(self.w[box]))
            e_o = (1.0 - self.LAMBDAw) * (y - 0.5)
            self.e[box] = e_o
            xbar_o = (1.0 - self.LAMBDAv)
            self.xbar[box] = xbar_o

            oldp = self.v[box]
            self.cart_pole(y)
            box = self.get_box()

            if box < 0:
                failed = 1
                failures = failures + 1
                print(f'Trial {failures} was {steps} steps. \n')
                steps = 0
                self.state = np.float32([0,0,0,0])
                r = -1.0
                p = 0.
            else:
                failed = 0
                r = 0
                p = self.v[box]

            rhat = r + self.GAMMA * p - oldp

            self.w = np.add(self.w, (self.ALPHA * rhat) * self.e)
            self.v = np.add(self.v, (self.BETA * rhat) * self.xbar)

            if (failed):
                    self.e = np.zeros(self.N_BOXES)
                    self.xbar = np.zeros(self.N_BOXES)
            else:
                self.e = self.LAMBDAw * self.e
                self.xbar = self.LAMBDAv * self.xbar
        if (failures ==self.MAX_FAILURES):
            print(f'Pole not balanced. Stopping after {failures} failures.')
        else:
            print(f'Pole balanced successfully for at least {steps} steps\n')


p = Pole()
p.main()
