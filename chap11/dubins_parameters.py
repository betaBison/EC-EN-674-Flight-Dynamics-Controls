# dubins_parameters
#   - Dubins parameters that define path between two configurations
#
# mavsim_matlab
#     - Beard & McLain, PUP, 2012
#     - Update history:
#         3/26/2019 - RWB

import numpy as np
import sys
sys.path.append('..')


class dubins_parameters:
    def __init__(self):
        self.p_s = np.inf*np.ones((3,1))  # the start position in re^3
        self.chi_s = np.inf  # the start course angle
        self.p_e = np.inf*np.ones((3,1))  # the end position in re^3
        self.chi_e = np.inf  # the end course angle
        self.radius = np.inf  # turn radius
        self.length = np.inf  # length of the Dubins path
        self.center_s = np.inf*np.ones((3,1))  # center of the start circle
        self.dir_s = np.inf  # direction of the start circle
        self.center_e = np.inf*np.ones((3,1))  # center of the end circle
        self.dir_e = np.inf  # direction of the end circle
        self.r1 = np.inf*np.ones((3,1))  # vector in re^3 defining half plane H1
        self.r2 = np.inf*np.ones((3,1))  # vector in re^3 defining position of half plane H2
        self.r3 = np.inf*np.ones((3,1))  # vector in re^3 defining position of half plane H3
        self.n1 = np.inf*np.ones((3,1))  # unit vector in re^3 along straight line path
        self.n3 = np.inf*np.ones((3,1))  # unit vector defining direction of half plane H3

    def update(self, ps, chis, pe, chie, R):
        ell = np.linalg.norm(ps - pe)
        if ell < 2 * R:
            print('Error in Dubins Parameters: The distance between nodes must be larger than 2R.')
        else:
            crs = ps + np.matmul(R*rotz(np.pi/2.), np.array([[np.cos(chis),np.sin(chis),0.]]).T)
            cls = ps + np.matmul(R*rotz(-np.pi/2.), np.array([[np.cos(chis),np.sin(chis),0.]]).T)
            cre = pe + np.matmul(R*rotz(np.pi/2.), np.array([[np.cos(chie),np.sin(chie),0.]]).T)
            cle = pe + np.matmul(R*rotz(-np.pi/2.), np.array([[np.cos(chie),np.sin(chie),0.]]).T)
            e1 = np.array([[1.,0.,0.]]).T

            # L1 calculation
            theta = np.pi/2. - np.arctan2(cre.item(0)-crs.item(0),cre.item(1)-crs.item(1))
            L1 = np.linalg.norm(crs-cre) + R*mod(2.*np.pi + mod(theta-np.pi/2.) - mod(chis - np.pi/2.)) \
                + R*mod(2.*np.pi + mod(chie-np.pi/2.) - mod(theta - np.pi/2.))

            # L2 calculation
            theta = np.pi/2. - np.arctan2(cle.item(0)-crs.item(0),cle.item(1)-crs.item(1))
            ll = np.linalg.norm(cle-crs)
            theta2 = theta - np.pi/2. + np.arcsin(2.*R/ll)
            L2 = np.sqrt(ll**2-4.*R**2) + R*mod(2.*np.pi + mod(theta2) - mod(chis - np.pi/2.)) \
                + R*mod(2.*np.pi + mod(theta2 + np.pi) - mod(chie + np.pi/2.))

            # L3 calculation
            theta = np.pi/2. - np.arctan2(cre.item(0)-cls.item(0),cre.item(1)-cls.item(1))
            ll = np.linalg.norm(cre-cls)
            theta2 = np.arccos(2.*R/ll)
            L3 = np.sqrt(ll**2-4.*R**2) + R*mod(2.*np.pi + mod(chis + np.pi/2.) - mod(theta-theta2)) \
                + R*mod(2.*np.pi + mod(chie - np.pi/2.) - mod(theta+theta2-np.pi))

            # L4 calculation
            theta = np.pi/2. - np.arctan2(cle.item(0)-cls.item(0),cle.item(1)-cls.item(1))
            L4 = np.linalg.norm(cls-cle) + R*mod(2.*np.pi + mod(chis + np.pi/2.) - mod(theta + np.pi/2.) ) \
                + R*mod(2.*np.pi + mod(theta + np.pi/2.) - mod(chie + np.pi/2.))

            L = np.argmin([L1,L2,L3,L4])
            if L == 0:
                self.center_s = crs
                self.dir_s = 1
                self.center_e = cre
                self.dir_e = 1
                self.n1 = (self.center_e - self.center_s)/np.linalg.norm(self.center_e - self.center_s)
                self.r1 = self.center_s + np.matmul(R*rotz(-np.pi/2.), self.n1)
                self.r2 = self.center_e + np.matmul(R*rotz(-np.pi/2.), self.n1)
            elif L == 1:
                self.center_s = crs
                self.dir_s = 1
                self.center_e = cle
                self.dir_e = -1
                theta = np.pi/2. - np.arctan2(cle.item(0)-crs.item(0),cle.item(1)-crs.item(1))
                ll = np.linalg.norm(cle-crs)
                theta2 = theta - np.pi/2. + np.arcsin(2.*R/ll)
                self.n1 = np.matmul(rotz(theta2 + np.pi/2.), e1)
                self.r1 = self.center_s + np.matmul(R*rotz(theta2), e1)
                self.r2 = self.center_e + np.matmul(R*rotz(theta2 + np.pi), e1)
            elif L == 2:
                self.center_s = cls
                self.dir_s = -1
                self.center_e = cre
                self.dir_e = 1
                theta = np.pi/2. - np.arctan2(cre.item(0)-cls.item(0),cre.item(1)-cls.item(1))
                ll = np.linalg.norm(cre-cls)
                theta2 = np.arccos(2.*R/ll)
                self.n1 = np.matmul(rotz(theta + theta2 - np.pi/2.), e1)

                self.r1 = self.center_s + np.matmul(R*rotz(theta + theta2), e1)
                self.r2 = self.center_e + np.matmul(R*rotz(theta + theta2 - np.pi), e1)
            else:
                self.center_s = cls
                self.dir_s = -1
                self.center_e = cle
                self.dir_e = -1
                self.n1 = (self.center_e - self.center_s)/np.linalg.norm(self.center_e - self.center_s)
                self.r1 = self.center_s + np.matmul(R*rotz(np.pi/2.), self.n1)
                self.r2 = self.center_e + np.matmul(R*rotz(np.pi/2.), self.n1)

            self.r3 = pe
            self.n3 = np.matmul(rotz(chie), e1)
            # input parameters for graphing use
            self.p_s = ps
            self.chi_s = chis
            self.p_e = pe
            self.chi_e = chie
            self.radius = R
            self.length = [L1,L2,L3,L4][L]


def rotz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])


def mod(x):
    # make x between 0 and 2*pi
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x
