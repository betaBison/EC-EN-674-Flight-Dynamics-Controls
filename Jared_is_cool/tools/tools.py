import numpy as np
from math import cos, sin

def Quaternion2Euler(quad):
  e0 = quad.item(0)
  e1 = quad.item(1)
  e2 = quad.item(2)
  e3 = quad.item(3)

  phi = np.arctan2(2*(e0*e1+e2*e3),(e0**2+e3**2-e1**2-e2**2))
  theta = np.arcsin(2*(e0*e2 - e1*e3))
  psi = np.arctan2(2*(e0*e3+e1*e2),(e0**2+e1**2-e2**2-e3**2))

  return phi, theta, psi


def Euler2Quaternion(phi,theta,psi):
  c_psi = np.cos(psi/2.)
  c_theta = np.cos(theta/2.)
  c_phi = np.cos(phi/2.)
  s_psi = np.sin(psi/2.)
  s_theta = np.sin(theta/2.)
  s_phi = np.sin(phi/2.)

  e0 = c_psi*c_theta*c_phi + s_psi*s_theta*s_phi
  e1 = c_psi*c_theta*s_phi - s_psi*s_theta*c_phi
  e2 = c_psi*s_theta*c_phi + s_psi*c_theta*s_phi
  e3 = s_psi*c_theta*c_phi - c_psi*s_theta*s_phi

  return np.array([e0,e1,e2,e3]).reshape(4,1)


def Euler2Rotation(phi, theta, psi):
  """
  Converts euler angles to rotation matrix (R_b^i, i.e., body to inertial)
  """
  # only call sin and cos once for each angle to speed up rendering
  c_phi = np.cos(phi)
  s_phi = np.sin(phi)
  c_theta = np.cos(theta)
  s_theta = np.sin(theta)
  c_psi = np.cos(psi)
  s_psi = np.sin(psi)
  R_roll = np.array([[1, 0, 0],
                     [0, c_phi, s_phi],
                     [0, -s_phi, c_phi]])
  R_pitch = np.array([[c_theta, 0, -s_theta],
                      [0, 1, 0],
                      [s_theta, 0, c_theta]])
  R_yaw = np.array([[c_psi, s_psi, 0],
                    [-s_psi, c_psi, 0],
                    [0, 0, 1]])
  R = R_roll @ R_pitch @ R_yaw  # inertial to body (Equation 2.4 in book)
  return R.T  # transpose to return body to inertial


def Quaternion2Rotation(quad):
  e0 = quad.item(0)
  e1 = quad.item(1)
  e2 = quad.item(2)
  e3 = quad.item(3)

  # also returning rotation from body to inertial
  return np.array([[e0**2+e1**2-e2**2-e3**2, 2*(e1*e2-e0*e3), 2*(e1*e3+e0*e2)],
                    [2*(e1*e2+e0*e3), e0**2-e1**2+e2**2-e3**2, 2*(e2*e3-e0*e1)],
                    [2*(e1*e3-e0*e2), 2*(e2*e3+e0*e1), e0**2-e1**2-e2**2+e3**2]])




def jacobian(fun, x, state):
  # compute jacobian of fun with respect to x
  f = fun(x, state)
  m = f.shape[0]
  n = x.shape[0]
  eps = 0.01  # deviation
  J = np.zeros((m, n))
  for i in range(0, n):
    x_eps = np.copy(x)
    x_eps[i][0] += eps
    f_eps = fun(x_eps, state)
    df = (f_eps - f) / eps
    J[:, i] = df[:, 0]
  return J
