import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, dt, u_x,u_y, std_acc, x_std_meas, y_std_meas):
        """
        :param dt: sampling time (time for 1 cycle)
        :param u_x: acceleration in x-direction
        :param u_y: acceleration in y-direction
        :param std_acc: process noise magnitude
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        """

        self.initialization(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)


    def initialization(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):

        # Define sampling time
        self.dt = dt

        # Define the  control input variables
        self.u = np.matrix([[u_x],[u_y]])

        # Intial State
        self.x = np.matrix([[0], [0], [0], [0]])

        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Define the Control Input Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        #Initial Process Noise Covariance
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        #Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2,0],
                           [0, y_std_meas**2]])

        #Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])


    def predict(self):

        # Update time state
        #x_k =Ax_(k-1) + Bu_(k-1)     Eq.(9)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(10)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]

    def update(self, z):

        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  #Eq.(11)

        # self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))   #Eq.(12)
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))   #Eq.(12)

        I = np.eye(self.H.shape[1])

        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P   #Eq.(13)
        return self.x[0:2]

    def filtering(self, measure):

        '''
        measure : seq_len x 2
        '''

        update_ = []
        for z in measure:

            # predict
            x_p, y_p = self.predict()

            # update
            x_u, y_u = self.update(np.matrix(z.reshape(2, 1)))
            update_.append(np.array([x_u.item(0), y_u.item(0)]).reshape(1, 2))

        update_ = np.concatenate(update_, axis=0)

        return update_


    def prediction(self, measure):


        '''
        measure : seq_len x 2
        '''

        update_ = []
        for z in measure:

            # predict
            x_p, y_p = self.predict()

            if (z[0] == -1000):
                z = np.array([x_p.item(0), y_p.item(0)])

            # update
            x_u, y_u = self.update(np.matrix(z.reshape(2, 1)))
            update_.append(np.array([x_u.item(0), y_u.item(0)]).reshape(1, 2))

        update_ = np.concatenate(update_, axis=0)

        return update_

class AverageFilter(object):
    def __init__(self, filter_size):

        self.f_size = filter_size
        self.f_size_h = int(filter_size / 2)

    def op(self, measure):

        seq_len = measure.shape[0]
        measure_filter = np.copy(measure)

        for t in range(self.f_size_h, seq_len-self.f_size_h):
            if (True not in np.isnan(measure[t-self.f_size_h:t+self.f_size_h+1, 0]).tolist()):
                measure_filter[t, :] = np.mean(measure[t-self.f_size_h:t+self.f_size_h+1], axis=0)

        return measure_filter

class LinearModelIntp(object):
    def __init__(self):

        self.f_size = 3

    def op(self, measure):

        seq_len = measure.shape[0]
        measure_filter = np.copy(measure)

        for t in range(self.f_size, seq_len):

            ppprev = measure[t - 3]
            pprev = measure[t - 2]
            prev = measure[t - 1]
            cur = measure[t]

            # val, val, nan
            if (not np.isnan(ppprev[0]) and not np.isnan(pprev[0]) and not np.isnan(prev[0]) and np.isnan(cur[0])):
                measure_filter[t, :] = self.linear_model(ppprev, pprev, prev)

        return measure_filter


    def linear_model(self, p0, p1, p2):

        return (p2 + (p2 - p1) + 0.5*(p2 + p0 -2*p1))

class HoleFilling(object):
    def __init__(self):

        self.f_size = 3
        self.f_size_h = 1

    def op(self, measure):

        seq_len = measure.shape[0]
        measure_filter = np.copy(measure)

        for t in range(self.f_size_h, seq_len-self.f_size_h):

            prev = measure[t-1]
            cur = measure[t]
            next = measure[t+1]

            if (np.isnan(cur[0]) and not np.isnan(prev[0]) and not np.isnan(next[0])):
                measure_filter[t, :] = 0.5 * (prev + next)

        return measure_filter