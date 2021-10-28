import numpy as np
class Sys_dynamics():
    def __init__(self) -> None:
        pass

    def launch_vehicle(self, x, u, dt):
        
        """
        Sun, Bo, and Erik-Jan van Kampen. 
        "Incremental model-based global dual heuristic programming for flight control." 
        IFAC-PapersOnLine 52.29 (2019): 7-12.
        https://doi.org/10.1016/j.ifacol.2019.12.613
        """

        # 
        alpha = x[0]
        q = x[1]
        deltae = x[2]
        deltae_c = u[0]



        Ma = 2.0
        tao_deltae = 0.01
        # constant parameters (Aermican unit system)
        # Richard A. Hull, 1995


        g = 32.2 # ft/sec^2
        W = 450 # lbs
        V = 3109.3 # ft/sec
        Iyy = 182.5 # slug*ft^2
        q_ = 6132.8 # lb/ft^2
        S = 0.44 # ft^2
        dl = 0.75 # ft

        m = W/g

        C1 = q_*S/(m*V)
        C2 = (q_*S*dl)/Iyy



        # areodynamics 
        # Seung-Hwan Kim, 2004

        b1 = 1.6238
        b2 = -6.7240
        b3 = 12.0393
        b4 = -48.2246

        h1 = -288.7
        h2 = 50.32
        h3 = -23.89

        h4 = 303.1
        h5 = -246.3
        h6 = -37.56

        h7 = -13.53
        h8 = 4.185

        h9 = 71.51
        h10= 10.01



        phiz1 = h1*alpha**3 + h2*alpha*np.abs(alpha) + h3*alpha
        phim1 = h4*alpha**3 + h5*alpha*np.abs(alpha) + h6*alpha

        phiz2 = h7*alpha*np.abs(alpha) + h8*alpha
        phim2 = h9*alpha*np.abs(alpha) + h10*alpha

        Cz1 = phiz1 + phiz2*Ma
        Cm1 = phim1 + phim2*Ma

        Bz = b1*Ma + b2
        Bm = b3*Ma + b4

        f1 = C1*Cz1
        f2 = C2*Cm1

        g1 = C1*Bz
        g2 = C2*Bm


        dalpha = q + f1 + g1*deltae
        dq = f2 + g2*deltae
        ddelate = (deltae_c - deltae)/tao_deltae

        d_x = np.array([dalpha, dq, ddelate])

        x_next = d_x*dt + x

        return x_next
        
    def f16_lateral(self, x, u, dt):
        
        """
        Brian L Stevens, Frank L Lewis, and Eric N Johnson.
        Aircraft control and simulation:  dynamics, controlsdesign, and autonomous systems. 
        John Wiley & Sons, 2015.
        """

        # 
        A = np.array([[-0.3220,0.0640,0.0364,-0.9917,0.0003,0.0008],
        [0,0,1,0.0037,0,0],
        [-30.6492,0,-3.6784,0.6646,-0.7333,0.1315],
        [8.5396,0,-0.0254,-0.4764,-0.0319,-0.0620],
        [0,0,0,0,-20.2,0],
        [0,0,0,0,0,-20.2],
        ])
        B = np.array([[0,0],[0,0],[0,0],[0,0],
        [20.2,0],[0,20.2]])
        d_x = np.matmul(A,x) + np.matmul(B,u)
        x_next = d_x*dt + x

        return x_next

    def sate_liqslosh(self, x, u, dt):
        """
        Sun, Bo, and Erik-Jan van Kampen. 
        "Incremental model-based heuristic dynamic programming 
        with output feedback applied to aerospace system identification and control." 
        2020 IEEE Conference on Control Technology and Applications (CCTA). IEEE, 2020.
        https://doi.org/10.1109/CCTA41146.2020.9206261
        """
        
        # states
        theta = x[0]
        dTheta = x[1]
        psi = x[2]
        dPsi = x[3]

        # input
        fs = u[0]
        Ms = u[1]

        # model A
        ms = 600 #  kg
        Is = 720 # kg/m^2
        mp = 100 # 100kg
        Ip = 90 # kg/m^2
        a = 0.3 # m
        b = 0.3 # m
        Fs = 500 # N
        kappa = 0.19 # kg*m^2/s

        # # model B
        # ms = 1200 #  kg
        # Is = 900 # kg/m^2
        # mp = 50 # 100kg
        # Ip = 80 # kg/m^2
        # a = 0.2 # m
        # b = 0.5 # m
        # Fs = 600 # N
        # kappa = 0.19 # kg*m^2/s

        bStar = ms*b/(ms + mp)
        aStar = mp*a/(ms + mp)

        T1 = (Is+ms*b**2) - bStar*ms*b - bStar*mp*a*np.cos(psi)
        T2 = bStar*mp*a*np.cos(psi)
        T3 = Ms + b*fs + kappa*dPsi - bStar*fs - bStar*mp*a*(dPsi + dTheta)**2*np.sin(psi)

        T4 = aStar*(np.sin(psi)*(Fs - ms*b*dTheta) + np.cos(psi)*fs) + kappa*dPsi
        T5 = aStar*np.cos(psi)*ms*b
        T6 = mp*a**2 + Ip - aStar*mp*a



        d2Theta = (T3*T6-T2*T4)/((T6-T5)*T2+T1*T6)
        d2Psi = (T1*T4+(T6-T5)*T3)/(T2*(T5-T6)-T1*T6)

        d_x = np.array([dTheta, d2Theta, dPsi, d2Psi])

        x_next = d_x*dt + x

        return x_next


