# -*- coding: utf-8 -*-
"""
lib full cython 
"""
from libc.math cimport exp, cos
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from cython.operator cimport dereference, preincrement
import numpy as np
cimport numpy as np
from libcpp.queue cimport queue
from cython.parallel cimport parallel, prange
cimport cython


cdef class OriginCompartment:
    cdef  double V, Isyn, Iext, Icoms
    cdef np.ndarray Vhist
    cdef np.ndarray LFP
    cdef np.ndarray firing
    
    def __cinit__(self, params):
        pass
    
    cdef double getV(self):
        return self.V
        
    
    cdef void setIext(self, double Iext):
        self.Iext = Iext
        
    def addIsyn(self, double Isyn):
        self.Isyn += Isyn
    
    def addIcoms(self, double Icoms):
        self.Icoms += Icoms  
    
        
    def getVhist(self):
        return self.Vhist
    
    def getLFP(self):
        return self.LFP
    
    def getFiring(self):
        return self.firing
        
    cpdef integrate(self, double dt, double duration):
        pass
    
    def getCompartmentsNames(self):
        return ["soma"]

    def getCompartmentByName(self, name): 
        return self
 
    cpdef checkFired(self, double t_):
       pass

####################
cdef class PyramideCA1Compartment(OriginCompartment):
    cdef double Capacity, Iextmean, Iextvarience, ENa, EK, El, ECa, CCa, sfica, sbetaca
    cdef double gbarNa, gbarK_DR, gbarK_AHP, gbarK_C, gl, gbarCa
    cdef double th
    cdef bool countSp
    cdef double m, h, n, s, c, q
    cdef double INa, IK_DR, IK_AHP, IK_C, ICa, Il
    cdef double distance
    cdef np.ndarray input_conduntance, conduntances_Erev
    cdef int conduntance_counter
    
    def __cinit__(self, params):
        self.V = params["V0"]
        self.Capacity = params["C"]
        
        self.Iextmean = params["Iextmean"]        
        self.Iextvarience = params["Iextvarience"]
        
        self.ENa = params["ENa"]
        self.EK = params["EK"]
        self.El = params["El"]
        self.ECa = params["ECa"]
        
        self.CCa = params["CCa"]
        self.sfica = params["sfica"]
        self.sbetaca = params["sbetaca"]
        
        self.gbarNa = params["gbarNa"]
        self.gbarK_DR = params["gbarK_DR"]

        self.gbarK_AHP = params["gbarK_AHP"]        
        self.gbarK_C = params["gbarK_C "]        

        self.gl = params["gl"]
        self.gbarCa = params["gbarCa"]

        self.input_conduntance = params["input_conduntance"]
        self.conduntances_Erev = params["conduntances_Erev"]
        self.conduntance_counter = 0

        self.Vhist = np.array([])
        self.LFP = np.array([])
        self.distance = np.random.normal(8, 2)
        
        self.firing = np.array([])
        self.th = self.El + 40
        
        self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
        self.h = self.alpha_h() / (self.alpha_h() + self.beta_h())
        self.n = self.alpha_n() / (self.alpha_n() + self.beta_n())
        self.s = self.alpha_s() / (self.alpha_s() + self.beta_s())
        self.c = self.alpha_c() / (self.alpha_c() + self.beta_c())
        self.q = self.alpha_q() / (self.alpha_q() + self.beta_q())
        
        self.calculate_currents()

    cdef void calculate_currents(self):
        self.Il = self.gl * (self.V - self.El)
        self.INa = self.gbarNa * self.m * self.m * self.h * (self.V - self.ENa)
        self.IK_DR = self.gbarK_DR * self.n * (self.V - self.EK)
        self.IK_AHP = self.gbarK_AHP * self.q * (self.V - self.EK)
        self.IK_C = self.gbarK_C * self.c * (self.V - self.EK)
        
        
        cdef double tmp = self.CCa / 250.0
        if (tmp < 1):
            self.IK_C *= tmp    
        
        self.ICa = self.gbarCa * self.s * self.s * (self.V - self.ECa)
        self.Iext = np.random.normal(self.Iextmean, self.Iextvarience)
        self.Isyn = 0
        self.Icoms = 0

    cdef double alpha_m(self):
        cdef double x = 13.1 - self.V
        if (x == 0):
            x = 0.01
        cdef double alpha = 0.32 * x / (exp(0.25 * x) - 1)
        return alpha
        
        
    cdef double beta_m(self):
        cdef double x = self.V - 40.1
        if (x == 0):
            x = 0.01
        cdef double beta = 0.28 * x / (exp(0.2 * x) - 1)
        return beta
        
    cdef double alpha_h(self):
        cdef double alpha = 0.128 * exp((17 - self.V) / 18)
        return alpha
        
    cdef double beta_h(self):
        cdef double x = 40 - self.V 
        if (x == 0):
            x = 0.00001
        cdef double beta = 4 / (exp(0.2 * x) + 1)
        return beta

    cdef double alpha_n(self):
        cdef double x = 35.1 - self.V
        if (x == 0):
            x = 0.00001
        cdef double alpha = 0.016 * x / (exp(0.2 * x) - 1)
        return alpha

    cdef double beta_n(self):
        cdef double beta = 0.25 * exp(0.5 - 0.025 * self.V)
        return beta
        
    cdef double alpha_s(self):
        cdef double x = self.V - 65
        cdef double alpha = 1.6 / (1 + exp(-0.072 * x))
        return alpha
    
    cdef double beta_s(self):
        cdef double x = self.V - 51.1
        if (x == 0):
            x = 0.00001
        cdef double beta = 0.02 * x / (exp(0.2 * x) - 1)
        return beta

    cdef double alpha_c(self):
        cdef double alpha
        if(self.V > 50):
            alpha = 2 * exp((6.5 - self.V)/27)
        else:
            alpha = exp( ((self.V - 10)/11) - ((self.V - 6.5)/27) ) / 18.975   
        return alpha
    
    cdef double beta_c(self):
        cdef double beta
        if (self.V > 0):
            beta = 0
        else:
            beta = 2 * exp((6.5 - self.V)/27) - self.alpha_c()
        return beta
    
    cdef double alpha_q(self):
        cdef double alpha = 0.00002 * self.CCa
        if (alpha > 0.01):
            alpha = 0.01
        return alpha
    
    cdef double beta_q(self):
        return 0.001
    

    cdef double h_integrate(self, double dt):
        cdef double h_0 = self.alpha_h() / (self.alpha_h() + self.beta_h())
        cdef double tau_h = 1 / (self.alpha_h() + self.beta_h())
        return h_0 - (h_0 - self.h) * exp(-dt / tau_h)


    cdef double n_integrate(self, double dt):
        cdef double n_0 = self.alpha_n() / (self.alpha_n() + self.beta_n() )
        cdef double tau_n = 1 / (self.alpha_n() + self.beta_n())
        return n_0 - (n_0 - self.n) * exp(-dt / tau_n)
        
    cdef double s_integrate(self, double dt):
        cdef double s_0 = self.alpha_s() / (self.alpha_s() + self.beta_s() )
        cdef double tau_s = 1 / (self.alpha_s() + self.beta_s())
        return s_0 - (s_0 - self.s) * exp(-dt / tau_s)
    
    cdef double c_integrate(self, double dt):
        cdef double c_0 = self.alpha_c() / (self.alpha_c() + self.beta_c() )
        cdef double tau_c = 1 / (self.alpha_c() + self.beta_c())
        return c_0 - (c_0 - self.c) * exp(-dt / tau_c)
    
    cdef double q_integrate(self, double dt):
        cdef double q_0 = self.alpha_q() / (self.alpha_q() + self.beta_q() )
        cdef double tau_q = 1 / (self.alpha_q() + self.beta_q())
        return q_0 - (q_0 - self.q) * exp(-dt / tau_q)
    
    cdef double CCa_integrate(self, double dt):
        cdef double k1 = self.CCa
        cdef double k2 = k1 + 0.5 * dt * (- self.sfica * self.ICa - self.sbetaca * k1)
        cdef double k3 = k2 + 0.5 * dt * (- self.sfica * self.ICa - self.sbetaca * k2)
        cdef double k4 = k1 + dt * (- self.sfica * self.ICa - self.sbetaca * k1)        
        return (k1 + 2*k2 + 2*k3 + k4) / 6

    cpdef integrate(self, double dt, double duration):
        cdef double t = 0

        while (t < duration):
            self.Vhist = np.append(self.Vhist, self.V)

            I = -self.Il - self.INa - self.IK_DR - self.IK_AHP - self.IK_C - self.ICa - self.Isyn - self.Icoms + self.Iext
            if self.input_conduntance.shape[1] > self.conduntance_counter:
                I += np.sum( self.input_conduntance[:, self.conduntance_counter] * (self.conduntances_Erev - self.V ) )

            self.conduntance_counter  += 1
            #lfp = (I + self.Icoms) / (4 * np.pi * 0.3)
           
            #self.LFP = np.append(self.LFP, lfp)
            
            # if (self.Isyn > 10 or self.Isyn < -10):
            #    print (self.Isyn)
            
            self.V += dt * I / self.Capacity
     
            self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
            self.h = self.h_integrate(dt)
            self.n = self.n_integrate(dt)
            self.s = self.s_integrate(dt)
            self.c = self.c_integrate(dt)
            self.q = self.q_integrate(dt)
            self.CCa = self.CCa_integrate(dt)
     
            self.calculate_currents()
            self.Isyn = 0
            t += dt
    
    
    def checkFired(self, double t):
    
        if (self.V >= self.th and self.countSp):
            self.firing = np.append(self.firing, t)
            self.countSp = False
        
        if (self.V < self.th):
            self.countSp = True        
        
cdef class IntercompartmentConnection:
    cdef OriginCompartment comp1
    cdef OriginCompartment comp2
    cdef double g, p
    def __cinit__(self, OriginCompartment comp1, OriginCompartment comp2, double g, double p):
        self.comp1 = comp1
        self.comp2 = comp2
        self.g = g
        self.p = p
    
    def activate(self):
        
        cdef double Icomp1 = (self.g / self.p) * (self.comp1.getV() - self.comp2.getV())
        cdef double Icomp2 = (self.g/(1 - self.p)) * (self.comp2.getV() - self.comp1.getV())
        
        self.comp1.addIcoms(Icomp1)
        self.comp2.addIcoms(Icomp2)       

cdef class ComplexNeuron:
    cdef dict compartments # map [string, OriginCompartment*] compartments
    cdef list connections # vector [IntercompartmentConnection*] connections
    
    def __cinit__(self, list compartments, list connections):
        self.compartments = dict()
        
        for comp in compartments:
            key, value = comp.popitem()
            self.compartments[key] = PyramideCA1Compartment(value)
        

        self.connections = []
        for conn in connections:
            self.connections.append(IntercompartmentConnection(self.compartments[conn["compartment1"]], self.compartments[conn["compartment2"]], conn["g"], conn["p"]   ) )
        
    def getCompartmentsNames(self):
        return self.compartments.keys()
    
    def integrate(self, double dt, double duration):
        cdef double t = 0
        
        while(t < duration):
            for p in self.compartments.values():
                p.integrate(dt, dt)
                
            for c in self.connections:
                c.activate()
            
            t += dt
            
    def getCompartmentByName(self, name): 
        return self.compartments[name]
        
