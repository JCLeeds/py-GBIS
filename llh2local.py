"""
llh2local in python format based on matlab code used in GBIS written by Peter Cervelli in 2000 updated by Andy hopper in 2014 
converted to this python format by John Condon 2024 

Converts from longitude and latitude to local coorindates
given an origin.  llh (lon; lat; height) and origin should
be in decimal degrees. Note that heights are ignored and
that xy is in km.
origin in lon,lat
""" 
import numpy as np 

def llh2local(llh,origin):
    # WGS84 Constants
    a = 6378137
    e = 0.08209443794970
    #Convert to Radians 
    llh = llh*np.pi/180
    origin = origin*np.pi/180
    print(len(llh[1,:]))
    print(len(llh[:,1]))
    print(np.shape(llh))
   
    z = np.where(llh[1,:] != 0) 
    print(z)
    dlamba = llh[0,z] - origin[0]
    M=a*((1-e**2/4-3*e**4/64-5*e**6/256)*llh[1,z] - 
        (3*e**2/8+3*e**4/32+45*e**6/1024)*np.sin(2*llh[1,z]) + 
        (15*e**4/256 +45*e**6/1024)*np.sin(4*llh[1,z]) - 
        (35*e**6/3072)*np.sin(6*llh[1,z]))
    
    M0=a*((1-e**2/4-3*e**4/64-5*e**6/256)*origin[1] - 
        (3*e**2/8+3*e**4/32+45*e**6/1024)*np.sin(2*origin[1]) + 
        (15*e**4/256 +45*e**6/1024)*np.sin(4*origin[1]) - 
        (35*e**6/3072)*np.sin(6*origin[1]))
    
    N=a/np.sqrt(1-e**2*np.sin(llh[1,z])**2)
    E=dlamba*np.sin(llh[1,z])
    xy = np.zeros(np.shape(llh))
    xy[0,z]=N*cot(llh[1,z])*np.sin(E)
    xy[1,z]=M-M0+N*cot(llh[1,z])*(1-np.cos(E))

    z_zero_catch = np.where(llh[1,:] == 0)
    dlamba=llh[0,z_zero_catch]-origin[0]
    xy[0,z_zero_catch]=a*dlamba
    xy[1,z_zero_catch]=-M0

    return xy  

def cot(angle_in_rad):
    return 1/np.tan(angle_in_rad)

    

