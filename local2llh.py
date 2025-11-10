import numpy as np

def local2llh(xy, origin):
    # Set ellipsoid constants (WGS84)
    a = 6378137.0
    e = 0.08209443794970

    # Convert to radians / meters
    xy = np.array(xy) * 1000
    origin = np.array(origin) * np.pi / 180

    # Iterate to perform inverse projection
    M0=a*((1-e**2/4-3*e**4/64-5*e**6/256)*origin[1] - 
        (3*e**2/8+3*e**4/32+45*e**6/1024)*np.sin(2*origin[1]) + 
        (15*e**4/256 +45*e**6/1024)*np.sin(4*origin[1]) - 
        (35*e**6/3072)*np.sin(6*origin[1]))
    


    # z = xy[1, :] != -M0
    z = np.where(xy[1,:] != -M0) 

    A = (M0 + xy[1, z]) / a
    B = xy[0, z]**2 / a**2 + A**2

    llh = np.zeros_like(xy)
    llh[1, z] = A

    delta = np.inf
    c = 0
    delta_limit = 1e-8
    while np.max(np.abs(delta)) > delta_limit:
        C = np.sqrt((1 - e**2 * np.sin(llh[1, z])**2)) * np.tan(llh[1, z])

        M = a * ((1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * llh[1, z] -
                 (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * np.sin(2*llh[1, z]) +
                 (15*e**4/256 + 45*e**6/1024) * np.sin(4*llh[1, z]) -
                 (35*e**6/3072) * np.sin(6*llh[1, z]))

        Mn = 1 - e**2/4 - 3*e**4/64 - 5*e**6/256 - \
             -2 * (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * np.cos(2*llh[1, z]) + \
             4 * (15*e**4/256 + 45*e**6/1024) * np.cos(4*llh[1, z]) + \
             -6 * (35*e**6/3072) * np.cos(6*llh[1, z])

        Ma = M / a

        delta = -(A * (C * Ma + 1) - Ma - 0.5 * (Ma**2 + B) * C) / \
                (e**2 * np.sin(2*llh[1, z]) * (Ma**2 + B - 2*A * Ma) / (4*C) + (A - Ma) * (C * Mn - 2/np.sin(2*llh[1, z])) - Mn)

        llh[1, z] = llh[1, z] + delta

        c = c + 1
        if c > 100:
            delta_limit = 1e-7
            # raise ValueError('Convergence failure.')
        if c > 1000:
            delta_limit = 1e-6 
        if c > 1500:
            delta_limit = 1e-5
        if c > 2000: 
            raise ValueError('Convergence failure.')


    llh[0, z] = (np.arcsin(xy[0, z] * C / a) / np.sin(llh[1, z])) + origin[0]

    # Handle special case of latitude = 0
    z_zero_catch = np.where(xy[1,:] == -M0)
    # dlamba=llh[0,z_zero_catch]-origin[0]
    llh[0,z_zero_catch]= xy[0,z_zero_catch]/a + origin[0]
    llh[1,z_zero_catch]=0

    # Convert back to decimal degrees
    llh = llh * 180 / np.pi

    return llh



if __name__ == '__main__':
   llh = local2llh([[-8.78396],[2.06516]],[29.690,65.379])
  
   print(llh)