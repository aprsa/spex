#!/usr/bin/python

"""
This script fits the observed histogram of eccentricities and orbital
periods with the selection-corrected theoretical histogram based on the
Galaxia model.
"""

import argparse
import sys
import numpy as np
import numpy.random as rng
import scipy.stats as st
import phoebe
from phoebe import u
import libphoebe as lp
#~ import matplotlib as mpl
#~ import matplotlib.pyplot as plt
import operator
import ebf
import datetime as dt

from scipy.optimize import newton
import time

rng.seed(0)


def calculate_radius(mass, logg):
    """
    Compute radius from mass and logg. The constant is sqrt(GMSunNom)/RSunNom.
    """

    return 16.558988847673433*(mass/10**logg)**0.5

def calculate_sma(M1, M2, P):
    """
    Computes the semi-major axis of the binary (in solar radii).
    The masses M1 and M2 are in solar masses, the period is in days.
    """

    return 4.20661*((M1+M2)*P**2)**(1./3)

def requiv2pot(Requiv, q, F, delta, cplim=0.0):
    """
    @q:     mass ratio, M2/M1
    @F:     synchronicity parameter
    @delta: instantaneous separation
    @cplim: limiting critical potential

    Calculate surface potential from equivalent radius.
    """
    pot_guess = lp.roche_Omega(q, F, delta, np.array([0, 0, Requiv]))
    # print('q=%f, F=%f, d=%f, Req/a=%f, %f < %f?' % (q, F, delta, Requiv, pot_guess, cplim))
    if pot_guess < cplim:
        return -1
    pot = lp.roche_Omega_at_vol(4./3*np.pi*Requiv**3, q, F, delta, pot_guess)
    return pot

def conjunction_separation(a, e, w):
    """
    Calculates instantaneous separation at superior and inferior conjunctions.
    """

    dp = a*(1-e*e)/(1+e*np.sin(w))
    ds = a*(1-e*e)/(1-e*np.sin(w))
    return (dp, ds)

def dOmegadx (x, y, z, D, q, F):
    return -x*(x**2+y**2+z**2)**-1.5 - q*(x-D)*((x-D)**2+y**2+z**2)**-1.5 - q/D**2 + F**2*(1+q)*x

def d2Omegadx2 (x, y, z, D, q, F):
    return (2.*x**2-y**2-z**2)/(x**2+y**2+z**2)**2.5 + q*(2.*(x-D)**2-y**2-z**2)/((x-D)**2+y**2+z**2)**2.5 + F**2*(1+q)

def distsq(d1, d2):
    return (d1.ra-d2.ra)**2+(d1.dec-d2.dec)**2

def draw_ecc(P0, method='envelope', A=3.5, B=3.0, C=0.23, E=0.98):
    """
    Some form of eccentricity distribution derived from the Kepler EB sample.
    """

    if method == 'stupid':
        if P0 < 0.18:
            return 0.0
        elif P0 < 5:
            return min(st.expon.rvs(0, 0.05), 0.9)
        elif P0 < 15:
            return min(st.expon.rvs(0, 0.12), 0.9)
        else:
            return 0.9*rng.random()

    if method == 'envelope':
        # emax(P0) = E - A*exp(-(B*P0)**C)
        emax = E - A*np.exp(-(B*P0)**C)
        
        # dN/de(a, c) = uniform or exponentiated Weibull
        #             = a*c*[1-exp(-x*c)]**(a-1)*exp(-x*c)*x**(c-1)
        e = rng.random()
        if e < emax:
            return e
        else:
            return st.exponweib.rvs(3.6163625281792133, 0.42393548261761904, 0, 0.0016752233584087976)

def count_eccs(sample, Pdist, thresh=0.025, A=3.5, B=3.0, C=0.23, E=0.98):
    """
    Counts the number of EBs with eccentricities smaller than @thresh,
    between @thresh and the envelope, and above the envelope.
    
    @sample: a sample of (logP, ecc) pairs
    @Pdist = (Prange, Phist)
    """

    emax = E - A*np.exp(-(B*10**sample[0,:])**C)
    emax[emax < thresh] = thresh

    #~ plt.plot(sample[0,:], sample[1,:], 'bo')
    #~ plt.plot(sample[0,:], emax, 'r.')
    #~ plt.show()

    idx = np.digitize(sample[0,:], Pdist[0,:])
    
    return np.array([
        (len(sample[0,:][(idx == i) & (sample[1,:] < thresh)]), 
         len(sample[0,:][(idx == i) & (sample[1,:] >= thresh) & (sample[1,:] < emax)]),
         len(sample[0,:][(idx == i) & (sample[1,:] >= emax)]))
         for i in range(len(Pdist[0,:]))])

def draw_per0():
    """
    Draw argument of periastron from a uniform distribution:
    """
    return 2*np.pi*rng.random()

def draw_incl():
    """
    Draw the inclination from a uniform distribution in cos(i).
    """
    return np.arccos(rng.random())

def draw_cosi():
    """
    Draw cos(incl) from a uniform distribution.
    """
    return rng.random()

def draw_period(P0hist, P0ranges):
    """
    Draw the orbital period from the passed distribution.
    """
    idx = np.random.choice(range(len(P0hist)), p=P0hist)
    logP0 = P0ranges[idx] + (P0ranges[idx+1]-P0ranges[idx])*rng.random()

    return 10**logP0

def draw_from_distribution(dist):
    """
    Draw from the given discrete distribution histogram. The value will
    be continuous, drawn uniformly from the chosen histogram bin.
    """

    ranges, hist = dist
    binidx = np.random.choice(range(len(hist)), p=hist)
    return ranges[binidx] + rng.random()*(ranges[binidx+1]-ranges[binidx])

def draw_meanan():
    """
    Draw mean anomaly from a uniform distribution.
    """
    return 2*np.pi*rng.random()

def join_mags(mag1, mag2):
    """
    Add two magnitudes.
    """
    # m1-m2 = -5/2 log(f1/f2)
    # f1/f0 = 10**[-0.4(m1-m0)]

    return 14.0-2.5*np.log10(10**(-0.4*(mag1-14.0))+10**(-0.4*(mag2-14.0)))

def qdist_raghavan():
    # Raghavan et al. (2010):
    qrange = np.linspace(0, 1, 21)
    qhist = np.linspace(0.05, 0.05, 20)
    qhist[ 0] = 0.000   # 0.00-0.05
    qhist[ 1] = 0.005   # 0.05-0.10
    qhist[ 2] = 0.005   # 0.10-0.15
    qhist[ 3] = 0.030   # 0.15-0.20
    qhist[19] = 0.100   # 0.95-1.00
    qhist = qhist/qhist.sum()
    return (qrange, qhist)

def mdist_raghavan():
    # Raghavan et al. (2010): $56\% \pm 2\%$ single, $33\% \pm 2\%$ binary, $8\% \pm 1\%$ triple systems and $3\% \pm 1\%$ multis.
    return {'S': 0.56, 'B': 0.33, 'T': 0.08, 'M': 0.03}

def join_Teffs(Teffs, Rs):
    return (np.sum(Rs**2*Teffs**4)/np.sum(Rs**2))**0.25

def join_loggs(loggs, Mbols):
    Ls = 10**(Mbols)
    return np.sum(Ls*loggs)/np.sum(Ls)

def fl2mag(fl, mzero=20):
    return mzero-5./2*np.log10(fl)

######################################################################################

def generate_header(out):
    link = 'https://ui.adsabs.harvard.edu/#abs/2016ApJS..227...29P'

    out.write('SURVEY: LSST\n')
    out.write('FILTERS: ugrizY\n')
    out.write('MODEL: PHOEBE\n')
    out.write('MODEL_PARNAMES: PERIOD,TRAT,RSUM,ECOSW,ESINW,COSI,Q,FF\n')
    # out.write('NEVENT: N/A\n')
    out.write('RECUR_CLASS: RECUR-PERIODIC\n')
    out.write('\n')
    out.write('COMMENT: Created on %s by Andrej Prsa\n' % (dt.date.today()))
    out.write('COMMENT: PHOEBE stands for PHysics Of Eclipsing BinariEs\n')
    out.write('COMMENT: See %s\n' % (link))
    out.write('COMMENT: PERIOD is the orbital period, in days\n')
    out.write('COMMENT: TRAT   is the temperature ratio, T2/T1\n')
    out.write('COMMENT: RSUM   is the sum of fractional radii, (R1+R2)/a\n')
    out.write('COMMENT: ECOSW  is tangential eccentricity, e*cos(w)\n')
    out.write('COMMENT: ESINW  is radial eccentricity, e*sin(w)\n')
    out.write('COMMENT: COSI   is the cosine of inclination\n')
    out.write('COMMENT: Q      is the mass ratio, M2/M1\n')
    out.write('COMMENT: FF     is the fillout factor, (pot-potL1)/(potL2-potL1)\n')
    out.write('\n')

def generate_event(out, data, pars, glon=0.0, glat=0.0):
    out.write('START_EVENT: 1\n')
    out.write('NROW: %d GLON: %f GLAT: %f\n' % (len(data), glon, glat))
    out.write('PARVAL: %f,%f,%f,%f,%f,%f,%f,%f\n' % (pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7]))
    # out.write('ANGLEMATCH: 10.0 # optional\n')
    for d in data:
        out.write('S: %7.4f %.3f %.3f %.3f %.3f %.3f %.3f\n' % (d[0], d[1], d[2], d[3], d[4], d[5], d[6]))
    out.write('END_EVENT: 1\n')


######################################################################################


class Table:
    def __init__(self, tablename, maxlines=0, fov=None, DEBUG=False):
        """
        This class stores the entire galaxy model table.
        
        @tablename: Galaxia ebf output file.
        @maxlines: maximum number of lines to read in from the file.
        @fov: if not None, all stars outside the fov will be dropped.
        """

        #~ self.data = []
        
        table = ebf.read(tablename)
        self.Nstars = len(table['mact']) # these are stellar masses
        self.idx = np.arange(self.Nstars)
        self.data = table

        #~ if fov is not None:
            #~ if not fov.within_box(ra, dec):
                #~ continue
            #~ if not fov.within_outline(ra, dec):
                #~ continue
            #~ if not fov.on_silicon(ra, dec):
                #~ continue

        #~ self.glon = table['glon']
        #~ self.glat = table['glat']
        #~ self.age = table['age']
        #~ self.mass = table['mact']


        #~ for s in zip(table['sdss_r'], table['teff'], table['grav'], table['age'], table['mact'], table['feh'], table['glon'], table['glat'], table['exbv_solar'], table['lum'], np.sqrt(table['px']**2+table['py']**2+table['pz']**2)):
            #~ self.data.append(Star(s))
    
    #~ def sort_by_mass(self):
        #~ self.data.sort(key=operator.attrgetter('mact'))

    def filter(self, masses=None, ages=None, glons=None, glats=None):
        """
        Returns the array mask that corresponds to the passed criteria.
        """
        
        if (masses == None) or (ages == None) or (glons == None) or (glats == None):
            return self.idx
        
        sel = ((self.data['mact'] >= masses[0]) & (self.data['mact'] <= masses[1]) &
               (self.data['age']  >= ages[0])   & (self.data['age']  <= ages[1])   & 
               (self.data['glon'] >= glons[0])  & (self.data['glon'] <= glons[1])  &
               (self.data['glat'] >= glats[0])  & (self.data['glat'] <= glats[1]))
               
        return self.idx[sel]

class Star:
    """
    This class stores a single entry from the Besancon table.
    """
    
    def __init__(self, c):
        self.type     = 1

        self.Rmag     = float(c[0])
        self.Teff     = float(c[1])
        self.logg     = float(c[2])
        self.age      = float(c[3]) # log(age[yr])
        self.mass     = float(c[4])
        self.met      = float(c[5])
        self.glon     = float(c[6])
        self.glat     = float(c[7])
        self.redden   = 0.53*float(c[8]) # see galaxia manual for 0.53
        self.lum      = float(c[9]) # log10(L/LSun)
        self.dist     = float(c[10])
        
        self.radius   = calculate_radius(self.mass, self.logg)
        self.Mbol     = -2.5*self.lum+4.85
        
        #~ self.steltype = int(c[10])
        #~ self.absmagV  = float(c[9])
        #~ self.lumclass = int(c[10])
        

    def __repr__(self):
        return "Rc=%f  T=%f  lg=%f  age=%f  M=%f  Mbol=%f  R=%f  MH=%f  lon=%f  lat=%f  D=%f  Av=%f" % (self.Rmag, self.Teff, self.logg, self.age, self.mass, self.Mbol, self.radius, self.met, self.glon, self.glat, self.dist, self.redden)

class Single:
    """
    Class attributes:
    
    @type:     number of stars -- always 1
    @glon:     galactic longitude
    @glat:     galactic latitude
    @age:      age
    @mag:      apparent magnitude of the system
    @distance: distance to the system
    @absmagV:  absolute magnitude in Johnson V band
    @radius:   stellar radius
    @Teff:     effective temperature
    @logg:     effective surface gravity
    """

    def __init__(self, table, age=None, DEBUG=False):
        self.type = 1

        # Draw a random star, constraining the age if requested:
        while True:
            i = int(rng.random()*len(table.data))
            if age != None and table.data[i].age != age:
                continue
            break

        self.glon     = table.data[i].glon
        self.glat     = table.data[i].glat
        self.age      = table.data[i].age
        self.mag      = table.data[i].Rmag
        self.distance = table.data[i].dist
        self.period   = 1e-7
        self.EB       = False
        self.SEB      = False
        self.radius   = table.data[i].radius
        self.Teff     = table.data[i].Teff
        self.logg     = table.data[i].logg

class Binary:
    """
    Class attributes:
    
    @type:     number of stars -- always 2
    @period:   orbital period in days (either passed or drawn)
    @ecc:      orbital eccentricity (drawn according to the period)
    @cosi:     cosine of orbital inclination (drawn)
    @per0:     argument of periastron (drawn)
    @meanan:   mean anomaly (drawn)
    @physical: does the system pass sanity check (True/False)
    @sma:      semi-major axis (computed)
    @supsep:   superior conjunction separation (computed)
    @infsep:   inferior conjunction separation (computed)
    @q:        mass ratio (computed)
    @r1:       fractional primary star radius (computed)
    @r2:       fractional secondary star radius (computed)
    @teff1:    effective temperature of the primary component (drawn)
    @teff2:    effective temperature of the secondary component (drawn)
    @pot1:     primary star surface potential (computed)
    @pot2:     secondary star surface potential (computed)
    @F1:       synchronicity parameter -- always set to 1
    @F2:       synchronicity parameter -- always set to 1
    @FF:       fillout factor: (pot-potL1)/(potL2-potL1)
    @mag:      apparent magnitude of the system (computed)
    @absmagV   absolute magnitude of the system (computed)
    @distance: distance to the system (via distance to the primary star)
    @EB:       eclipsing binary flag (True/False)
    @SEB:      singly eclipsing binary flag (True/False)
    @glon:     right ascension
    @glat:     declination
    @age:      age
    @Teff:     effective temperature of the binary
    @logg:     effective surface gravity
    """

    def __init__(self, table, period=None, q=None, age=None, Pdist=None, qdist=None, eccpars=None, check_sanity=True, safety_limit=1000, DEBUG=False):
        self.type = 2

        if Pdist is not None:
            P0ranges, P0hist = Pdist
            # Round the last bin so that the integral is exactly 1 (needed for choice):
            P0hist[-1] = 1-P0hist[:-1].sum()

        self.period = period if period != None else draw_period(P0hist, P0ranges)
        self.q = q if q != None else draw_from_distribution(qdist)
        
        if eccpars is not None:
            A, B, C, E = eccpars
            self.ecc = draw_ecc(self.period, A=A, B=B, C=C, E=E)
        else:
            self.ecc = draw_ecc(self.period)

        self.per0    = draw_per0()
        self.cosi    = draw_cosi()
        self.meanan  = draw_meanan()            # mean anomaly
        self.F1      = 1.0
        self.F2      = 1.0

        # A PHOEBE bundle placeholder
        self.b = None

        self.physical = False
        safety_counter = 0

        # The first while-loop check whether the drawn binary is physical.
        while True:
            attempt = 0

            # The second while-loop picks a random pair of stars that
            # are coeval, have the prescribed mass ratio within the
            # 5% tolerance and are within 1 arcsec^2.
            
            while True:
                # Draw a primary star randomly, possibly constraining
                # its age:
                while True:
                    # Yep, this might be ugly, but it's actually faster
                    # than filtering the table (as we do below).
                    i = int(rng.random()*table.Nstars)
                    if age != None and table.data['age'][i] >= age-0.25 and table.data['age'][i] <= age+0.25:
                        continue
                    break
                                
                # Create a pool for allowed secondaries:
                masses = (0.95*self.q*table.data['mact'][i], 1.05*self.q*table.data['mact'][i])
                ages = (table.data['age'][i]-0.25, table.data['age'][i]+0.25)
                glons = (table.data['glon'][i]-0.5, table.data['glon'][i]+0.5)
                glats = (table.data['glat'][i]-0.5, table.data['glat'][i]+0.5)
                poolidx = table.filter(masses, ages, glons, glats)
                
                # If secondary candidates are found, break out:
                if len(poolidx) > 0:
                    break
                
                # Otherwise keep an eye on it, it might be an
                # implausible mass ratio.
                attempt += 1
                if attempt > safety_limit:
                    # If q is passed, bail.
                    if q != None:
                        print('Mass ratio %f cannot be created from the Galaxia stars. Bailing out.' % q)
                        exit()
                    
                    # The drawn mass ratio is too extreme; pick a new one.
                    if DEBUG:
                        print('# Requested q: %f; max attempts reached; drawing another q.' % (self.q))
                    self.q = draw_from_distribution(qdist)
                    attempt = 0
            
            # The pool contains only candidate secondaries, so we draw
            # it from that pool randomly:
            # j = int(rng.random()*len(pool))
            j = int(rng.random()*len(poolidx))

            if DEBUG:
                print('# Requested q: %f; drawn q: %f; percent diff: % 2.2f%%; distance: %2.2f arcsec' % (self.q, pool[j].mass/table.data[i].mass, (self.q-pool[j].mass/table.data[i].mass)/self.q*100, ((table.data[i].ra-pool[j].ra)**2+(table.data[i].dec-pool[j].dec)**2)**0.5))

            pidx, sidx = i, poolidx[j] # primary and secondary star index

            # Check: does either of the stars overflow L2, and
            # do the stars fit into the binary:
            self.sma = calculate_sma(table.data['mact'][pidx], table.data['mact'][sidx], self.period)
            if safety_counter < safety_limit and self.sma*(1-self.ecc) < table.data['radius'][pidx]+table.data['radius'][sidx]:
                safety_counter += 1
                continue
            elif safety_counter == safety_limit:
                if period != None:
                    print('Period %f cannot be created from the given table. Bailing out.' % period)
                    exit()
                if DEBUG:
                    print('# Max attempts reached for P=%f, e=%f; drawing again.' % (self.period, self.ecc))
                self.period = draw_period(P0hist, P0ranges)
                self.ecc    = draw_ecc(self.period)
                safety_counter = 0
                continue
            
            # print('q=%f, 1/q=%f, F=%f, delta=%f' % (self.q, 1./self.q, self.F1, 1-self.ecc))
            crp1 = lp.roche_critical_potential(self.q, self.F1, 1-self.ecc)
            crp2 = lp.roche_critical_potential(1./self.q, self.F2, 1-self.ecc)
            self.pot1 = requiv2pot(table.data['radius'][pidx]/self.sma, self.q, self.F1, 1-self.ecc, crp1['L1'])
            self.pot2 = self.q*requiv2pot(table.data['radius'][sidx]/self.sma, 1./self.q, self.F2, 1-self.ecc, crp2['L1']) + 0.5*(1-self.q)
            cp1, cp2 = crp1['L1'], crp1['L2']

            # FIXME: we ignore contacts here because of the added complexity.
            # if safety_counter < safety_limit and (self.pot1 < cp2 or self.pot2 < cp2):
            if safety_counter < safety_limit and (self.pot1 < cp1 or self.pot2 < cp1):
                safety_counter += 1
                continue
            elif safety_counter == safety_limit:
                # No star from the table will fit into this binary.
                if period != None:
                    print('Period %f cannot be created from the given table. Bailing out.' % period)
                    exit()
                
                # The drawn period/eccentricity combination is too extreme. Pick a new one.
                if DEBUG:
                    print('# Max attempts reached for P=%f, e=%f; drawing again.' % (self.period, self.ecc))
                self.period = draw_period(P0hist, P0ranges)
                self.ecc    = draw_ecc(self.period)
                safety_counter = 0
                continue

            # Checks survived!
            break

        self.r1 = table.data['radius'][pidx]/self.sma
        self.r2 = table.data['radius'][sidx]/self.sma

        # contact systems:
        if self.pot1 < cp1:
            self.pot2 = self.pot1
        elif self.pot2 < cp1:
            self.pot1 = self.pot2

        # Fillout factor, has to come below the block above.
        self.FF = (self.pot1-cp1)/(cp2-cp1)

        # Compute instantaneous separation at both conjunctions:
        self.supsep, self.infsep = conjunction_separation(self.sma, self.ecc, self.per0)
        
        # We assume the distance to the binary is the distance to the first drawn star:
        self.distance = table.data['distance'][pidx]
        
        # The same with coordinates and age:
        self.glon = table.data['glon'][pidx]
        self.glat = table.data['glat'][pidx]
        self.age  = table.data['age'][pidx]

        # When joining magnitudes, we transform the original magnitude
        # of the secondary to the distance of the primary, i.e.:
        #
        # m_new = m_old + 5*log10(D_new/D_old)
        
        self.mag = join_mags(table.data['mag_sdss_r'][pidx], table.data['mag_sdss_r'][sidx] + 5*np.log10(self.distance/table.data['distance'][sidx]))
        self.lum = join_mags(table.data['lum'][pidx], table.data['lum'][sidx])
        
        # Effective temperature of the components:
        self.teff1 = 10**table.data['teff'][pidx]
        self.teff2 = 10**table.data['teff'][sidx]

        # Effective temperature and logg of the binary:
        self.Teff = np.log10(join_Teffs(np.array((10**table.data['teff'][pidx], 10**table.data['teff'][sidx])), np.array((table.data['radius'][pidx], table.data['radius'][sidx]))))
        self.logg = join_loggs(np.array((table.data['grav'][pidx], table.data['grav'][sidx])), np.array((table.data['lum'][pidx], table.data['lum'][sidx])))
        
        # We need to check if we have an eclipse at superior and/or at inferior conjunction:
        supEB = (table.data['radius'][pidx]+table.data['radius'][sidx] > abs(self.supsep*self.cosi))
        infEB = (table.data['radius'][pidx]+table.data['radius'][sidx] > abs(self.infsep*self.cosi))
        self.EB = supEB or infEB
        
        # Do we have just a single eclipse?
        self.SEB = (supEB != infEB)

    def compute_lc(self, filename):
        if self.b == None:
            self.times = np.linspace(0, self.period, 201)
            self.b = phoebe.default_binary()
            self.b.flip_constraint('pot@primary', solve_for='rpole')
            self.b.flip_constraint('pot@secondary', solve_for='rpole')

            self.b.add_dataset('lc', dataset='u', times=self.times, passband='LSST:u')
            self.b.add_dataset('lc', dataset='g', times=self.times, passband='LSST:g')
            self.b.add_dataset('lc', dataset='r', times=self.times, passband='LSST:r')
            self.b.add_dataset('lc', dataset='i', times=self.times, passband='LSST:i')
            self.b.add_dataset('lc', dataset='z', times=self.times, passband='LSST:z')
            self.b.add_dataset('lc', dataset='y', times=self.times, passband='LSST:y')

            self.b.add_dataset('mesh', columns=['abs_normal_intensities@*', 'areas', 'volume', 'ldint@*', 'ptfarea@*'])

            self.b.set_value_all('atm',       'blackbody')
            self.b.set_value_all('ld_func',   'linear')
            self.b.set_value_all('ld_coeffs', [0.5])

        self.b['t0@system'] = self.meanan
        self.b['period@orbit'] = self.period

        self.b['sma@orbit'] = self.sma
        self.b['incl@orbit'] = (np.arccos(self.cosi),'rad')
        self.b['q@orbit'] = self.q
        self.b['ecc@orbit'] = self.ecc
        self.b['per0@orbit'] = self.per0
        
        self.b['teff@primary'] = self.teff1
        self.b['teff@secondary'] = self.teff2
        self.b['pot@primary'] = self.pot1
        self.b['pot@secondary'] = self.pot2

        self.b.run_compute(irrad_method='none', model='myrun')

        # here's the tricky part: we need coupled pblums across the passbands. To do that,
        # we compute luminosities in absolute units and then compute the scaling manually
        # as phoebe currently does not support that option.

        lum = np.array([np.sum( self.b['value@abs_normal_intensities@primary@%s' % x]*self.b['areas@primary'].get_value(unit=u.m**2)*self.b['value@ldint@primary@%s' % x])*self.b['value@ptfarea@primary@%s' % x]*np.pi for x in ['u', 'g', 'r', 'i', 'z', 'y']])
        for pbi, pb in enumerate(['u', 'g', 'r', 'i', 'z', 'y']):
            self.b['value@fluxes@model@%s' % pb] *= lum[pbi]/lum[1]

        mzero = rng.uniform(17, 23)
        data = np.stack((self.times, fl2mag(self.b['value@fluxes@u@model'], mzero), fl2mag(self.b['value@fluxes@g@model'], mzero), fl2mag(self.b['value@fluxes@r@model'], mzero), fl2mag(self.b['value@fluxes@i@model'], mzero), fl2mag(self.b['value@fluxes@z@model'], mzero), fl2mag(self.b['value@fluxes@y@model'], mzero)), axis=-1)
        pars = [self.period, self.teff2/self.teff1, self.r1+self.r2, self.ecc*np.cos(self.per0), self.ecc*np.sin(self.per0), self.cosi, self.q, self.FF]

        with open(filename, 'w') as mylc:
            generate_header(mylc)
            generate_event(mylc, data, pars, self.glon, self.glat)

    def __repr__(self):
        return "%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %11.5f %5.1f  %d  %d" % (self.glon, self.glat, self.period, self.ecc, self.per0, self.cosi, self.sma, self.mag, self.distance, self.Teff, self.age, self.EB, self.SEB)

class Triple:
    """
    Class attributes:
    
    @type:     number of stars -- always 3
    @ra:       right ascension
    @dec:      declination
    @age:      age
    @distance: distance to the system
    @mag:      apparent magnitude of the system
    @period:   period of the binary
    @EB:       whether the binary is eclipsing
    @SEB:      whether the binary is singly eclipsing
    @ecc:      eccentricity of the binary
    @Teff:     effective temperature of the triple
    @logg:     effective surface gravity
    """

    def __init__(self, table, period=None, q=None, age=None, Pdist=None, qdist=None, eccpars=None, check_sanity=True, safety_limit=1000, DEBUG=False):
        self.type = 3

        # We generate a triple by generating a single star and a binary.
        # We constrain the age of the single star to the age of the binary.
        
        binary = Binary(table, period, q, age, Pdist, qdist, eccpars, check_sanity, safety_limit, DEBUG)
        single = Single(table, age=binary.age, DEBUG=DEBUG)

        self.ra       = binary.ra
        self.dec      = binary.dec
        self.age      = binary.age
        self.distance = binary.distance
        self.mag      = join_mags(binary.mag, single.absmagV+5.*np.log10(self.distance*100.))
        self.period   = binary.period
        self.EB       = binary.EB
        self.SEB      = binary.SEB
        self.ecc      = binary.ecc

        # Effective temperature and logg of the triple:
        self.Teff = join_Teffs(np.array((binary.primary.Teff, binary.secondary.Teff, single.Teff)), np.array((binary.primary.radius, binary.secondary.radius, single.radius)))
        self.logg = join_loggs(np.array((binary.primary.logg, binary.secondary.logg)), np.array((binary.primary.Mbol, binary.secondary.Mbol)))

class Multiple:
    """
    Class attributes:
    
    @type:     number of stars -- always 4
    @ra:       right ascension
    @dec:      declination
    @age:      age
    @mag:      apparent magnitude of the system
    @distance: distance to the system
    @Teff:     effective temperature of the multiple
    @logg:     effective surface gravity
    """

    def __init__(self, table, period=None, q=None, age=None, Pdist=None, qdist=None, eccpars=None, check_sanity=True, safety_limit=1000, DEBUG=False):
        self.type = 4

        # We generate a multiple by generating two binary stars (i.e. a
        # hierarchical quadruple). We constrain the ages of the binaries
        # to be the same. We should probably constrain ra and dec as
        # well, and store /both/ periods, but that would change the logic
        # of the code significantly for minimal practical benefit.
        
        b1 = Binary(table, period, q, age, Pdist, qdist, eccpars, check_sanity, safety_limit, DEBUG)
        b2 = Binary(table, period, q, b1.age, Pdist, qdist, eccpars, check_sanity, safety_limit, DEBUG)

        self.ra       = b1.ra
        self.dec      = b1.dec
        self.age      = b1.age
        self.distance = b1.distance
        self.mag      = join_mags(b1.mag, b2.absmagV+5.*np.log10(self.distance*100.))
        self.period   = max(b1.period, b2.period)
        self.EB       = b1.EB or b2.EB
        self.SEB      = b1.SEB or b2.SEB
        self.ecc      = b1.ecc if b1.EB else b2.ecc

        # Effective temperature and logg of the triple:
        self.Teff = join_Teffs(np.array((b1.primary.Teff, b1.secondary.Teff, b2.primary.Teff, b2.secondary.Teff)), np.array((b1.primary.radius, b1.secondary.radius, b2.primary.radius, b2.secondary.radius)))
        self.logg = join_loggs(np.array((b1.primary.logg, b1.secondary.logg, b2.primary.logg, b2.secondary.logg)), np.array((b1.primary.Mbol, b1.secondary.Mbol, b2.primary.Mbol, b2.secondary.Mbol)))

class KepFOV:
    def __init__(self):
        # Kepler FOV parameters:
        self.fov = np.loadtxt('kepfov.data').reshape((84, 4, 2))

        self.outline = np.array([
            self.fov[55][0], self.fov[14][3], self.fov[19][0], self.fov[3][0],
            self.fov[10][3], self.fov[27][0], self.fov[31][0], self.fov[70][3],
            self.fov[67][0], self.fov[83][0], self.fov[74][3], self.fov[59][0]])

        self.n = len(self.outline)

        #~ if not simplified:
            #~ import K2fov.fov as fov
            #~ from K2fov.K2onSilicon import angSepVincenty as sphere_dist
            #~ fra, fdec, froll = 290.6688, 44.4952, 303
            #~ froll = fov.getFovAngleFromSpacecraftRoll(froll)
            #~ self.fov = fov.KeplerFov(fra, fdec, froll)

    def within_box(self, ra, dec):
        return (279.60813 < ra < 301.85564) and (36.523277 < dec < 52.481925)

    def within_outline(self, ra, dec):
        inside = False

        p1x, p1y = self.outline[0]
        for i in range(self.n+1):
            p2x, p2y = self.outline[i % self.n]
            if dec > min(p1y, p2y):
                if dec <= max(p1y, p2y):
                    if ra <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (dec-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or ra <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def on_silicon(self, ra, dec):
        inside = False

        for ccd in range(84):
            p1x, p1y = self.fov[ccd][0]
            for i in range(5):
                p2x, p2y = self.fov[ccd][i % 4]
                if dec > min(p1y, p2y):
                    if dec <= max(p1y, p2y):
                        if ra <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (dec-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                            if p1x == p2x or ra <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            if inside:
                return True

        return False

    #~ def on_silicon_detailed(self, ra, dec):
        #~ """
        #~ The original RA, Dec, and roll (all degrees) of Kepler was:
#~ 
        #~ 290.6688, 44.4952, 110
#~ 
        #~ The pointing moved slightly (at the arcsec level) throughout
        #~ the mission. The roll is the angle that gets the spacecraft Y &
        #~ Z axis pointed at the Kepler FOV. There is an additional hidden
        #~ 13-deg roll that is built into the photometer corresponding to
        #~ the roll angle of the focal plane wrt the spacecraft axes and a
        #~ 180-deg due to the reflection in the spherical mirror. So, I
        #~ think the total roll angle that you want to use for comparison
        #~ with the K2 roll numbers specified in the Campaign descriptions
        #~ is:  110 + 13 + 180 = 303-deg (and don't forget about the
        #~ seasonal 90-deg rolls that get added in. The summer (season 0)
        #~ is the one that corresponds to this roll angle.
        #~ """
#~ 
        #~ try:
            #~ dist = sphere_dist(self.fov.ra0_deg, self.fov.dec0_deg, ra, dec)
            #~ if dist >= 90.:
                #~ return False
            #~ ch = self.fov.pickAChannel(ra, dec)
            #~ ch, col, row = self.fov.getChannelColRow(ra, dec)
        #~ 
            #~ return True
        #~ except ValueError:
            #~ return False


class Observation:
    def __init__(self, targets):
        # The passed field is a list of stars generated from the Besancon
        # model. This class applies instrumental selection effects.
        self.targets = targets
        
        # Read in target selection ratios. These are the ratios between
        # Batalha et al. (2010)'s Table 2 and the equivalent table
        # produced by the forward model. The fractions tell us the
        # suppression factor as a function of mag, Teff, logg to get
        # the equivalent selection of Kepler targets.
        self.fractions = np.loadtxt('fractions.tab')[:,1:]
        self.fractions[self.fractions > 1.0] = 1.0
        
        # Duty cycle correction takes a long time to compute, so instead
        # of computing it here, we just read in the results of the
        # standalone computation.
        self.dc_p0, self.dc_prob = np.loadtxt('duty_cycle.data', unpack=True)
        self.dc_prob /= self.dc_prob[0]
        
        # SNR correction is a bit more manageable, so we do it here, on
        # the fly.
        period, pdepth, sigma = np.loadtxt("kepEBs.csv", delimiter=",", unpack=True, usecols=(1, 3, 7))
        logPobs = np.log10(period)
        
        # First let's take all well sampled SNRs (say, logP < 1) and
        # figure out what the "ground" distribution looks like.
        SNR_flat = pdepth[(pdepth > 0) & (sigma > 0) & (logPobs < 1)]/sigma[(pdepth > 0) & (sigma > 0) & (logPobs < 1)]
        SNR_hist, SNR_range = np.histogram(SNR_flat, bins=100)
        
        # Next let's compute the distributions of all SNRs per logP bin:
        logP = np.linspace(-1, 3, 100)
        SNR_per_bin = [pdepth[(pdepth > 0) & (sigma > 0) & (logPobs >= logP[i]) & (logPobs < logP[i+1])]/sigma[(pdepth > 0) & (sigma > 0) & (logPobs >= logP[i]) & (logPobs < logP[i+1])] for i in range(len(logP)-1)]
        
        # Now let's fit a straight line to the minimum log(SNR) on the
        # 1 < logP < 3 range.
        SNR_min = np.array([SNR.min() for SNR in SNR_per_bin])
        SNR_min_for_fit = np.log10(SNR_min[logP[:-1] > 1])
        SNR_logP_for_fit = logP[logP[:-1] > 1]
        p, v = np.polyfit(SNR_logP_for_fit, SNR_min_for_fit, 1, cov=True)
        
        # This line is what determines what part of the original
        # S/N population we lose because of the increased minimum S/N.
        self.snr_baseline = SNR_flat
        self.snr_coeffs = p

    def eta_dc(self, period):
        """
        For the passed period in days, return the probability that we
        detect at least two eclipses due to duty cycle.
        """

        return np.interp(period, self.dc_p0, self.dc_prob)

    def eta_snr(self, period):
        """
        For the passed period in days, return the probability that we
        detect at least two eclipses due to signal-to-noise ratio.
        """

        snrmin = 10**(self.snr_coeffs[0]*np.log10(period) + self.snr_coeffs[1])
        if not hasattr(snrmin, '__len__'):
            return float(len(self.snr_baseline[self.snr_baseline > snrmin]))/len(self.snr_baseline)
        else:
            return np.array( [float(len(self.snr_baseline[self.snr_baseline > x]))/len(self.snr_baseline) for x in snrmin] )

    def selected(self, target, DEBUG=False):
        """
        For the passed mag, teff and logg, return the probability that
        the object will be on the target list.
        """

        if target.mag > 16: return False
        
        col = min(max(int((11000.-target.Teff)/1000), 0), 8)
        if target.logg < 3.5:
            col += 8
        row = min(max(int(target.mag-6.0), 0), 10)
        if (row == 10) or (col == 16) or (col == 8 and target.logg >= 3.5):
            return False
        
        if DEBUG:
            print('# %6.0f %8.2f %7.2f %4d %4d %6.3f' % (target.Teff, target.mag, target.logg, col, row, self.fractions[row,col]))
        
        return rng.random() < self.fractions[row,col]

        # 383 EBs were known before Kepler's first light:
        # - 59 from Simbad
        # - 127 from ASAS
        # - 7 from HET
        # - 190 from Vulcan
        
    def observe(self, fov=None):
        for target in self.targets:
            # If on-silicon test is requested, perform it:
            if fov is not None:
                target.on_silicon = False
                if fov.within_box(ra, dec):
                    if fov.within_outline(ra, dec):
                        if fov.on_silicon(ra, dec):
                            target.on_silicon = True
                if target.on_silicon == False:
                    target.is_target = False
                    target.detected = False
                    continue
            else:
                target.on_silicon = True

            # Is the target on the target list:
            target.is_target = self.selected(target)

            # Given the period of the target, compute the detection probability:
            eta = self.eta_dc(target.period) * self.eta_snr(target.period)
            
            # Roll a dice to see if that target is going to be observed:
            prob = rng.random()
            
            if prob <= eta:
                target.detected = True
            else:
                target.detected = False

    def observe_one(self, target, fov=None):
        if fov is not None:
            target.on_silicon = False
            if fov.within_box(ra, dec):
                if fov.within_outline(ra, dec):
                    if fov.on_silicon(ra, dec):
                        target.on_silicon = True
            if target.on_silicon == False:
                target.is_target = False
                target.detected = False
                return
        else:
            target.on_silicon = True

        # Is the target on the target list:
        target.is_target = self.selected(target)

        # Given the period of the target, compute the detection probability:
        eta = self.eta_dc(target.period) * self.eta_snr(target.period)
        
        # Roll a dice to see if that target is going to be observed:
        prob = rng.random()
        
        if prob <= eta:
            target.detected = True
        else:
            target.detected = False

def simulate_field(table, argv, mdist, Pdist, qdist, eccpars, DEBUG=True):
    """
    Pdist = (P0ranges, P0hist)
    """
    field = []
    if DEBUG:
        print('# requested sample size: %d' % argv.sample_size)
    
    if argv.maxEBs != 0:
        if DEBUG:
            print('# maximum number of EBs set at: %d' % (argv.maxEBs))
        
        Snum, Bnum, Tnum, Mnum, EBnum = 0, 0, 0, 0, 0
        
        # We need to observe this as we create them so that we know
        # how many EBs we have created.
        run = Observation(None)

        while EBnum != argv.maxEBs:
            # Let's roll a dice to see which type of system we will
            # create. We need to do this so that we can count the number
            # of EBs that we create.
            roll = rng.random()
            if roll < mdist['S']:
                field.append(Single(table))
                Snum += 1
            elif roll < mdist['S'] + mdist['B']:
                field.append(Binary(table, Pdist=Pdist, qdist=qdist, eccpars=eccpars, check_sanity=True, safety_limit=100))
                Bnum += 1
            elif roll < mdist['S'] + mdist['B'] + mdist['T']:
                field.append(Triple(table, Pdist=Pdist, qdist=qdist, eccpars=eccpars, check_sanity=True, safety_limit=100))
                Tnum += 1
            else:
                field.append(Multiple(table, Pdist=Pdist, qdist=qdist, eccpars=eccpars, check_sanity=True, safety_limit=100))
                Mnum += 1

            # Observe this target:
            run.observe_one(field[-1])

            # Is it an eclipsing binary?
            if field[-1].EB and field[-1].on_silicon and field[-1].is_target:
                EBnum += 1

        if DEBUG:
            print('# %d single stars created.' % (Snum))
            print('# %d binaries created.' % (Bnum))
            print('# %d triples created.' % (Tnum))
            print('# %d multiples created.' % (Mnum))
    else:
        Bnum = int(mdist['B']*argv.sample_size)
        Tnum = int(mdist['T']*argv.sample_size)
        Mnum = int(mdist['M']*argv.sample_size)
        Snum = argv.sample_size-Bnum-Tnum-Mnum

        # Generate single stars:
        for i in range(Snum):
            field.append(Single(table))
        print('# %d single stars created.' % (Snum))
        
        # Generate binary stars:
        for i in range(Bnum):
            field.append(Binary(table, Pdist=Pdist, qdist=qdist, check_sanity=True, safety_limit=100))
        print('# %d binaries created.' % (Bnum))
        
        # Generate triple stars:
        for i in range(Tnum):
            field.append(Triple(table, Pdist=Pdist, qdist=qdist, check_sanity=True, safety_limit=100))
        print('# %d triples created.' % (Tnum))
        
        # Generate multiple stars:
        for i in range(Mnum):
            field.append(Multiple(table, Pdist=Pdist, qdist=qdist, check_sanity=True, safety_limit=100))
        print('# %d multiples created.' % (Mnum))
    
        run = Observation(field)
        strt = time.time()
        run.observe()
        print('# Observed in %.3fs' % (time.time() - strt))

    return field

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='spex -- stellar population explorer.')
    parser.add_argument('-o', '--observe',      action='store_true',  help='generate a synthetic data-set of the Kepler field\
                                                                            (default: False)')
    parser.add_argument('-c', '--count',        action='store_true',  help='use a previously determined underlying period\
                                                                            distribution to count SEBs and EBs (default: False)')
    parser.add_argument('-s', '--solve',        action='store_true',  help='run the forward model and compare to the observed\
                                                                            period and eccentricity distributions (default: False)')
    parser.add_argument('-t', '--table',        metavar='file',       help='filename of the galaxy model table (default: gtable12.ebf)',   type=str,   default='gtable12.ebf')
    # parser.add_argument('-b', '--bins',         metavar='num',        help='number of histogram bins (default: 20)',                   type=int,   default=20)
    # parser.add_argument('-x', '--xi',           metavar='val',        help='step size for differential corrections (default: 0.05)',   type=float, default=0.05)
    parser.add_argument('-q', '--qdist',        metavar='dist',       help='underlying mass ratio distribution (default: raghavan)',   type=str,   choices=['single', 'raghavan'], default='raghavan')
    parser.add_argument('-m', '--mdist',        metavar='mdist',      help='underlying multiplicity distribution (default: raghavan)', type=str,   choices=['raghavan'], default='raghavan')
    # parser.add_argument('-P', '--Pdist',        metavar='pdist',      help='underlying period distribution (default: uniform)',  type=str,   choices=['uniform', 'ulogP'], default='uniform')
    parser.add_argument(      '--lpexcess',     metavar='val',        help='fraction of long period EBs (default: 0.65)',              type=float, default=0.65)
    parser.add_argument(      '--lpbin',        action='store_true',  help='include a bin for long period EBs (default: False)')
    parser.add_argument(      '--ulogP',        metavar='file',       help='filename of the underlying period distribution\
                                                                            (default: ulogP.dist)',                                    type=str,   default='ulogP.dist')
    parser.add_argument(      '--maxstars',     metavar='num',        help='maximum number of stars to be read in from the\
                                                                            galaxy model table (default: all)',                        type=int,   default=0)
    parser.add_argument(      '--maxEBs',       metavar='num',        help='stop when the passed number of EBs has been created\
                                                                            (default: no limit)',                                      type=int,   default=-1)
    parser.add_argument(      '--sample-size',  metavar='num',        help='number of objects to be generated (default: 200000)',      type=int,   default=200000)
    parser.add_argument(      '--output-dir',   metavar='file',       help='directory to store generated light curves (default: lcs)', type=str,   default='lcs')
    parser.add_argument(      '--on-silicon',   action='store_true',  help='generate only targets on silicon (default: False)')
    argv = parser.parse_args()

    # Initialize Kepler FOV:
    #~ kepfov = KepFOV()

    # Initialize galaxy table:
    table = Table(argv.table, maxlines=argv.maxstars, DEBUG=False)
    Nstars = table.Nstars
    print("# %d entries read in from the Galaxia table %s." % (Nstars, argv.table))
    
    # Mass ratio distribution:
    if argv.qdist == 'raghavan':
        print('# using Raghavan et al. (2010) for mass ratio distribution.')
        qdist = qdist_raghavan()

    if argv.count:
        # Read in the previously computed underlying period distribution:
        P0ranges, P0hist, P0histerr = np.loadtxt(argv.ulogP, unpack=True)
        P0ranges = np.append(P0ranges-(P0ranges[1]-P0ranges[0])/2, [P0ranges[-1]+(P0ranges[1]-P0ranges[0])/2])

        if argv.lpbin:
            # Add the last bin for the Long Period EBs (LPEBs):
            P0ranges = np.append(P0ranges, [np.inf])
            P0hist = np.append(P0hist*(1-argv.lpexcess), argv.lpexcess)

        # Number of binary stars in the sample:
        if argv.lpbin:
            Bnum = int(0.33*Nstars)
        else:
            Bnum = int((1-argv.lpexcess)*0.33*Nstars)

        print('# Number of binaries to be generated: %d' % (Bnum))

        total_EBs = 0
        total_SEBs = 0
        for i in range(Bnum):
            b = Binary(table, Pdist=(P0ranges, P0hist), qdist=qdist, check_sanity=True, safety_limit=100)

            total_EBs += b.EB
            total_SEBs += b.SEB
            print(b, ' %2.2f%%' % ((float(i)+1)/Bnum*100))

            if b.EB:
                b.compute_lc(filename='%s/eb%04d.lc' % (argv.output_dir, total_EBs))

            if total_EBs == argv.maxEBs:
                break

        print("# Total EBs:  %d/%d (%2.2f%%)" % (total_EBs, Bnum, 100*float(total_EBs)/Bnum))
        print("# Total SEBs: %d/%d (%2.2f%%)" % (total_SEBs, Bnum, 100*float(total_SEBs)/Bnum))
        
        exit()

    if argv.observe:
        # Initialize multiplicity distribution:
        if argv.mdist == 'single':
            mdist = {'S': 1.0, 'B': 0.0, 'T': 0.0, 'M': 0.0}
        elif argv.mdist == 'raghavan':
            mdist = mdist_raghavan()
        else:
            print('Unsupported multiplicity distribution, aborting.')
            exit()
        
        print('# multiplicity distribution: %2.2f single, %2.2f binary, %2.2f triple, %2.2f multi systems' % (mdist['S'], mdist['B'], mdist['T'], mdist['M']))
        
        # Read in the previously computed underlying period distribution:
        P0ranges, P0hist, P0histerr = np.loadtxt(argv.ulogP, unpack=True)
        P0ranges = np.append(P0ranges-(P0ranges[1]-P0ranges[0])/2, [P0ranges[-1]+(P0ranges[1]-P0ranges[0])/2])
        print('# underlying binary period distribution loaded from %s.' % (argv.ulogP))
        
        if argv.lpbin:
            # Add the last bin for the Long Period EBs (LPEBs):
            print('# adding long-period binary and multiple star bin.')
            P0ranges = np.append(P0ranges, [np.inf])
            P0hist = np.append(P0hist*(1-argv.lpexcess), argv.lpexcess)
        else:
            # Otherwise correct for the long period excess:
            print('# correcting occurrence rates by long period excess factor %3.3f.' % (argv.lpexcess))
            mdist['B'] *= (1-argv.lpexcess)
            mdist['T'] *= (1-argv.lpexcess)
            mdist['M'] *= (1-argv.lpexcess)
            mdist['S'] = 1.-mdist['B']-mdist['T']-mdist['M']
        
        # Build a synthetic sample of the Kepler field.
        field = []
        print('# requested sample size: %d' % argv.sample_size)
        
        if argv.maxEBs != 0:
            print('# maximum number of EBs set at: %d' % (argv.maxEBs))

            # Numbers of single, binary, triple, multiple and EB.
            Snum, Bnum, Tnum, Mnum, EBnum = 0, 0, 0, 0, 0

            # We need to observe this as we create them so that we know
            # how many EBs we have created.
            run = Observation(targets=None)

            while EBnum != argv.maxEBs:
                # Let's roll a dice to see which type of system we will
                # create. We need to do this so that we can count the number
                # of EBs that we create.
                roll = rng.random()
                if roll < mdist['S']:
                    field.append(Single(table))
                    Snum += 1
                elif roll < mdist['S'] + mdist['B']:
                    field.append(Binary(table, Pdist=(P0ranges, P0hist), qdist=qdist, check_sanity=True, safety_limit=100))
                    Bnum += 1
                elif roll < mdist['S'] + mdist['B'] + mdist['T']:
                    field.append(Triple(table, Pdist=(P0ranges, P0hist), qdist=qdist, check_sanity=True, safety_limit=100))
                    Tnum += 1
                else:
                    field.append(Multiple(table, Pdist=(P0ranges, P0hist), qdist=qdist, check_sanity=True, safety_limit=100))
                    Mnum += 1

                # Observe this target:
                run.observe_one(field[-1])

                # Is it an eclipsing binary?
                if field[-1].EB and field[-1].on_silicon and field[-1].is_target:
                    EBnum += 1

            print('# %d single stars created.' % (Snum))
            print('# %d binaries created.' % (Bnum))
            print('# %d triples created.' % (Tnum))
            print('# %d multiples created.' % (Mnum))
        else:
            Bnum = int(mdist['B']*argv.sample_size)
            Tnum = int(mdist['T']*argv.sample_size)
            Mnum = int(mdist['M']*argv.sample_size)
            Snum = argv.sample_size-Bnum-Tnum-Mnum

            # Generate single stars:
            for i in range(Snum):
                field.append(Single(table))
            print('# %d single stars created.' % (Snum))
            
            # Generate binary stars:
            for i in range(Bnum):
                field.append(Binary(table, Pdist=(P0ranges, P0hist), qdist=qdist, check_sanity=True, safety_limit=100))
            print('# %d binaries created.' % (Bnum))
            
            # Generate triple stars:
            for i in range(Tnum):
                field.append(Triple(table, Pdist=(P0ranges, P0hist), qdist=qdist, check_sanity=True, safety_limit=100))
            print('# %d triples created.' % (Tnum))
            
            # Generate multiple stars:
            for i in range(Mnum):
                field.append(Multiple(table, Pdist=(P0ranges, P0hist), qdist=qdist, check_sanity=True, safety_limit=100))
            print('# %d multiples created.' % (Mnum))
        
            run = Observation(field)
            strt = time.time()
            run.observe()
            print('# Observed in %.3fs' % (time.time() - strt))
        
        print('# Type    R.A.         Dec.       Period  OnSilicon? Detected? Selected? EB?      Teff    logg     Kp')
        for target in field:
             print('%5d  %10.6f  %10.6f  %10.6f   %5s     %5s    %5s    %5s    %5.0f    %0.2f   %5.1f' % (target.type, target.ra, target.dec, target.period, target.on_silicon, target.detected, target.is_target, target.EB, target.Teff, target.logg, target.mag))
        
        print('# Total targets on silicon: %d' % (len([t for t in field if t.on_silicon])))
        print('# Total targets selected: %d' % (len([t for t in field if t.is_target])))
        print('# Total EBs: %d' % (len([t for t in field if t.on_silicon and t.is_target and t.EB])))
        print('# Total SEBs: %d' % (len([t for t in field if t.on_silicon and t.is_target and t.SEB])))

        exit()

    if argv.solve:
        # Read in observed periods:
        catKIC, catP0 = np.loadtxt('kepEBs.csv', delimiter=',', usecols=(0, 1), unpack=True)
        print('# %d systems with measured orbital periods read in.' % (len(catKIC)))
        
        # Read in observed periods and eccentricities:
        eccKIC, eccP0, obsEB_ecc = np.loadtxt('ecc.final_with_p0.res', usecols=(0, 1, 2), unpack=True)
        print('# %d systems with measured eccentricities read in.' % (len(eccKIC)))

        # Initialize multiplicity distribution:
        if argv.mdist == 'raghavan':
            mdist = mdist_raghavan()
        print('# multiplicity distribution: %2.2f single, %2.2f binary, %2.2f triple, %2.2f multi systems' % (mdist['S'], mdist['B'], mdist['T'], mdist['M']))

        # Initialize the starting P0 histogram.
        bins = argv.bins
        P0hist, P0ranges = np.histogram(np.log10(catP0), bins=bins)

        if argv.Pdist == 'uniform':
            synP0hist = np.array([len(catP0)/bins]*(bins-len(catP0)%bins)+[len(catP0)/bins+1]*(len(catP0)%bins))
            synP0hist = np.array([float(v)/len(catP0) for v in synP0hist])
            print('# uniform underlying log10(P0) histogram with %d bins created.' % (bins))
        elif argv.Pdist == 'ulogP':
            P0ranges, synP0hist = np.loadtxt(argv.ulogP, usecols=(0, 1), unpack=True)
            P0ranges = np.append(P0ranges-(P0ranges[1]-P0ranges[0])/2, [P0ranges[-1]+(P0ranges[1]-P0ranges[0])/2])
            print('# initial underlying binary period distribution loaded from %s.' % (argv.ulogP))
        else:
            print("can't ever get here, right?")
            exit()
        
        if argv.lpbin:
            # Add the last bin for the Long Period EBs (LPEBs):
            print('# adding long-period binary and multiple star bin.')
            P0ranges = np.append(P0ranges, [np.inf])
            synP0hist = np.append(synP0hist*(1-argv.lpexcess), argv.lpexcess)
            bins += 1
        else:
            # Otherwise correct for the long period excess:
            print('# correcting occurrence rates by long period excess factor %3.3f.' % (argv.lpexcess))
            mdist['B'] *= (1-argv.lpexcess)
            mdist['T'] *= (1-argv.lpexcess)
            mdist['M'] *= (1-argv.lpexcess)
            mdist['S'] = 1.-mdist['B']-mdist['T']-mdist['M']
                
        eccpars = [3.5, 3.0, 0.23, 0.98]
        print('# initial eccentricity envelope parameters set: %s' % eccpars)
        
        # This is where the loop needs to begin.
        while True:
            # Build a synthetic sample of the Kepler field.
            field = simulate_field(table, argv, mdist, (P0ranges, synP0hist), qdist, eccpars, DEBUG=True)

            # Simulated EBs comprise our comparison sample.
            simEBs  = [t for t in field if t.on_silicon and t.is_target and t.EB]
            simDEBs = [t for t in field if t.on_silicon and t.is_target and t.EB and not t.SEB] # only doubly-eclipsing EBs should be in this sample
            simEB_P0  = np.array([eb.period for eb in simEBs])
            simEB_ecc = np.array([eb.ecc    for eb in simDEBs])
            
            simEB_hist, simEB_ranges = np.histogram(np.log10(simEB_P0), bins=P0ranges)
            
            print('# Total targets on silicon: %d' % (len([t for t in field if t.on_silicon])))
            print('# Total targets selected: %d' % (len([t for t in field if t.is_target])))
            print('# Total EBs: %d' % (len(simEBs)))
            print('# Total SEBs: %d' % (len([t for t in field if t.on_silicon and t.is_target and t.SEB])))
            
            print('# Comparison:')
            print('# PERIODS:')
            print('# %12s %12s %12s' % ('observed:', 'simulated:', 'difference:'))
            for i in range(bins):
                print('# %12.6f %12.6f %12.6f' % (float(P0hist[i])/P0hist.sum(), float(simEB_hist[i])/simEB_hist.sum(), float(P0hist[i])/P0hist.sum()-float(simEB_hist[i])/simEB_hist.sum()))
            
            print('# NUMBERS:')
            print('# EB fraction observed:  %12.6f%%' % (2775./201775*100))
            print('# EB fraction simulated: %12.6f%%' % (100*float(len(simEBs))/(len([t for t in field if t.on_silicon]))))

            sim_ecc_hist, sim_ecc_range = np.histogram(simEB_ecc, bins=np.linspace(0, 1, 10))
            obs_ecc_hist, obs_ecc_range = np.histogram(obsEB_ecc, bins=np.linspace(0, 1, 10))

            print('# ECCENTRICITIES:')
            print('# %12s %12s %12s' % ('observed:', 'simulated:', 'difference:'))
            for i in range(len(sim_ecc_hist)):
                print('# %12.6f %12.6f %12.6f' % (float(obs_ecc_hist[i])/obs_ecc_hist.sum(), float(sim_ecc_hist[i])/sim_ecc_hist.sum(), float(obs_ecc_hist[i])/obs_ecc_hist.sum()-float(sim_ecc_hist[i])/sim_ecc_hist.sum()))

            logL = -0.5*((P0hist.astype(float)/P0hist.sum()-simEB_hist.astype(float)/simEB_hist.sum())**2).sum() + 1000*(2775./201775 - float(len(simEBs))/(len([t for t in field if t.on_silicon])))**2 + ((obs_ecc_hist.astype(float)/obs_ecc_hist.sum()-sim_ecc_hist.astype(float)/sim_ecc_hist.sum())**2).sum()

            print('# logL = %f' % (logL))
            print('# logL = %f = -0.5*(%f + %f + %f)' % (logL, ((P0hist.astype(float)/P0hist.sum()-simEB_hist.astype(float)/simEB_hist.sum())**2).sum(), 1000*(2775./201775 - float(len(simEBs))/(len([t for t in field if t.on_silicon])))**2, ((obs_ecc_hist.astype(float)/obs_ecc_hist.sum()-sim_ecc_hist.astype(float)/sim_ecc_hist.sum())**2).sum()))

            break

        exit()

    # Initialize the ranges for various histograms:
    #~ qbins = np.linspace(0, 1, 20)
    #~ Rsumbins = [x for x in np.linspace(0,1,29)]+[100]

    # Reduce the font size for plots:
    #~ mpl.rcParams.update({'font.size': 7})
    
    #~ for cnt in range(1,201):    
        #~ P0dist = st.rv_discrete(name='discrete', values=(np.arange(bins), vP0syn))

        #~ P0sel, eccsel, qsel, Rsumsel, Rratsel, sinisel = [], [], [], [], [], []
        #~ numBs, numEBs = 0, 0
        
        #~ while numEBs != len(P0obs):
            #~ rP0 = P0dist.rvs()
            #~ P0 = 10**st.uniform.rvs(rP0obs[rP0], binw)
            #~ b = Binary(table.data, mode=MODE, period=P0, check_sanity=True, safety_limit=10000)

            #~ while not b.physical:
                #~ rP0 = P0dist.rvs()
                #~ P0 = 10**st.uniform.rvs(rP0obs[rP0], binw)
                #~ b = Binary(table.data, mode=MODE, period=P0, check_sanity=True, safety_limit=10000)

            #~ numBs += 1

            #~ if b.EB:
                #~ numEBs += 1
                #~ P0sel.append(np.log10(b.period))
                #~ eccsel.append(b.ecc)
                #~ qsel.append(b.q if b.q <= 1 else 1./b.q)
                #~ Rsumsel.append(b.r1+b.r2)
                #~ Rratsel.append((b.r2/b.r1) if (b.r2/b.r1) <= 1 else (b.r1/b.r2))
                #~ sinisel.append(np.sin(b.incl))

            #~ print ("# run %03d: numBs = %d, numEBs = %d" % (cnt, numBs, numEBs))

        #~ vP0sel, rP0sel = np.histogram(P0sel, bins=rP0obs)
        #~ vP0sel = [float(v)/len(P0obs) for v in vP0sel]
        
        #~ delta = vP0sel-vP0obs
        #~ cf = (delta**2).sum()
        #~ print cnt, cf

        #~ font = {'family': 'serif', 'variant': 'normal', 'weight': 'normal', 'size': 16}
        #~ mpl.rc('font', **font)
            
        #~ if EXTRA == 'onlyP':
            #~ plt.close()
            #~ fig = plt.figure(1)
            #~ fig.set_size_inches(3.6,7.2)
            #~ fig.patch.set_alpha(0.0)
            #~ plt.suptitle("i=%03d, %d bins, xi=%2.2f, cf=%f" % (cnt, bins, xi, cf), fontsize=12)
            #~ plt.subplot(311)
            #~ plt.ylabel("dN/dlogP")
            #~ plt.xlim(-1.2, 3.03)
            #~ plt.ylim(0,0.14)
            #~ plt.bar(rP0sel[:-1], vP0syn, binw)
            #~ plt.subplot(312)
            #~ plt.ylabel("Nsyn")
            #~ plt.xlim(-1.2, 3.03)
            #~ plt.ylim(0, 200)
            #~ plt.hist(P0sel, bins=rP0sel)
            #~ plt.subplot(313)
            #~ plt.xlim(-1.2, 3.03)
            #~ plt.ylim(0, 200)
            #~ plt.xlabel("logP")
            #~ plt.ylabel("Nobs")
            #~ plt.hist(P0obs, bins=rP0obs)
            #~ plt.subplots_adjust(left=0.23, right=0.98, top=0.93, bottom=0.1)
            #~ plt.savefig("img%03d.png" % cnt, dpi=100)

        #~ else:
            #~ plt.clf()
            #~ fig = plt.figure(1)
            #~ fig.patch.set_alpha(0.0)
            
            #~ plt.suptitle("Iteration %03d, %d bins, bin width=%3.3f, xi=%2.2f, cf=%f" % (cnt, bins, binw, xi, cf), fontsize=12)
            #~ plt.subplot(331)
            #~ plt.ylabel("dN/dlogP")
            #~ plt.xlim(-1.2, 3.03)
            #~ plt.ylim(0,0.14)
            #~ plt.bar(rP0sel[:-1], vP0syn, binw)
            #~ plt.subplot(334)
            #~ plt.ylabel("Nsyn")
            #~ plt.xlim(-1.2, 3.03)
            #~ plt.ylim(0, 200)
            #~ plt.hist(P0sel, bins=rP0sel)
            #~ plt.subplot(337)
            #~ plt.xlim(-1.2, 3.03)
            #~ plt.ylim(0, 200)
            #~ plt.xlabel("logP")
            #~ plt.ylabel("Nobs")
            #~ plt.hist(P0obs, bins=rP0obs)
            #~ plt.subplot(332)
            #~ plt.xlabel("logP")
            #~ plt.ylabel("ecc_syn")
            #~ plt.plot(P0obs, eccobs, 'rx', markersize=0.5)
            #~ plt.plot(P0sel, eccsel, 'b.', markersize=0.5)
            #~ plt.subplot(335)
            #~ plt.xlim(0, 1)
            #~ plt.ylim(0, 600)
            #~ plt.xlabel("Eccentricity")
            #~ plt.ylabel("Nsyn")
            #~ plt.hist(eccsel, bins=20)
            #~ plt.subplot(338)
            #~ plt.xlim(0, 1)
            #~ plt.ylim(0, 600)
            #~ plt.xlabel("Mass ratio")
            #~ plt.ylabel("Nsyn")
            #~ plt.hist(qsel, qbins)
            #~ plt.subplot(333)
            #~ plt.xlabel("(R1+R2)/a")
            #~ plt.ylabel("Nsyn")
            #~ plt.xlim(0, 1.033)
            #~ plt.hist(Rsumsel, bins=Rsumbins)
            #~ plt.subplot(336)
            #~ plt.xlabel("R2/R1")
            #~ plt.ylabel("Nsyn")
            #~ plt.hist(Rratsel, bins=30)
            #~ plt.subplot(339)
            #~ plt.xlabel("sin(i)")
            #~ plt.ylabel("Nsyn")
            #~ plt.hist(sinisel, bins=30)
            #~ plt.savefig("img%03d.png" % cnt, dpi=200)

        # Correct the input histogram:
        #~ for i in range(len(vP0syn)):
            #~ vP0syn[i] -= xi*delta[i]
        #~ vP0syn /= vP0syn.sum()
