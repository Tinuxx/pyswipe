# -*- coding: utf8 -*-

from sys import exit, argv;
from scipy.io import wavfile;
from scipy.signal import spectrogram;
import numpy as np;
import argparse;
from collections import namedtuple;

def swipep(x, fs,
		   pmin = 30.,
		   pmax = 5000.,
		   dt = .001,
		   dlog2p = 1./48.,
		   d_erbs = .1,
		   woverlap = .5,
		   sthr = -np.inf):
	if (x.dtype == np.dtype('int16')):
		x = x / (2.**15)
	if pmin is None:
		pmin = 30.0
	if pmax is None:
		pmax = 5000.0
	if dt is None:
		dt = 0.001
	if dlog2p is None:
		dlog2p = 1./48.
	if d_erbs is None:
		d_erbs = 0.1,
	if woverlap is None:
		woverlap = 0.5
	if sthr is None:
		sthr = -np.inf
	# woverlap must be between 0 and 1
	if (woverlap < 0.0 or woverlap > 1.0):
		raise ValueError('Window overlap must be between 0 and 1')

	t = np.arange(0., x.size / np.float_(fs) + dt, dt)

	# Define pitch candidates
	log2pc = np.arange(np.log2(pmin), np.log2(pmax), dlog2p)
	pc = 2 ** log2pc.transpose()
	S = np.zeros((pc.size, t.size))

	# Determine P2-WSs
	log_ws = np.round(np.log2(8 * fs / np.array([pmin, pmax])))

	# TODO One is substracted from log_ws[1] in order for it to be included;
	# this bit may cause issues
	ws = 2 ** np.arange(log_ws[0], log_ws[1]-1, -1) # P2-WSs
	pO = 8 * fs / ws # Optimal pitches for P2-WSs

	# Determine window sizes used by each pitch candidates
	d = 1 + log2pc - np.log2(8 * fs / ws[0])

	# Create ERB-scale uniformly-spaced frequencies (in Hertz)
	f_erbs = erbs_to_hz(np.arange(hz_to_erbs(pc.min() / 4),hz_to_erbs(fs/2), d_erbs))

	for i in xrange(0, ws.size):
		dn = max(1 , np.round(8 * (1 - woverlap) * fs / pO[i])) # Hop size
		# Zero pad signal
		xzp = np.hstack((np.zeros((int(ws[i]/2))), x, np.zeros((int(dn + ws[i]/2)))))

		# Compute spectrum

		# NOTE Numpy's Hanning window has slightly different values when
		# compared to Matlab's. Results from this point forward are inevitably
		# going to vary. So far, this has not proven to have a tangible effect
		# on final results

		w = np.hanning(ws[i])
		o = int(max(0, np.round(ws[i] - dn)))

		# NOTE Similarly, the magnitudes given by Scipy's spectrogram() are
		# different to the ones obtained in Matlab. To make it worse, Matlab's
		# own deprecated function specgram() is inconsistent with its equivalent
		# spectrogram() call. The original implementation uses specgram(), not
		# spectrogram(). From this point forward, I stopped checking for equal
		# numeric results, and judged this code's accuracy by comparing various
		# function plots. Doing this has served well to this moment.

		# NOTE An important difference is the time instances return array, ti.
		# In specgram(), the first value is alwayz zero. In spectrogram(), each
		# value in ti is each section's midpoint. When needed, substract ti[0]
		# from ti in order to achieve the same behavior as specgram().

		f, ti, X = spectrogram(
			x = xzp,
			fs = fs,
			window = w,
			nperseg = w.size,
			noverlap = o,
			nfft = int(ws[i]),
			scaling = 'spectrum',
			mode = 'complex'
		)

		# TODO if has not been tested for all branches

		if ws.size == 1:
			j = pc.T
			k = np.array([])
		else:
			if i == ws.size:
				j = np.nonzero(d - (i + 1) > 1.)[0]
				k = np.nonzero(d[j] - (i + 1) > 0.)[0]
			else:
				if i == 0:
					j = np.nonzero(d - (i + 1) < 1.)[0]
					k = np.nonzero(d[j] - (i + 1) > 0.)[0]
				else:
					j = np.nonzero(np.absolute(d - (i + 1)) < 1.)[0]
					k = np.arange(0, j[0].size)
		f_erbs = f_erbs[np.argmax(f_erbs > pc[j[0]]/4) : ]

		# NOTE MATLAB does interpolation on each column of a matrix. Scipy has a
		# method that behaves similarly, but cannot have the query points
		# specified by the user. Numpy's interp() accepts a query points array,
		# but operates one column at a time. Therefore, one must iterate through
		# all columns, perform interpolation, and join the results afterwards
		for m in xrange(0, X.shape[1]):
			if m == 0:
				L = np.sqrt(np.interp(f_erbs, f, np.abs(X[:,m])))
			else:
				L = np.vstack((L, np.sqrt(np.interp(f_erbs, f, np.abs(X[:,m])))))
		L = L.T

		# Compute pitch strength
		Sip = pitch_strength_all_candidates(f_erbs, L, pc[j])

		# Interpolate pitch strength at desired times
		if Si.shape[1] > 1:
			# NOTE Read line 115 on the need to iterate columns while
			# interpolating matrices

			tii = ti - ti[0]
			for m in xrange(0, Sip.shape[0]):
				if m == 0:
					Si = np.interp(t, tii, Sip[m], left = np.nan, right = np.nan)
				else:
					Si = np.vstack((Si, np.interp(t, tii, Sip[m], left = np.nan, right = np.nan)))
		else:
			# TODO This if branch has not been tested
			Si = np.empty((Sip.shape[0], t.size))
			Si[:] = np.nan

		# Add pitch strength to combination
		lmbd = d[j[k]] - (i + 1)
		mu = np.ones((j.size))
		mu[k] = 1 - np.absolute(lmbd)
		S[j] = S[j] + np.tile(mu, (Si.shape[1], 1)).T * Si

		# FIXME While plotting S, one can observe notable leaps and
		# irregularities between this and the one obtained in Matlab. The
		# overall shapes seems to match up. Check what is happening for each and
		# all loop iterations, especially for large values of j

	# Fine tune pitch using parabolic interpolation
	p = np.empty((S.shape[1]))
	p[:] = np.nan
	s = np.empty((S.shape[1]))
	s[:] = np.nan
	for j in xrange(0, S.shape[1]):
		s[j] = np.nanmax(S[:,j])
		i = np.argmax(S[:,j])
		if s[j] < sthr:
			continue
		if i == 0 or i == pc.size - 1:
			p[j] = pc[i]
		else:
			I = np.arange(i - 1, i + 2)
			tc = 1 / pc[I]
			ntc = (tc / tc[1] - 1) * 2 * np.pi
			c = np.polyfit(ntc, S[I, j], 2)
			ftc = 1 / 2 ** np.arange(np.log2(pc[I[0]]), np.log2(pc[I[2]]), 1./12./100.)
			nftc = (ftc / tc[1] - 1) * 2 * np.pi
			polyfit_nftc = np.polyval(c, nftc)
			s[j] = np.nanmax(polyfit_nftc)
			k = np.argmax(polyfit_nftc)
			p[j] = 2 ** (np.log2(pc[I[0]])) + (k - 1) / 12. / 100.

	l = L.sum() # Total loudness
	l = np.interp(t, ti, l, left=np.nan, right=np.nan)

	SwipeOutput = namedtuple('SwipeOutput', 'estimated_pitch times strength loudness candidates pitch_matrix')
	return SwipeOutput(estimated_pitch=p, times= t, strength= s, loudness= l, candidates= pc, pitch_matrix= S)

def pitch_strength_all_candidates(f, L, pc):
	# Create pritch strength matrix
	S = np.zeros((pc.size, L.shape[1]))

	# Define integration regions
	k = np.zeros(pc.size + 1)

	for j in xrange(0, k.size - 1):
		k[j+1] = k[j] + np.argmax(f[np.int16(k[j]):] > pc[j]/4)

	k = k[1:]

	# Create loudness normalization matrix
	N = np.sqrt(np.flipud(np.cumsum(np.flipud(L * L), 0)))

	for j in xrange(0, pc.size):
		# Normalize loudness
		n = N[np.int16(k[j]),:]
		n[n == 0] = -np.inf # to make zero-loudness equal zero after normalization
		NL = L[np.int16(k[j]) :] / np.tile(n, (L.shape[0] - k[j], 1))

		# Compute pitch strength
		S[j] = pitch_strength_one_candidate(f[np.int16(k[j]) :], NL, pc[j])

	return S

def pitch_strength_one_candidate(f, NL, pc):
	n = np.fix(f[-1]/pc - 0.75) # Number of harmonics
	if n == 0:
		return np.nan
	k = np.zeros(f.shape) # Kernel

	#Normalize frequency w.r.t. candidate
	q = f / pc;

	# Create kernel
	for i in np.concatenate((np.array([1.]), primes_from_2_to(n))):
		a = np.absolute(q - i)

		# Peak's weight
		p = a < .25
		k[p] = np.cos(2*np.pi * q[p])

		# Valley's weights
		v = np.logical_and(.25 < a, a < .75)
		k[v] = k[v] + np.cos(2*np.pi * q[v]) / 2.

	# Apply envelope
	k = k * np.sqrt(1./f)

	# K+-normalize kernel
	k = k / np.linalg.norm(k[k>0])

	# Compute pitch strength
	return k.dot(NL);

def hz_to_erbs(hz):
	return 6.44 * (np.log2(229. + hz) - 7.84)

def erbs_to_hz(erbs):
	return (2. ** (erbs / 6.44 + 7.84)) - 229.;

# Taken from Stack Overflow:
# http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n/3035188#3035188
#
# NOTE This function always returns [2, 3], for n < 6. A special case for n <= 2
# has been added.
# NOTE Originally, this function did not return n itself, even if it was prime.
# One is added to it in order to resemble the behavior of Matlab's primes()
def primes_from_2_to(n):
	""" Input n>=6, Returns a array of primes, 2 <= p < n """
	if n <= 2:
		return np.array([2.])
	n += 1
	sieve = np.ones(n/3 + (n%6==2), dtype=np.bool)
	for i in xrange(1,int(n**0.5)/3+1):
		if sieve[i]:
			k=3*i+1|1
			sieve[       k*k/3     ::2*k] = False
			sieve[k*(k-2*(i&1)+4)/3::2*k] = False
	return np.r_[2,3,((3*np.nonzero(sieve)[0][1:]+1)|1)]

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Pitch estimation using SWIPE' algorithm");
	parser.add_argument("filename",
						help="Name of the file containing the signal")
	parser.add_argument("-i",
						"--pmin",
						help="Lowest frequency value (in Hertz) in search range",
						type=float)
	parser.add_argument("-a",
						"--pmax",
						help="Highest frequency value (in Hertz) in search range",
						type=float)
	parser.add_argument("-t",
						"--dt",
						help="Time interval length (in seconds). The pitch will be estimated every t seconds",
						type=float)
	parser.add_argument("-r",
						"--dlog2p",
						help="Pitch candidates resolution, i.e. amount of steps per octave",
						type=float)
	parser.add_argument("-e",
						"--d_erbs",
						help="Spectrum sampling step size (in ERBs)",
						type=float)
	parser.add_argument("-o",
						"--woverlap",
						help="Window overlap, between 0 to 1",
						type=float)
	parser.add_argument("-s",
						"--sthr",
						help="Minimum pitch strength. Values lower than sTHR will be treated as undefined",
						type=float)

	args = parser.parse_args()

	fs, x = wavfile.read(args.filename)
	pmin = None
	pmax = None
	dt = None
	dlog2p = None
	d_erbs = None
	woverlap = None
	sthr = None

	if args.pmin:
		pmin = args.pmin
	if args.pmax:
		pmax = args.pmax
	if args.dt:
		dt = args.dt
	if args.dlog2p:
		dlog2p = 1/args.dlog2p
	if args.d_erbs:
		d_erbs = args.d_erbs
	if args.woverlap:
		woverlap = float(args.woverlap)
	if args.sthr:
		sthr = args.sthr

	swipep(x=x, fs=fs, pmin=pmin, pmax=pmax, dt=dt, dlog2p=dlog2p, d_erbs=d_erbs, woverlap=woverlap, sthr=sthr)

	# TODO Decide what to do when invoked from console. Output to a CSV file and
	# generating an image of the result's plot could be two possibilities
