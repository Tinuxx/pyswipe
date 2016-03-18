# -*- coding: utf8 -*-

from scipy.io import wavfile;
from scipy.signal import spectrogram;
from numpy import testing;
import numpy as np;

import os;

import pyswipe;

class TestPitchStrengthAllCandidatesFunction():
	test_data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_vardumps', 'pitch_strength_all_candidates')
	def test_correct_return_values(self):
		"""
		Checkear similitud de resultados de pitch strength (one candidate) con implementación original
		"""
		for i in range(1, 10):
			yield self.check_output_pitch_strength_all_candidates, i

	def check_output_pitch_strength_all_candidates(self, n):
		input_f_filename = os.path.join(self.test_data_directory, 'f_' + str(n) + '.txt')
		input_L_filename = os.path.join(self.test_data_directory, 'L_' + str(n) + '.txt')
		input_pc_filename = os.path.join(self.test_data_directory, 'pc_' + str(n) + '.txt')
		output_S_filename = os.path.join(self.test_data_directory, 'S_' + str(n) + '.txt')
		f = np.loadtxt(input_f_filename)
		L = np.loadtxt(input_L_filename)
		pc = np.loadtxt(input_pc_filename)
		S = np.loadtxt(output_S_filename)
		result = pyswipe.pitch_strength_all_candidates(f, L, pc)
		testing.assert_array_almost_equal(result, S)

class TestPitchStrengthOneCandidateFunction():
	test_data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_vardumps', 'pitch_strength_one_candidate')
	def test_correct_return_values(self):
		"""
		Checkear similitud de resultados de pitch strength (one candidate) con implementación original
		"""
		for i in range(1, 10):
			yield self.check_output_pitch_strength_one_candidate, i

	def check_output_pitch_strength_one_candidate(self, n):
		input_f_filename = os.path.join(self.test_data_directory, 'f_' + str(n) + '.txt')
		input_NL_filename = os.path.join(self.test_data_directory, 'NL_' + str(n) + '.txt')
		input_pc_filename = os.path.join(self.test_data_directory, 'pc_' + str(n) + '.txt')
		output_S_filename = os.path.join(self.test_data_directory, 'S_' + str(n) + '.txt')
		f = np.loadtxt(input_f_filename)
		NL = np.loadtxt(input_NL_filename)
		pc = np.loadtxt(input_pc_filename)
		S = np.loadtxt(output_S_filename)
		result = pyswipe.pitch_strength_one_candidate(f, NL, pc)
		testing.assert_array_almost_equal(result, S)

class TestHzToErbsFunction():
	def test_correct_return_type(self):
		"""
		Debe devolver un tipo flotante
		"""
		result = pyswipe.hz_to_erbs(hz=np.float_(0.2943))
		assert (result.dtype == np.dtype('float64'))

	def test_correct_values(self):
		"""
		Para 7.5, debe dar 0.294284644305706
		Para 22050, debe dar 42.5258757645858
		"""
		input_array = np.array([7.5, 22050.])
		expected_output = np.array([0.294284644305706, 42.5258757645858])
		result = pyswipe.hz_to_erbs(hz=input_array)
		testing.assert_almost_equal(result, expected_output)

class TestErbsToHzFunction():
	test_data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_vardumps')
	def test_correct_return_type_single_value(self):
		"""
		Debe devolver un tipo flotante
		"""
		result = pyswipe.erbs_to_hz(erbs=np.float_(0.2943))
		assert (result.dtype == np.dtype('float64'))

	def test_correct_return_type_array(self):
		"""
		Debe devolver un arreglo de float64
		"""
		array = np.arange(0.294284644305706, 42.5942846443057, .1)
		result = pyswipe.erbs_to_hz(erbs=array)
		testing.assert_equal(result.size, array.size)

	def test_correct_return_value(self):
		"""
		Checkear similitud de resultados de pitch strength (one candidate) con implementación original
		"""
		input_array = np.loadtxt(os.path.join(self.test_data_directory, 'test_erbs_to_hz_correct_values_input.txt'))
		output_array = np.loadtxt(os.path.join(self.test_data_directory, 'test_erbs_to_hz_correct_values_output.txt'))
		result = pyswipe.erbs_to_hz(erbs=input_array)
		testing.assert_array_almost_equal(result, output_array)
