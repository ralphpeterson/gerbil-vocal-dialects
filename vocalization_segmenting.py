import sys
from scipy.io import wavfile
from scipy.signal import stft, spectrogram
from scipy.ndimage.filters import gaussian_filter
import os
import numpy as np
from librosa import feature as libfeature


def get_spec_min_max(wav_path, start_s=0, stop_s=-1):

	"""
	Automatically calculate the 'spec_min_val' and 'spec_max_val' parameter for ava segmenting.
	
	Parameters:
	wav_path: str, path
		Path to an audio file with vocalizations
	start_s, stop_s: int
		If you're audio file is huge, use a small segment with a diversity of loud and soft sounds, bounded onset start_s (seconds) and offset stop_s (seconds)

	Returns:
	percentiles: tuple
		spec_min_val and spec_max_val for segmentation
	sr: int
		audio sampling rate for segmentation
	"""

	#load audio
	sr, audio = wavfile.read(wav_path)

	#chunk the audio, if your file is huge
	audio_segment = audio[int(start_s*sr):int(stop_s*sr)] 

	#compute stft of chunk
	_, _, scipy_spec = stft(audio_segment,
							fs=sr,
							nperseg=512,
							noverlap=256)
	#compute spectrogram
	scipy_spec = np.log(np.abs(scipy_spec) + 1e-9)

	#calculate percentiles for ava spectrogram processing
	percentiles = np.quantile(scipy_spec, (.95, .975))

	return percentiles, sr

def load_oo(wav_path):

	
	"""
	Loads onsets/offsets from segment files.

	Parameters:
		wav_path: str, path
			Path to an audio file with vocalizations

	"""    
	dirname, basename = os.path.split(wav_path)
	segment_txt_file = os.path.join(dirname, 'segment', basename.replace('wav', 'txt'))
	
	data = np.genfromtxt(segment_txt_file)

	if data.shape == (0,):
		return

	if data.shape == (2,):
		data = np.reshape(data, (1,2))

	#load in onset/offset times from text file
	onsets, offsets = data[:,0], data[:,1]
	
	return onsets, offsets



def get_sf(audio):
	"""
	Get the mean spectral flatness of an audio snippet.

	Paramters:
		audio: np.array
			Audio of a single segment extracted by AVA.

	Returns:
		sf: float
			Spectral flatness of the snippet. Typically less than 0.3 is a vocalization.
	"""

	sf = libfeature.spectral_flatness(y=audio, n_fft=256, 
										hop_length=128, win_length=256, 
										center = False, power=2.0)[0]
	return np.mean(sf)


def filter_segments(wav_path, sf_threshold=0.3):
	"""
	This function filters out putative vocalizations from noise using spectral flatness.

	Parameters:
		wav_path: str, path
			Path to an audio file with vocalizations
		sf_threshold: float
			Spectral flatness threshold. More strict is towards 0.0.
	Returns:
		onsets_filt, offsets_filt: np.array
			Onsets/offsets of putatitve vocalizations
	"""

	#load audio
	sr, audio = wavfile.read(wav_path)

	#get onsets/offsets
	onsets, offsets = load_oo(wav_path)

	#get spectral flatness for all sounds detected
	sfs = np.array([get_sf(audio[on:off]) for on, off in zip((onsets*sr).astype(int), (offsets*sr).astype(int))])

	#filter the onsets/offsets
	onsets_filt, offsets_filt = onsets[sfs<sf_threshold], offsets[sfs<sf_threshold]

	#save new onsets/offsets
	dirname, basename = os.path.split(wav_path)
	new_seg_dir = os.path.join(dirname, 'segment_filt')
	if not os.path.exists(new_seg_dir):
		os.makedirs(new_seg_dir)

	outfile = os.path.join(new_seg_dir, basename.replace('wav', 'txt'))
	np.savetxt(outfile, np.vstack((onsets_filt, offsets_filt)).T)
	print('{} segments after filtering. Saved to: {}'.format(len(onsets_filt), outfile))