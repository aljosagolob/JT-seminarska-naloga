from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm

diarizationErrorRate = DiarizationErrorRate()

reference = load_rttm("reference.rttm")["aepyx"]
hypothesis = load_rttm("output.rttm")["aepyx"]

print("DER = {0:.3f}".format(diarizationErrorRate(reference, hypothesis)))