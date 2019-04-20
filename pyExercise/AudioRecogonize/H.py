'''import logging
import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
print(__name__)
capture_string = "- " + str(datetime.datetime.now()) + "  :- "
logger.info(capture_string + 'Start reading database')
# read database here
records = {'john': 55, 'tom': 66}
logger.debug('Records: %s', records)
logger.info('Updating records ...')
# update records here
logger.info('Finish updating records')'''
import speech_recognition as sr

r = sr.Recognizer()
r.dynamic_energy_threshold = False

print(sr.WavFile("AudioRecogonize\audio\44.1k_16PCM_eng.mp3").DURATION)
with sr.WavFile("AudioRecogonize\audio\44.1k_16PCM_eng.mp3") as source:              # use "test.wav" as the audio source
    audio = r.record(source)                        # extract audio data from the file
    
try:
    list = r.recognize(audio,True)                  # generate a list of possible transcriptions
    print("Possible transcriptions:")
    for prediction in list:
        print(" " + prediction["text"] + " (" + str(prediction["confidence"]*100) + "%)")
except LookupError:                                 # speech is unintelligible
    print("Could not understand audio")