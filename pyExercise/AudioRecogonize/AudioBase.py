# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 02:16:22 2018

@author: Aswin
"""
import speech_recognition as sr
import InitMongoClient as imc
#import Capture as cap
from datetime import datetime
import json

class AudioBase:
    """
    """
    #instance = create_Instance
#    NEW = AudioBase.create_Instance()

    def __init__(self, fileName, audioFormat):
        self.FILE_NAME = fileName
        self.FORMAT = audioFormat

#
#    def create_Instance():
#        global NEW
#        NEW = sr.Recognizer()
#        return NEW

    def start_recon(self):
        NEW = sr.Recognizer()
        # NEW.dynamic_energy_threshold = True
        # NEW.adjust_for_ambient_noise(0.5)
        # obtain audio from the microphone
        with sr.Microphone() as source:
            print("Say something!")
            audio = NEW.listen(source)

            # recognize speech using Google Speech Recognition
            try:
                # for testing purposes, we're just using the default API key
                # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
                # instead of `r.recognize_google(audio)`
                #OUTPUT = NEW.recognize_google(audio) 
                OUTPUT = "for testing purposes"
                #cap.CAPTURE.sketch(OUTPUT)
                print("(Google Speech Recognition) guessed :-  " + OUTPUT)
                desc = "desc"
                TO_DICT = {OUTPUT : datetime.now()}
                imc.create_Instance().insert_one(TO_DICT).inserted_id
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print(
                    "Could not request results from Google Speech Recognition service; {0}".format(e))

        return OUTPUT

    def WAV_Transcribe(self):
        r = sr.Recognizer()
        audio_file_name = self.FILE_NAME + self.FORMAT

        # use "test.wav" as the audio source
        with sr.WavFile(audio_file_name) as source:
            # extract audio data from the file
            audio = r.record(source)

        try:
            list = r.recognize_google(audio)
            # recognize speech using Google Speech Recognition
            print("Transcription: " + r.recognize_google(audio))

            for prediction in list:
               print(" " + prediction["text"] + " (" + str(prediction["confidence"]*100) + "%)")
        except LookupError:                                 # speech is unintelligible
            print("Could not understand audio")


a = AudioBase("AudioRecogonize\audio\OSR_us_000_0010_8k.wav", "")
#a.WAV_Transcribe()
a.start_recon()
