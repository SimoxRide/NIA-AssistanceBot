
from concurrent.futures import thread
from glob import glob
from tokenize import Special


from turtle import color, speed
from unicodedata import name
from unittest import result
import keyring
import speech_recognition as sr
from gtts import gTTS 
import subprocess
import playsound
import os,sys
from datetime import datetime
import time as t
from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
import pyautogui as pa
import asyncio
from threading import Thread
import random
import pyfiglet
from sqlalchemy import true
from termcolor import colored
from chatterbot.chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.response_selection import get_first_response
from chatterbot.comparisons import LevenshteinDistance
from chatterbot.trainers import ChatterBotCorpusTrainer
import spacy
import logging
import wx
import pyHook
import win32clipboard
import json
import time as t
from pytube import YouTube

import cv2,numpy

import operator


IANAME="Bob"
driver=[]
kdomanda=1


ops = {
    '+' : operator.add,
    '-' : operator.sub,
    '*' : operator.mul,
    '/' : operator.truediv,  # use operator.div for Python 2
    '%' : operator.mod,
    '^' : operator.xor,
    'diviso' : operator.truediv,
    'elevato' : operator.mul,
}

def eval_binary_expr(op1, oper, op2):
    op1, op2 = int(op1), int(op2)
    if  oper=="elevato":
        result=op1
        for i in range(0,op2,1):
            result=result*op1
        if len(str(result)>12):
            Speech("é una cifra cosi grande che non so neanche come chiamarla")
        return result

    return ops[oper](op1, op2)

#nlp = spacy.load("en_core_web_sm")
cantalk=False
savingface=False

chosingquality=False
video_object=""
def on_complete(stream, filepath):
	Speech('Download  Completato')
	print(filepath)

def on_progress(stream, chunk, bytes_remaining):
	progress_string = f'{round(100 - (bytes_remaining / stream.filesize * 100),2)}%'
	Speech("Download al"+str(progress_string))


haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'  #All the faces data will be present this folder


def RegUser(sub_data):
    path = os.path.join(datasets, sub_data)
    if not os.path.isdir(path):
        os.mkdir(path)
    (width, height) = (130, 100)    # defining the size of image


    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0) #'0' is use for my webcam, if you've any other camera attached use '1' like this

    # The program loops until it has 30 images of the face.



    count = 1
    Speech("Segui le instruzioni nella preview")
    while count < 1000: 
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            cv2.imwrite('%s/%s.png' % (path,count), face_resize)
            if(count <200):
                cv2.putText(im, "Look in front of you",(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0)) 
            elif(count <400 ):
                cv2.putText(im, "Look a bit right of you",(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            elif(count <600 ):
                cv2.putText(im,"Look a bit left of you",(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0)) 
            elif(count <800 ):
                cv2.putText(im,"Look a bit up",(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))  
            elif(count <1000 ):
                cv2.putText(im, "Look a bit down",(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))    
        count += 1
	
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break




def RecognizeFace():
    size = 4
    haar_file = 'haarcascade_frontalface_default.xml'
    datasets = 'datasets'
    # Part 1: Create fisherRecognizer
    print('Recognizing Face Please Be in sufficient Light Conditions...')
    # Create a list of images and a list of corresponding names
    (images, lables, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                lable = id
                images.append(cv2.imread(path, 0))
                lables.append(int(lable))
            id += 1
    (width, height) = (130, 100)

    # Create a Numpy array from the two lists above
    (images, lables) = [numpy.array(lis) for lis in [images, lables]]

    # OpenCV trains a model from the images
    # NOTE FOR OpenCV2: remove '.face'
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, lables)

    # Part 2: Use fisherRecognizer on camera stream
    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0)
    Find=True
    while Find:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            # Try to recognize the face
            prediction = model.predict(face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if prediction[1]<100:
                cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0)) 
                Speech(f"Sei {names[prediction[0]]}")
                Find=False
                break
            else:
                cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                Speech("Non ti conosco chi sei?")
                savingface=True
       
    	

        #cv2.imshow('OpenCV', im)
    
    
        key = cv2.waitKey(10)
        if key == 27:
            break





logging.basicConfig(level=logging.CRITICAL)

bot = ChatBot(
    f"{IANAME}",
    filters=["chatterbot.filters.RepetitiveResponseFilter"],
    nlp = spacy.load("it_core_news_sm"),
    storage_adapter = "chatterbot.storage.SQLStorageAdapter",
    database = "./db.sqlite3",
    logic_adapters = [
        "chatterbot.logic.BestMatch",
        
    ],
    statement_comparison_function = LevenshteinDistance,
    response_selection_method = get_first_response
)
trainer = ListTrainer(bot)
with open("chatterprova.txt") as f:
    conversation = f.readlines()
    trainer.train(conversation)

with open("food.txt") as f:
    foods=f.readlines()
    trainer.train(foods)
    

while False:
    try:
        user_input = input("Tu: ")
        bot_response = bot.get_response(user_input)
        print(f"{IANAME}: ", bot_response)
    except(KeyboardInterrupt, EOFError, SystemExit):
        print("GoodBye!")
        break


text=f"{IANAME.upper()} AI 2.0"

print(colored(pyfiglet.figlet_format(text),"red"))





notMusic=True


class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback
        self._task = asyncio.ensure_future(self._job())

    async def _job(self):
        await asyncio.sleep(self._timeout)
        await self._callback()

    def cancel(self):
        self._task.cancel()


async def timeout_callback():
    await asyncio.sleep(0.1)
    Speech("Tempo Timer scaduto")
    


async def main():
    print('\nfirst example:')
    timer = Timer(2, timeout_callback)  # set timer for two seconds
    await asyncio.sleep(2.5)  # wait to see timer works

    print('\nsecond example:')
    timer = Timer(2, timeout_callback)  # set timer for two seconds
    await asyncio.sleep(1)
    timer.cancel()  # cancel it
    await asyncio.sleep(1.5)  # and wait to see it won't call callback


async def TimerEnd(time):
    timer = Timer(time, timeout_callback)  # set timer for two seconds
    await asyncio.sleep(time+0.1)
    


def callback(recognizer, source):
    print("Ok! sto ora elaborando il messaggio!")
    global chosingquality
    new_string = ""
   
    try:

        text = recognizer.recognize_google(source, language="it-IT")
        print(f"{IANAME.upper()}: \n", text)
        new_string+=text
        if str(text).lower().startswith(f"{IANAME.lower()}") or  cantalk or savingface:
            CheckMatch(text)
        elif chosingquality and (str(text).lower()=="alta" or str(text).lower()=="bassa" or str(text).lower()=="solo audio"):
            CheckMatch(text)
        sr.Recognizer().listen_in_background(sr.Microphone(),callback,10)
        exit()

    except sr.RequestError as  exc:
        print(exc)
    except sr.UnknownValueError:
        print("unable to recognize")
    except Exception as e:
        print(e)
    
    return
    
    
        



def RecognizeMessage():
    recognizer_instance = sr.Recognizer() # Crea una istanza del recognizer
    
    mic=sr.Microphone()
    with mic:   
        recognizer_instance.adjust_for_ambient_noise(mic)
        print("\033[31;42mSono in ascolto... parla pure!")
       
    recognizer_instance.listen_in_background(mic,callback,10)
    
    
    
    
    
    

def Speech(text):
    tts = gTTS(text=text, lang='it')
    x = random.randint(0,300)
    path2="tempaudio"+str(x)+".mp3"
    tts.save(path2)
    
    playsound.playsound(path2)
    if os.path.exists(path2):
        try:
            os.remove(path2)
        except Exception as e:
            print(f"{IANAME.upper()}[ERROR]:Non sono riuscito a rimuovere il file audio")



def CheckCall(text):
    try:
        a=""
        a=str(RecognizeMessage())
        print(a)
        if a ==IANAME:
            Speech("Si?")
            CheckMatch(RecognizeMessage())
        elif a.startswith(IANAME) or a.startswith(IANAME.lower()):
            CheckMatch(a)

    except Exception as e:
        Speech("Non ho capito bene")
        print("[ERROR]"+str(e))

    
def tstart(tick):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(TimerEnd(tick))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()



def CheckMatch(text):
    global video_object
    global chosingquality
    global cantalk
    global kdomanda
    global driver
    global savingface
    try:
        text=str(text).lower()
        if (str(text).__contains__("tempo") or str(text).__contains__("ore")) and not str(text).__contains__("non"):
            now = datetime.now()
            Speech("Sono le"+now.strftime("%H e %M"))
        elif (str(text).__contains__("cerca") or str(text).__contains__("ricerca"))and not str(text).__contains__("non"):
            DRIVER_PATH = os.path.join("BrowserDriver","chromedriver.exe")
            driver = webdriver.Chrome(executable_path=DRIVER_PATH)
            if str(text).__contains__("cerca"):
                kdomanda=1
                print(str(text).split('cerca')[1])
                driver.get("https://www.google.com/search?q="+str(text).split('cerca')[1])
                driver.fullscreen_window()
            else:
                driver.get("https://www.google.com/search?q="+str(text).split('ricerca')[1])
                driver.fullscreen_window()
                
        elif(str(text).__contains__("ferma") and str(text).__contains__("musica") ) or ((str(text).__contains__("riproduci") or str(text).__contains__("avvia"))and str(text).__contains__("musica")):
            global notMusic
            if  notMusic and (str(text).__contains__("riproduci") or str(text).__contains__("avvia")):
                pa.hotkey("playpause")
                notMusic=False
            elif not notMusic and str(text).__contains__("ferma"):
                pa.hotkey("playpause")
                notMusic=True
        elif str(text).__contains__("volume"):
            if (str(text).__contains__("alza") or str(text).__contains__("aumenta"))and not str(text).lower().__contains__("spotify"):
                if str(text).__contains__("di"):
                    for i in range(0,int(int(str(text).split("di")[1])/2),1):
                        pa.hotkey("volumeup")
                elif str(text).__contains__(" a "):
                    for i in range(0,int(int(str(text).split(" a ")[1])/2),1):
                        pa.hotkey("volumeup")
                else:
                    for i in range(0,3,1):
                        pa.hotkey("volumeup")
            elif (str(text).lower().__contains__("alza") or str(text).lower().__contains__("aumenta"))and str(text).lower().__contains__("spotify"):
                pa.hotkey("win")
                pa.hotkey("win")
                pa.hotkey("win")
                t.sleep(0.2)
                pa.write("spotify")
                t.sleep(0.1)
                pa.hotkey("return")
                t.sleep(0.5)
                pa.hotkey("ctrl","up")
            elif (str(text).lower().__contains__("abbassa") or str(text).lower().__contains__("diminuisci")) and not str(text).lower().__contains__("spotify"):
                if str(text).__contains__(" di "):
                    for i in range(0,int(int(str(text).split(" di ")[1])/2),1):
                        pa.hotkey("volumedown")
                elif str(text).__contains__(" a "):
                    for i in range(0,int(int(str(text).split("a")[1])/2),1):
                        pa.hotkey("volumedown")
                else:
                    for i in range(0,3,1):
                        pa.hotkey("volumedown")
            elif (str(text).__contains__("abbassa") or str(text).__contains__("diminuisci"))and str(text).lower().__contains__("spotify"):
                pa.hotkey("win")
                pa.hotkey("win")
                pa.hotkey("win")
                t.sleep(0.2)
                pa.write("spotify")
                t.sleep(0.1)
                pa.hotkey("return")
                t.sleep(0.5)
                pa.hotkey("ctrl","down")
        elif (str(text).__contains__("imposta") or str(text).__contains__("avvia")) and str(text).__contains__("timer"):
            tick=0
            if str(text).__contains__(" da "):
                stringa=str(text).split("da")[1]
                if(stringa.__contains__("ora")and not stringa.__contains__("e")):
                    tick=3600
                elif stringa.__contains__("ore") and not stringa.__contains__(" e "):
                    tick=int(str(stringa).split("ore")[0])*3600
                elif stringa.__contains__("ore") and stringa.__contains__(" e "):
                    tick=(int(str(stringa).split("ore")[0])*3600)+(int(str(stringa).split("e")[1].split("minuti")[0])*60)
                elif stringa.__contains__("minuti") and stringa.__contains__(" e "):
                    tick=(int(str(stringa).split("minuti")[0])*60)+(int(str(stringa).split("e")[1].split("secondi")[0]))
                elif stringa.__contains__("minuti") and not stringa.__contains__(" e "):
                    tick=int(str(stringa).split("minuti")[0])*60
                elif stringa.__contains__("secondi") and not stringa.__contains__(" e "):
                    tick=int(str(stringa).split("secondi")[0])
                elif stringa.__contains__("ora") and stringa.__contains__(" e "):
                    tick=3600+(int(str(stringa).split("e")[1].split("minuti")[0])*60)
                elif stringa.__contains__("minuto") and stringa.__contains__(" e "):
                    tick=60+(int(str(stringa).split("e")[1].split("minuti")[0]))
                elif stringa.__contains__("minuto")and not stringa.__contains__("e"):
                    tick=60
                
            else:
                stringa=str(text).split("timer")[1]
                if(stringa.__contains__("ora")):
                    tick=3600
                elif stringa.__contains__("ore") and not stringa.__contains__(" e "):
                    tick=int(str(stringa).split("ore")[0])*3600
                elif stringa.__contains__("ore") and stringa.__contains__(" e "):
                    tick=(int(str(stringa).split("ore")[0])*3600)+(int(str(stringa).split("e")[1].split("minuti")[0])*60)
                elif stringa.__contains__("minuti") and stringa.__contains__(" e "):
                    tick=(int(str(stringa).split("minuti")[0])*60)+(int(str(stringa).split("e")[1].split("secondi")[0]))
                elif stringa.__contains__("minuti") and not stringa.__contains__(" e "):
                    tick=int(str(stringa).split("minuti")[0])*60
                elif stringa.__contains__("secondi") and not stringa.__contains__(" e "):
                    tick=int(str(stringa).split("secondi")[0])
                elif stringa.__contains__("ora") and stringa.__contains__(" e "):
                    tick=3600+(int(str(stringa).split("e")[1].split("minuti")[0])*60)
                elif stringa.__contains__("minuto") and stringa.__contains__(" e "):
                    tick=60+(int(str(stringa).split("e")[1].split("minuti")[0]))
                elif stringa.__contains__("minuto")and not stringa.__contains__("e"):
                    tick=60
            Speech("Timer Impostato a partire da ora")
            ts = Thread(target=tstart, args=(tick,))
            ts.start()
            

        elif str(text).__contains__("apri") or str(text).__contains__("avvia") or str(text).__contains__("lancia"):
            pa.hotkey("win")
            pa.hotkey("win")
            pa.hotkey("win")
            t.sleep(0.2)
            if str(text).__contains__("apri"):
                pa.write(str(text).split("apri")[1])
                #pa.hotkey("return")
                Speech("Ho aperto"+str(text).split("apri")[1])
            if str(text).__contains__("avvia"):
                pa.write(str(text).split("avvia")[1])
                #pa.hotkey("return")
                Speech("Ho avviato"+str(text).split("avvia")[1])
            if str(text).__contains__("lancia"):
                pa.write(str(text).split("lancia")[1])
                #pa.hotkey("return")
                Speech("Ho lanciato photoshop"+str(text).split("lancia")[1])
            pa.hotkey("return")
        elif str(text).__contains__("chiudi") or str(text).__contains__("termina"):
            pa.hotkey("altleft","f4")
            Speech("Ho chiuso la finestra corrente")
        elif str(text).__contains__("trovami") and str(text).__contains__("ragazza"):
            Speech("Scusami, ma non sono un mago.")
        elif str(text).__contains__("salva") and not str(text).__contains__("faccia"):
            if str(text).__contains__("come"):
                try:
                    split1=str(text).split("salva")[1]
                    try:
                        split2=split1.split("come")[0]
                        if(split2.lower().__contains__("video")):
                            win32clipboard.OpenClipboard()
                            clipboarditem = win32clipboard.GetClipboardData()
                            win32clipboard.CloseClipboard()
                            link = str(clipboarditem)
                            
                            video_object = YouTube(link, on_complete_callback = on_complete, on_progress_callback = on_progress)
                            Speech(f"Sto scaricando {video_object.title} in che qualità vuoi che venga scaricato Alta bassa o solo audio?")
                            chosingquality=True
                            
                            
                        else:
                            win32clipboard.OpenClipboard()
                            clipboarditem = win32clipboard.GetClipboardData()
                            win32clipboard.CloseClipboard()
                            if( not  os.path.isfile("DBSave.json")):

                                wr= open("DBSave.json","w")
                                data={}
                                data["SavedData"]=""
                                json.dump(data,wr)
                                wr.close()
                    
                            try:
                                content=open("DBSave.json")
                                db=json.load(content)
                                content.close()
                                dict1={}
                                dict2={}

                                for data in db["SavedData"]:
                                
                                    dict1[data]=db["SavedData"][data]
                                dict1[split1.split("come")[1]]={"data":split1.split("come")[0]}
                                dict2["SavedData"]=dict1
                                out_file=open("DBSave.json","w")
                                json.dump(dict2,out_file,indent=2)
                                out_file.close()
                            except:
                                Speech("Qualcosa è andato storto")




                    except Exception  as ex:
                        print(str(ex))
                        win32clipboard.OpenClipboard()
                        clipboarditem = win32clipboard.GetClipboardData()
                        win32clipboard.CloseClipboard()
                        print("contents "+clipboarditem)
                        if( not  os.path.isfile("DBSave.json")):
                            wr= open("DBSave.json","w")
                            data={}
                            data["SavedData"]=""
                            json.dump(data,wr)
                            wr.close()
                    
                        try:
                            content=open("DBSave.json")
                            db=json.load(content)
                            content.close()
                            dict1={}
                            dict2={}

                            for data in db["SavedData"]:
                                
                                dict1[data]=db["SavedData"][data]
                            dict1[split1.split("come")[1]]={"data":str(clipboarditem)}
                            dict2["SavedData"]=dict1
                            out_file=open("DBSave.json","w")
                            json.dump(dict2,out_file,indent=2)
                            out_file.close()
                        

                        except:
                            Speech("Qualcosa è andato storto richiedi")
                except:
                    Speech("Non ho capito")
        elif (( str(text).lower().__contains__("alta") or  str(text).lower().__contains__("media") or not str(text).lower().__contains__("bassa"))  and chosingquality):
                
            download_choice=str(text)
            chosingquality=False
            desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop') 
            if(not download_choice.lower().__contains__("alta") and not download_choice.lower().__contains__("media")and not download_choice.lower().__contains__("bassa")):
                Speech("Non ho capito.")
            elif download_choice.lower().__contains__("alta"):
                    
                video_object.streams.get_highest_resolution().download(desktop)
            elif download_choice.lower().__contains__("bassa"):
                video_object.streams.get_lowest_resolution().download(desktop)
            elif download_choice.lower().__contains__("audio"):
                video_object.streams.get_audio_only().download(desktop)
        elif str(text).lower().__contains__("ti va di parlare") or str(text).lower().__contains__("parliamo") or str(text).lower().__contains__("parlare")  or str(text).lower().__contains__("chicchieriamo"):
            cantalk=True
            Speech("Va bene , parliamo")
        elif str(text).lower().__contains__(f"ok {IANAME.lower()} alla prossima") or str(text).lower().__contains__(f"basta {IANAME.lower()}") or str(text).lower().__contains__(f"io vado {IANAME.lower()}"):
            cantalk=False
            Speech("Okay,alla prossima")
        elif str(text).lower().__contains__("quanto fa"):
        
            Speech("Fa "+str(eval_binary_expr(*text.split("quanto fa ")[1].split())))
        elif str(text).lower()=="vero" or str(text).lower()=="per" or str(text).lower()=="vero vero" :
            
            driver.find_element_by_xpath(f"//*[@id=\"risposta{kdomanda}_1\"]").click()
        elif str(text).lower()=="falso" or str(text).lower()=="pulse" or str(text).lower()=="falso falso":
            driver.find_element_by_xpath(f"//*[@id=\"risposta{kdomanda}_0\"]").click()
        elif str(text).lower()=="avanti":
            kdomanda+=1
            if kdomanda==31:
                kdomanda=1
            driver.find_element_by_xpath("//*[@id=\"modulo\"]/div[7]/div[2]/img[1]").click()
        elif str(text).lower()=="indietro":
            kdomanda-=1
            if kdomanda==0:
                kdomanda=30
            driver.find_element_by_xpath("//*[@id=\"modulo\"]/div[7]/div[2]/img[2]").click()
        
        elif str(text).lower()=="correggi":
            driver.find_element_by_xpath("//*[@id=\"modulo\"]/div[8]/div/input").click()
            t.sleep(1.5)
            
            
            if(int(driver.find_element_by_xpath("//div[@class=\"box_esito\"]/strong").text)<=3):
                Speech("Bravo, sei stato promosso continua così")
            else:
                Speech("Non va bene, riprova")
        elif str(text).lower().__contains__("salva la mia faccia"):
            Speech("Tu sei?")
            savingface=True
        elif savingface:
            savingface=False
            if(len(str(text).upper().split(" "))>0):
                if str(text).upper().__contains__("io"):
                    Speech("Okay Adesso acquisiro la tua faccia per riconoscerti in futuro")
                    RegUser(str(text).upper().split("io")[1].split("sono")[1])
                else:
                    Speech("Okay Adesso acquisiro la tua faccia per riconoscerti in futuro")
                    RegUser(str(text).upper())       
        elif str(text).lower().__contains__("chi sono io"):
            Speech("Un attimo fatti vedere")
            RecognizeFace();             
            

        elif str(text)!="" and cantalk:
            if  str(text).lower().__contains__("f{IANAME}"):
                bot_response = bot.get_response(str(text).split(f"{IANAME}")[1])
                Speech(str(bot_response))
            else:
                bot_response = bot.get_response(text)
                Speech(str(bot_response))

                    

                        
               
                   
                
                
                
                



                
            
                    
                    
                

 
        
                
            
    except Exception as e:
        Speech("Non ho capito bene")
        print("[ERROR]2"+str(e))


    
def Main():
    global notMusic
    notMusic=True
    while True:
        CheckCall(RecognizeMessage)
        t.sleep(0.02)



#Main()

notMusic=False
CheckCall(RecognizeMessage)
while True:
    print("")
    t.sleep(1)
