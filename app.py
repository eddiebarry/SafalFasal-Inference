import datetime
#import modelpth
#from modelpth import Net
import os
from flask import Flask , render_template , url_for , request
import base64
from flask_mysqldb import MySQL
import struct
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader
from torchvision import models,datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time
import cv2
from torch.autograd import Variable


app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'HELLO'

mysql = MySQL(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


Name_dict = {0: 'Cabbage',
1: 'Kale' ,
2: 'apple' ,
3: 'apricot',
4 :'avacado',
5 :'banana',
6 :'bean',
7 :'beetroot',
8 :'berry',
9 :'brocolli',
10 :'carrot',
11:'cauliflower',
12 :'cherry',
13 :'chilli',
14 :'coconut',
15 :'corn',
16 :'cucumber',
17 :'date',
18 :'fig',
19 :'garlic',
20 :'grapes',
21 :'guava',
22 :'kiwi',
23 :'lemon',
24 :'lettuce',
25 :'mango',
26 :'melon',
27 :'mulberry',
28 :'mushroom',
29 :'onion',
30 :'orange',
31 :'pea',
32 :'pear',
33:'pineapple',
34:'plum',
35: 'pomegranate',
36 :'pomelo',
37 :'potato',
38 :'pumpkin',
39 :'quince',
40 :'radish',
41 :'spinach',
42 :'tamarind',
43 :'tomato',
44 :'turnip',
45 :'watermelon',
46 :'yam'}

net = models.vgg19()
new_classifier = net.classifier[:-1]
new_classifier.add_module('fc' , nn.Linear(4096 , 47))
net.classifier = new_classifier
net.load_state_dict(torch.load('vgg.ckpt', map_location = 'cpu') )

def pred(filepath):
	print('Hello')
	
	

	img = Image.open(filepath)
	transform_pipeline = transforms.Compose([transforms.Resize((224 , 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
	img = transform_pipeline(img)
	img = img.unsqueeze(0)
	img = Variable(img)
	prediction = net(img)
	_ , pred = torch.max(prediction , 1)
	name = Name_dict[pred.item()]
	print(name)
	return str(name)
	

'''
	with torch.no_grad():
		imgpath = filepath
		img = cv2.imread(imgpath)
		img = cv2.resize(img ,(224 , 224))
		print(img.size)
		img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
		img = torch.tensor(img , dtype= torch.float32) 
		img = img.reshape(1 , 3 , 224 , 224)
		output =net(img)
		_ , pred = torch.max(output , 1)
		#name = Name_dict[pred.item()]		
		print(pred.item())
		
	return str(pred.item())


'''





@app.route("/")
def index():
   return render_template("upload.html")

@app.route('/info')
def info():
    cur = mysql.connection.cursor()
    
    resultValue = cur.execute("SELECT * FROM crop")
    if resultValue >= 0:
        user_det = cur.fetchall()
        return render_template('user.html' , user_det = user_det)
    
     
@app.route("/upload" , methods = ['POST'])
def upload():
       uploaded_files = request.files.getlist("file")
       print(len(uploaded_files))
       target = os.path.join(APP_ROOT , 'images/')
       print(target)
       
       if not os.path.isdir(target):
          os.mkdir(target)
      

       for file in uploaded_files:
         print(file)
         print("hello")
         filename = file.filename
         destination = "".join([target , filename])
         print(destination)
         file.save(destination)
       return render_template("complete.html")

@app.route("/processjson" , methods = ['POST'])
def processjson():
     
     data = request.get_json()
     print(request.is_json)
     imgdata = base64.b64decode(data['image'])
     now  = datetime.datetime.now()
     date_current = str(now)
     image_name = 'img'+ date_current + '.jpg'
     filename = image_name
     if not os.path.exists('INPUT_IMAGES'):
           os.makedirs('INPUT_IMAGES')
      
     folder_name = str('INPUT_IMAGES/')
     imagepath = os.path.join(folder_name,filename) 
     with open(os.path.join(folder_name,filename), 'wb') as f:
        f.write(imgdata)
     print(data['image'])
     
     region = data['name']
     if region == "":
         region = 'Region Not specified'
     print("hello")
     cur = mysql.connection.cursor()
     cur.execute("INSERT INTO crop (TIMESTAMP ,FILENAME , REGION) VALUES(%s ,%s , %s)",(date_current,image_name , region))
     mysql.connection.commit()
     cur.close()
     response = pred(imagepath)
     
     return "{'response':'"+response+"'}" 

if __name__ == "__main__":
   app.run(debug = True , host = '0.0.0.0')
