#import tkinter
from tkinter import*

root = Tk() #create blank window, Tk - internall class

theLabel = Label(root,text = 'This is too easy') #Create Label class object in root window with text
theLabel.pack() #place object anywhere

topFrame = Frame(root) #Frame inside main root window 
topFrame.pack() #put in  

botFrame = Frame(root) #Frame inside main root window 
botFrame.pack(side = BOTTOM) #To where to put the frame

#Buttons!:
button1 = Button(topFrame,text = 'Button 1', fg ='red' ) #button where/text/*
button2 = Button(topFrame,text = 'Button 2')
button3 = Button(topFrame,text = 'Button 3')
button4 = Button(botFrame,text = 'Button 4')



button1.pack(side = LEFT) #show dem buttons
button2.pack(side = LEFT)
button3.pack(side = LEFT)
button4.pack()

root.mainloop() #show the main window

