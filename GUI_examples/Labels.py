#import tkinter
from tkinter import*

root = Tk() #create blank window, Tk - internall class

one = Label(root,text = 'One', bg ='red')
two = Label(root,text = 'Two', bg = 'yellow')
three = Label(root,text = 'Three', bg = 'green')

one.pack()
two.pack(fill = X) #fill the x direction
three.pack(side = LEFT,fill = Y) #fill the y direction

root.mainloop() #show the main window

