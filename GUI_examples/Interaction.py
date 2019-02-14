from tkinter import *

root = Tk()

#one way
def printname():
    print('Hello World!')

#another way
def printname2(event):
    print('Hello World 2!')



button_1 = Button(root, text = 'Print', command = printname )
button_2 = Button(root, text = 'Print2' )

button_1.pack() 
button_2.bind('<Button-1>', printname2) #LMB pressed  
button_2.pack() 






root.mainloop()