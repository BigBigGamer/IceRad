from tkinter import *

root = Tk()

def lclick(event):
    print('left')

def mclick(event):
    print('right')

def rclick(event):
    print('middle')

frame = Frame(root, width = 400, height = 300)

frame.bind('<Button-1>',lclick)
frame.bind('<Button-2>',rclick)
frame.bind('<Button-3>',mclick)

frame.pack()



root.mainloop()