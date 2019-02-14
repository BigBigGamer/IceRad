from tkinter import *

class my_buttons:

    def __init__(self, master):
        frame = Frame(master)
        frame.pack()
        self.printButton = Button(frame,text = 'Text print',command = self.printMessage)
        self.printButton.pack(side = LEFT)

        self.quitButton = Button(frame,text = 'Quit',command = frame.quit)
        self.quitButton.pack(side = LEFT)


    def printMessage(self):
        print('Hello World!')



root = Tk()

b = my_buttons(root)

root.mainloop()