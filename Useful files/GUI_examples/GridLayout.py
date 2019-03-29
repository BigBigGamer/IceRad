from tkinter import *

root = Tk()

label_1 = Label(root,text = 'Name')
label_2 = Label(root,text = 'Password')
entry_1 = Entry(root) # field to put text in
entry_2 = Entry(root)


label_1.grid(row = 0, sticky = E) #the grid layout config ? stiky = E is for stiking the right side 
label_2.grid(row = 1, sticky = E)

entry_1.grid(row = 0, column = 1)
entry_2.grid(row = 1, column = 1)

#Checkbutton
c = Checkbutton(root,text = 'Keep me logged in')
c.grid(columnspan = 2, row = 2) #take 2 cells

root.mainloop()