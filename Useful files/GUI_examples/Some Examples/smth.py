from tkinter import *


def doNothing():
    print('ok, ok...i won"t...')


root = Tk()

#creating dropDown menu

menu1 = Menu(root)
root.config(menu = menu1)

subMenu = Menu(menu1)
menu1.add_cascade(label = 'File', menu = subMenu)
subMenu.add_command(label = 'Do Nothing',command = doNothing)
subMenu.add_command(label = 'Save File',command = doNothing)
subMenu.add_separator()
subMenu.add_command(label = 'Exit',command = doNothing)

subMenu2 = Menu(menu1)
menu1.add_cascade(label = 'Edit', menu = subMenu2)
subMenu2.add_command(label = 'Do Nothing',command = doNothing)
subMenu2.add_command(label = 'Do Everything',command = doNothing)


#creating a toolbar

toolbar = Frame(root, bg ='lightgrey')
insertSmth = Button(toolbar, text = 'Insert Image', command = doNothing,relief = FLAT)
printButton = Button(toolbar,text = 'Print',command = doNothing)

printButton.pack(side = LEFT,padx = 2, pady =2) 
insertSmth.pack(side = LEFT,padx = 2, pady =2) 

toolbar.pack(side = TOP, fill = X)

#status bar 

status = Label(root, text = 'Bobbig', bd = 1, relief = GROOVE, anchor = W , padx = 5, pady = 5) #bd - border


status.pack(side = BOTTOM, fill = X)
root.mainloop()