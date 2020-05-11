from tkinter import *

import pandas as pd
import numpy as np

window=Tk()
window.title("Project")
window.geometry('1280x800')

def read():
    df = pd.read_csv('spam.csv',delimiter=',',encoding='latin-1')
    f=df.head()
    f.pack()

    



btn1 = Button(window,text="Dataset" ,command=read).place(x=200,y=50)

btn2 = Button(window,text="Dataset-Graph").place(x=200,y=150)

btn3 = Button(window,text="Analyze").place(x=200,y=250)











window.mainloop()