from tkinter import *
from tkinter.filedialog import askopenfilename
import pandas as pd
 
class Window(Frame):
    def _init_(self, master=None):
        Frame._init_(self, master)
        self.master = master
        self.init_window()
 
 
    def init_window(self):
        self.master.title("TEST GUI")
        self.pack(fill=BOTH, expand=1)
 
        StopButton = Button(self, text="Stop", command=self.stop)
        GoButton = Button(self, text="Go", command=self.go)
 
        StopButton.place(x=25, y=25)
        GoButton.place(x=25, y=80)
 
    def stop(self):
        print('stop')
 
    def go(self):
        print('go')
 
Tk().withdraw()
filename = askopenfilename()
if filename == '':
    exit()
 
 
data = pd.read_excel(filename)
data['Stop_or_Go'] = ''
 
root = Tk()
root.geometry("400x300")
app = Window(root)
 
 
for index, series in data.iterrows():
    word_of_day = series['Word']
    #this is where I want the to be able to click Stop or Go
    data.at[index, 'Stop_or_Go'] = go()#GUI BUTTON RESPONSE
 
 
root.mainloop()