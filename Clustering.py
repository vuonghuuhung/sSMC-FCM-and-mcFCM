import mcfcm
import ssmcfcm
import MetricsCalculate


from tkinter.ttk import *
from tkinter import *
import tkinter.scrolledtext as sctxt
from PIL import ImageTk, Image

#Open window in Tkinter
win = Tk()
win.title('Clustering MC-FCM, sSMC-FCM')
win.geometry('1050x600')
win['bg'] = '#90ee90'
# win.attributes('-topmost', True)  # the window is always visible even if you switch to another page


#Tạo label
font_ = ('Times New Roman',16)
title_ = 'Ứng dụng kiểm tra sự ảnh hưởng của các tham số mờ hóa trong thuật toán phân cụm mờ'
name = Label(win, text = title_, font=font_,width=80,relief='raised')
name.place(relx=0.1,rely=0.05, relheight=0.05,relwidth=0.8)

Label(win, text = 'Chọn thuật toán').place(relx=0.02,rely=0.12,relheight=0.03,relwidth=0.13)

algorithm = Combobox(win, width = 20)
algorithm['value'] = ('MC-FCM', 'sSMC-FCM')
algorithm.current(0)
algorithm.place(relx=0.17, rely = 0.12, relheight=0.03,relwidth=0.08)

result = Label(win,text = 'Kết quả',font=13,relief='raised')
result.place(relx=0.3,rely=0.12,relheight=0.03, relwidth=0.63)

hLabel = Scrollbar(win,orient=HORIZONTAL)
resultLabel = sctxt.ScrolledText(win, wrap = 'none', xscrollcommand = hLabel.set)
resultLabel.place(relx=0.3,rely=0.17,relheight=0.13,relwidth=0.63)
hLabel['command'] = resultLabel.xview
hLabel.place(relheight=0.02,relx=0.3,rely=0.30,relwidth=0.63)

hData = Scrollbar(win,orient=HORIZONTAL)
resultData = sctxt.ScrolledText(win, wrap = 'none', xscrollcommand = hData.set)
resultData.place(relx=0.3,rely=0.33,relheight=0.3,relwidth=0.63)
hData['command'] = resultData.xview
hData.place(relheight=0.03,relx=0.3,rely=0.63,relwidth=0.63)

score = Label(win,text='Độ đo hiệu năng',font = 13,relief='raised')
score.place(relx=0.3,rely=0.68,relheight=0.03, relwidth=0.63)

resultScore = sctxt.ScrolledText(win, wrap = 'none')
resultScore.place(relx=0.3,rely=0.73,relheight=0.2,relwidth=0.63)





def runMCFCM():
    dataname = mcdata_.get()
    mL = float(mc_mL_.get())
    mU = float(mc_mU_.get())
    if mL<mU and mL>1:
        data,centre,labels,clus_label = mcfcm.mcfcm(dataname,mL,mU)

        resultLabel.delete(1.0,END)
        resultData.delete(1.0,END)
        for i in range(len(centre)):
            for j in range(len(centre[0])):
                centre[i][j] = round(centre[i][j],2)

        resultLabel.insert(END,'Cụm  ||  Số điểm thuộc cụm  ||  Thông tin\n')
        for i in range(len(centre)):
            nums = clus_label.count(i)
            resultLabel.insert(END,'Cụm '+str(i)+'  ||  ' +str(nums)+'   ||  '+str(centre[i])+'\n')

        for i in range(len(data)):
            for j in range(len(data[0])):
                data[i][j] = round(data[i][j],2)

        resultData.insert(END,'POINT  ||  Nhãn phân cụm  ||  Nhãn dữ liệu   ||   Dữ liệu\n')
        for i in range(len(data)):
            resultData.insert(END,'Point '+str(i)+'   ||   '+ str(clus_label[i])+'   ||   '+str(labels[i])+'   ||   '+str(data[i])+'\n')

        resultScore.delete(1.0,END)
        list_metrics = MetricsCalculate.getmetrics(labels,clus_label)
        resultScore.insert(END,'Accuracy: '+str(list_metrics[0])+'\n')
        resultScore.insert(END,'RandIndex: '+str(list_metrics[1])+'\n')

    else:
        resultData.delete(1.0,END)
        resultData.insert(END,'Điều kiện: 1<mL<mU')

    return
def runSSMC():
    dataname = ssmcdata_.get()
    mL = float(ssmc_mL_.get())
    mU = float(ssmc_mU_.get())
    sup = int(ssmc_sup_.get())
    if mL<mU and mL>1:
        data,centre,labels,clus_label,supervise = ssmcfcm.ssmcfcm(dataname,mL,mU,sup)

        resultLabel.delete(1.0,END)
        resultData.delete(1.0,END)
        for i in range(len(centre)):
            for j in range(len(centre[0])):
                centre[i][j] = round(centre[i][j],2)

        resultLabel.insert(END,'Cụm  ||  Số điểm thuộc cụm  ||  Thông tin\n')
        for i in range(len(centre)):
            nums = clus_label.count(i)
            resultLabel.insert(END,'Cụm '+str(i)+'  ||  ' +str(nums)+'   ||  '+str(centre[i])+'\n')

        for i in range(len(data)):
            for j in range(len(data[0])):
                data[i][j] = round(data[i][j],2)

        resultData.insert(END,'POINT  ||   Giám sát   ||   Nhãn phân cụm  ||  Nhãn dữ liệu   ||   Dữ liệu\n')
        for i in range(len(data)):
            resultData.insert(END,'Point '+str(i)+'   ||   '+str(supervise[i])+'   ||   '+ str(clus_label[i])+'   ||   '+str(labels[i])+'   ||   '+str(data[i])+'\n')


        resultScore.delete(1.0,END)
        list_metrics = MetricsCalculate.getmetrics(labels,clus_label)
        resultScore.insert(END,'Accuracy: '+str(list_metrics[0])+'\n')
        resultScore.insert(END,'RandIndex: '+str(list_metrics[1])+'\n')

    else:
        resultData.delete(1.0,END)
        resultData.insert(END,'Điều kiện: 1<mL<mU')
    return

#MCFCM INPUT
mcdata = Label(win, text='Chọn dữ liệu')
mcdata_ = Combobox(win)
mcdata_['value'] = ('iris','wine','heart','glass')
mcdata_.current(0)


mc_mL = Label(win, text='Nhập tham số mL')
mc_mL_ = Entry(win)


mc_mU = Label(win, text='Nhập tham số mU')
mc_mU_ = Entry(win)

RunMC = Button(win, text='Run',command=runMCFCM)


#SSMC INPUT
ssmcdata = Label(win, text='Chọn dữ liệu')
Frame(ssmcdata).pack(padx = 70, pady =8)
ssmcdata_ = Combobox(win, width = 20)
ssmcdata_['value'] = ('iris','wine','heart','glass')
ssmcdata_.current(0)


ssmc_mL = Label(win, text='Nhập tham số M')
Frame(ssmc_mL).pack(padx=70, pady=8)
ssmc_mL_ = Entry(win, width=20)


ssmc_mU = Label(win, text='Nhập tham số M\'')
Frame(ssmc_mU).pack(padx=70, pady=8)
ssmc_mU_ = Entry(win, width=20)

ssmc_sup = Label(win, text= 'Phần trăm giám sát')
Frame(ssmc_sup).pack(padx=70, pady=8)
ssmc_sup_ = Entry(win, width=20)

RunSSMC = Button(win, text='Run',command=runSSMC)

# Tạo button
def clickButton() :
    if algorithm.get() == 'MC-FCM':

        ssmcdata.place_forget()
        ssmcdata_.place_forget()
        ssmc_mL.place_forget()
        ssmc_mL_.place_forget()
        ssmc_mU.place_forget()
        ssmc_mU_.place_forget()
        ssmc_sup.place_forget()
        ssmc_sup_.place_forget()
        RunSSMC.place_forget()

        mcdata.place(relx = 0.02, rely=0.35,relheight=0.04,relwidth=0.12)
        mcdata_.place(relx=0.15, rely=0.35,relheight=0.04,relwidth=0.12)
        mc_mL.place(relx = 0.02, rely=0.42,relheight=0.04,relwidth=0.12)
        mc_mL_.place(relx = 0.15, rely=0.42,relheight=0.04,relwidth=0.12)
        mc_mU.place(relx = 0.02, rely=0.50,relheight=0.04,relwidth=0.12)
        mc_mU_.place(relx = 0.15, rely=0.50,relheight=0.04,relwidth=0.12)
        RunMC.place(relx = 0.23, rely=0.60,relheight=0.04,relwidth=0.04)


    else:
        mcdata.place_forget()
        mcdata_.place_forget()
        mc_mL.place_forget()
        mc_mL_.place_forget()
        mc_mU.place_forget()
        mc_mU_.place_forget()
        RunMC.place_forget()

        ssmcdata.place(relx = 0.02, rely=0.35,relheight=0.04,relwidth=0.12)
        ssmcdata_.place(relx=0.15, rely=0.35,relheight=0.04,relwidth=0.12)
        ssmc_mL.place(relx = 0.02, rely=0.42,relheight=0.04,relwidth=0.12)
        ssmc_mL_.place(relx = 0.15, rely=0.42,relheight=0.04,relwidth=0.12)
        ssmc_mU.place(relx = 0.02, rely=0.50,relheight=0.04,relwidth=0.12)
        ssmc_mU_.place(relx = 0.15, rely=0.50,relheight=0.04,relwidth=0.12)
        ssmc_sup.place(relx = 0.02, rely=0.57,relheight=0.04,relwidth=0.12)
        ssmc_sup_.place(relx = 0.15, rely=0.57,relheight=0.04,relwidth=0.12)
        RunSSMC.place(relx = 0.23, rely=0.65,relheight=0.04,relwidth=0.04)
   


button = Button(win, text = 'Tiếp tục', command = clickButton)
button.place(relx=0.2, rely =0.2)


# display interface
win.mainloop()
