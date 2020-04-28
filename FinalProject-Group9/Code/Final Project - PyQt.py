#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import df, full_df_onehot_ST
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
full_df_onehot_ST = pd.read_csv("https://raw.githubusercontent.com/rachelorey/Final-Project-Group9/master/full_df_onehot_ST.csv")
df = pd.read_table('https://raw.githubusercontent.com/rachelorey/Final-Project-Group9/master/MITU0022_OUTPUT.tab')
full_df_onehot = pd.read_csv("https://raw.githubusercontent.com/rachelorey/Final-Project-Group9/master/full_df_onehot.csv")
demdata_EQ = pd.read_csv("https://raw.githubusercontent.com/rachelorey/Final-Project-Group9/master/demdata_EQ.csv")


# In[2]:


def maxwaitingtime(county,state,polling,pollworkers,service=2):
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    arrivals = df[["Q10"]].dropna()
    arrivals["Arrivals"] = np.array(arrivals["Q10"])/(arrivals["Q10"].sum())
    counts = arrivals[["Q10","Arrivals"]]
    counts = counts[counts["Q10"]<=18]
    counts = counts.groupby(["Q10"]).sum()
    counts.reset_index(inplace=True)
    counts = counts.sort_values(by=["Q10"])

    method = df[["Q4"]].dropna()
    method_counts = pd.DataFrame(method["Q4"].value_counts(normalize=True))
    percent_inperson = method_counts[method_counts.index == 1]["Q4"][1]    

    turnout = full_df_onehot[["turnout","STNAME"]]
    turnout = turnout[turnout["STNAME"]==state]
    pop = int(full_df_onehot[full_df_onehot["STNAME"]==state][full_df_onehot["CTYNAME"] == county]["TOT_POP"].iloc[0])
    pop = turnout.mode().iloc[0,0]*pop
    hourly = (pop * np.array(counts["Arrivals"]))/polling/pollworkers
    hourly = hourly * percent_inperson
    hourly = np.ceil(hourly)
    
    interarrival = np.full((len(hourly)),60)/hourly

    l = []
    arrival = 0
    finish = 0
    for persons in range(len(hourly)):
        wait = 0
        for i in range(int(hourly[persons])):
            if persons == 0:
                arrival = 0
            else:
                arrival += interarrival[persons]
            start = max(arrival,finish)
            finish = start + service
            wait += (start - arrival)
        average_wait = wait/hourly[persons]
        l.append([[(datetime.strptime("04:00 AM","%I:%M %p") + timedelta(hours=int(counts["Q10"].iloc[persons]))).time()],[average_wait]])
    waittime_daily = pd.DataFrame(l,columns = ["Hour","Average Wait"])
    return(waittime_daily[waittime_daily["Average Wait"]==waittime_daily["Average Wait"].max()[0]])


# In[3]:


def lines_pred(Density = 500,vtype ="EQ_DRE without paper trail",POC=.1):
    from sklearn import preprocessing, utils, neighbors, datasets, tree
    from sklearn.model_selection import train_test_split

    target = full_df_onehot["Q13"]
    features = full_df_onehot.drop(["Q13","STNAME","CTYNAME","TOT_POP","countyfips","turnout"],axis=1)
    import pandas as pd

    types = [["PercentHBAC",POC],
             ["dens2020",Density],
             ["EQ_DRE with and without paper trail",0],
             ["EQ_DRE without paper trail",0],
             ["EQ_Mail",0],
             ["EQ_Paper and DRE with and without paper trail",0],
             ["EQ_Paper and DRE with paper trail",0],
             ["EQ_Paper and DRE without paper trail",0],
             ["EQ_Paper ballot",0]]

    for i in range(len(types)):
        if types[i][0] == "EQ_" + vtype:
            types[i][1] = 1

    types= pd.DataFrame(types).transpose()
    types.columns = types.iloc[0]
    types.drop(df.index[0],inplace=True)

    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(features, target)

    A = {1:"No wait",
        2:"Wait under 10 minutes",
        3:"Wait between 10 and 30 minutes",
        4:"Wait between 30 minutes and one hour",
        5:"Wait over one hour"}


    X_train, X_test, y_train, y_test = train_test_split(features,target, test_size = 0.2, random_state = 1) 

    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(features,target)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix, classification_report

    def train_using_gini(X_train, X_test, y_train): 
        clf_gini = DecisionTreeClassifier(criterion = "gini", 
                random_state = 100,max_depth=3, min_samples_leaf=5) 

        clf_gini.fit(X_train, y_train) 
        return clf_gini 

    def tarin_using_entropy(X_train, X_test, y_train): 
        clf_entropy = DecisionTreeClassifier( 
                criterion = "entropy", random_state = 100, 
                max_depth = 3, min_samples_leaf = 5) 

        clf_entropy.fit(X_train, y_train) 
        return clf_entropy 

    def prediction(X_test, clf_object): 
        y_pred = clf_object.predict(X_test) 
        return(y_pred)
        print(y_pred)

    def cal_accuracy(y_test, y_pred): 
        return(accuracy_score(y_test,y_pred)*100) 

    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 

    
    y_pred_gini = prediction(X_test, clf_gini) 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    gini_acc = cal_accuracy(y_test, y_pred_gini) 
    ent_acc = cal_accuracy(y_test, y_pred_entropy) 
    return(A.get(prediction([types.iloc[0,:]],clf_entropy)[0]),str(round(ent_acc,2))+"%",A.get(prediction([types.iloc[0,:]],clf_gini)[0]),str(round(gini_acc,2))+"%")


# In[4]:


def lines_pred_ST(Density = 500,vtype ="EQ_DRE without paper trail",POC=.1,state="Nebraska"):
    from sklearn import preprocessing, utils, neighbors, datasets, tree
    from sklearn.model_selection import train_test_split    
    target = full_df_onehot_ST["Q13"]
    features = full_df_onehot_ST.drop(["Q13","CTYNAME","TOT_POP","countyfips","turnout"],axis=1)
    import pandas as pd

    #converting state input to feature style
    st = list()
    for i in list(full_df_onehot_ST.columns):
        if "ST_" in i:
            st.append(i)
    st = np.array(list(zip(st,[0]*len(st))))
    for i in range(len(st)):
        if state in st[i][0]:
            st[i][1] = 1

    st = st.transpose()
    st = pd.DataFrame(list(st),columns=st[0,:])
    st.drop(0,inplace=True)

    types = [["PercentHBAC",POC],
             ["dens2020",Density],
             ["EQ_DRE with and without paper trail",0],
             ["EQ_DRE without paper trail",0],
             ["EQ_Mail",0],
             ["EQ_Paper and DRE with and without paper trail",0],
             ["EQ_Paper and DRE with paper trail",0],
             ["EQ_Paper and DRE without paper trail",0],
             ["EQ_Paper ballot",0]]


    for i in range(len(types)):
        if types[i][0] == "EQ_" + vtype:
            types[i][1] = 1

    types= pd.DataFrame(types).transpose()
    types.columns = types.iloc[0]
    types.drop(df.index[0],inplace=True)

    types = types.merge(st,right_index=True,left_index=True)

    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(features, target)

    A = {1:"No wait",
        2:"Wait under 10 minutes",
        3:"Wait between 10 and 30 minutes",
        4:"Wait between 30 minutes and one hour",
        5:"Wait over one hour"}


    X_train, X_test, y_train, y_test = train_test_split(features,target, test_size = 0.2, random_state = 1) 

    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(features,target)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix, classification_report

    def train_using_gini(X_train, X_test, y_train): 
        clf_gini = DecisionTreeClassifier(criterion = "gini", 
                random_state = 100,max_depth=3, min_samples_leaf=5) 

        clf_gini.fit(X_train, y_train) 
        return clf_gini 

    def tarin_using_entropy(X_train, X_test, y_train): 
        clf_entropy = DecisionTreeClassifier( 
                criterion = "entropy", random_state = 100, 
                max_depth = 3, min_samples_leaf = 5) 

        clf_entropy.fit(X_train, y_train) 
        return clf_entropy 

    def prediction(X_test, clf_object): 
        y_pred = clf_object.predict(X_test) 
        return(y_pred)
        print(y_pred)

    def cal_accuracy(y_test, y_pred): 
        return(accuracy_score(y_test,y_pred)*100) 
    
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 

    y_pred_gini = prediction(X_test, clf_gini) 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    gini_acc = cal_accuracy(y_test, y_pred_gini) 
    ent_acc = cal_accuracy(y_test, y_pred_entropy) 
    return(A.get(prediction([types.iloc[0,:]],clf_entropy)[0]),str(round(ent_acc,2))+"%",A.get(prediction([types.iloc[0,:]],clf_gini)[0]),str(round(gini_acc,2))+"%")


# In[5]:


from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget,QMessageBox)
from PyQt5.QtGui import QIcon,QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)
        self.resize(600,700)
        self.originalPalette = QApplication.palette()

        self.createTopLeftGroupBox()
        mainLayout = QGridLayout()
        mainLayout.addWidget(self.topLeftGroupBox, 1, 0)        
        self.setLayout(mainLayout)
        self.setWindowTitle("Predicting Polling Place Line Length Based on Vote Taking Method and Population Density")
        self.setStyle(QStyleFactory.create("Fusion"))
        self.setWindowIcon(QIcon("C:\\Users\\rache\\iconx.ico"))
  
    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("User Inputs")
        layout = QVBoxLayout()
        
        #setting label for model 1
        model1 = QLabel("Predicting Wait Time with Decision Tree Classifier")
        model1.setFont(QFont("Calibri",weight=QFont.Bold,pointSize=15))
        layout.addWidget(model1)
        
        #creating radio button label for voting equipment type input
        RadioLabels = QLabel("Vote Taking Method")
        RadioLabels.setFont(QFont("Calibri",weight=QFont.Bold))
        layout.addWidget(RadioLabels)
        
        #creating radio buttons for voting equipment type input
        radioButton1 = QRadioButton("DRE with and without paper trail")
        radioButton2 = QRadioButton("DRE with paper trail")
        radioButton3 = QRadioButton("Mail")
        radioButton4 = QRadioButton("Paper and DRE with and without paper trail")
        radioButton5 = QRadioButton("Paper and DRE with paper trail")
        radioButton6 = QRadioButton("Paper with DRE without paper trail")
        radioButton7 = QRadioButton("Paper ballot")
        
        #adding radio buttons to layout
        layout.addWidget(radioButton1)
        layout.addWidget(radioButton2)
        layout.addWidget(radioButton3)
        layout.addWidget(radioButton4)
        layout.addWidget(radioButton5)
        layout.addWidget(radioButton6)
        layout.addWidget(radioButton7)
        
        #setting default radio button and EQ type
        radioButton5.setChecked(True)
        vtype = radioButton5.text()
        
        #getting text from clicked radio button
        radioButton1.toggled.connect(lambda:btnstate(radioButton1))
        radioButton2.toggled.connect(lambda:btnstate(radioButton2))
        radioButton3.toggled.connect(lambda:btnstate(radioButton3))
        radioButton4.toggled.connect(lambda:btnstate(radioButton4))
        radioButton5.toggled.connect(lambda:btnstate(radioButton5))
        radioButton6.toggled.connect(lambda:btnstate(radioButton6))
        radioButton7.toggled.connect(lambda:btnstate(radioButton7))
        
        #function to set vtype variable to clicked radio button
        def btnstate(button):
            vtype = button.text()  
            
        #creating state lineedit input
        state  = QLineEdit()
        state_title = QLabel("State")
        state_title.setFont(QFont("Calibri",weight=QFont.Bold))
        state.setPlaceholderText('Ex: Wisconsin') 
        layout.addWidget(state_title)
        layout.addWidget(state)
        
        #creating HBAC lineedit input
        HBAC  = QLineEdit()
        HBAC_title = QLabel("Percent Black and Hispanic Population")
        HBAC_title.setFont(QFont("Calibri",weight=QFont.Bold))
        HBAC.setPlaceholderText('Ex: .013') 
        layout.addWidget(HBAC_title)
        layout.addWidget(HBAC)
        
        #population density input
        lineEdit = QLineEdit()
        lineedit_title = QLabel("Population Density")
        lineedit_title.setFont(QFont("Calibri",weight=QFont.Bold))
        lineEdit.setPlaceholderText('Persons per Square Mile (County Level)') 
        layout.addWidget(lineedit_title)
        layout.addWidget(lineEdit)        
        
        #creating and adding pushbutton to layout to run clf prediction
        defaultPushButton = QPushButton("Calculate Anticipated Line Length")
        layout.addWidget(defaultPushButton)
        defaultPushButton.clicked.connect(lambda: OpenClick(Density=int(lineEdit.text()),vtype=vtype,POC=HBAC.text(),state=state.text()))
        
        #function to run on button click
        def OpenClick(Density,vtype,POC,state):
            popup = QMessageBox()
            ent,ent_acc,gini,gini_acc = lines_pred_ST(Density,vtype,POC,state)
            popup.setText("Entropy Prediction\nAnticipated Line Length Based Solely on Demographic and Survey Data:\n"+ent+"\n\nAccuracy: "+ent_acc+"\n\nGini Prediction\nAnticipated Line Length Based Solely on Demographic and Survey Data:\n"+gini+"\n\nAccuracy: "+gini_acc)
            popup.exec_()
        
        #queue model label
        model2 = QLabel("Predicting Wait Time with a Queue Model")
        model2.setFont(QFont("Calibri",weight=QFont.Bold,pointSize=15))
        layout.addWidget(model2)    
        
        #count line input
        lineEdit1 = QLineEdit()
        lineedit1_title = QLabel("County")
        lineedit1_title.setFont(QFont("Calibri",weight=QFont.Bold))
        lineEdit1.setPlaceholderText('Ex: San Diego County')
        layout.addWidget(lineedit1_title)
        layout.addWidget(lineEdit1)
        
        #state input
        lineEdit2 = QLineEdit()
        lineedit2_title = QLabel("State")
        lineedit2_title.setFont(QFont("Calibri",weight=QFont.Bold))
        lineEdit2.setPlaceholderText('Ex: California')
        layout.addWidget(lineedit2_title)
        layout.addWidget(lineEdit2)
        
        #polling places input
        lineEdit3 = QLineEdit()
        lineedit3_title = QLabel("Current Number of Polling Places")
        lineedit3_title.setFont(QFont("Calibri",weight=QFont.Bold))
        layout.addWidget(lineedit3_title)
        layout.addWidget(lineEdit3)
        
        #poll workers input
        lineEdit4 = QLineEdit()
        lineedit4_title = QLabel("Current Number of Poll Workers per Polling Place")
        lineedit4_title.setFont(QFont("Calibri",weight=QFont.Bold))
        layout.addWidget(lineedit4_title)
        layout.addWidget(lineEdit4)
        
        #button to run graph
        defaultPushButton1 = QPushButton("Graph My County")
        layout.addWidget(defaultPushButton1)    
        defaultPushButton1.clicked.connect(lambda: refresh())
        #function to run graph
        def refresh():
            chart = PlotCanvas(county=lineEdit1.text(),state=lineEdit2.text(),start_place=lineEdit3.text(),start_worker=lineEdit4.text())
            test = QDialog()
            test.resize(700,500)
            Layout = QGridLayout()
            Layout.addWidget(chart)
            test.setLayout(Layout)
            test.setWindowTitle("Results of Queue Model")
            test.exec()
                    
        #button to get optimal values
        optimalbutton = QPushButton("Calculate Optimal Poll Worker and Polling Place Levels")
        layout.addWidget(optimalbutton)    
        #optimal function
        def optimal(county,state,start_place,start_worker):
            popup = QMessageBox()
            start_place_init = start_place
            start_place = max(1,int(start_place) - 500)
            stop_place = start_place + 800
            start_worker_init = start_worker
            start_worker = max(1,int(start_worker) - 10)
            stop_worker = start_worker + 15
            l = []
            for places in range(int(start_place),int(stop_place),int((stop_place-start_place)/5)):
                for workers in range(int(start_worker),int(stop_worker),int((stop_worker-start_worker)/5)):
                    returns = maxwaitingtime(county,state,int(places),int(workers))
                    l.append([places,workers,returns.iloc[0,1][0]])

            l = pd.DataFrame(l,columns = ["Polling Places","Poll Workers","Max Waiting Time"])

            min_polling = l[l["Max Waiting Time"] <20]["Polling Places"].min()
            min_workers_conditional = l[l["Max Waiting Time"] <20][l["Polling Places"] == min_polling]["Poll Workers"].min()

            min_workers = l[l["Max Waiting Time"] <20]["Poll Workers"].min()
            min_polling_conditional = l[l["Max Waiting Time"] <20][l["Poll Workers"] == min_workers]["Polling Places"].min()
            popup.setText("Minimum Polling Places Needed to Keep Wait Time Under Twenty Minutes: "+str(min_polling)+". This requires "+str(min_workers_conditional)+" poll workers.\n\nMinimum Number of Poll Workers to Keep Wait Time Under Twenty Minutes: "+str(min_workers)+". This requires "+str(min_polling_conditional)+" polling places open.")
            popup.exec()
        #connecting optimal button to function
        optimalbutton.clicked.connect(lambda: optimal(county=lineEdit1.text(),state=lineEdit2.text(),start_place=lineEdit3.text(),start_worker=lineEdit4.text()))
        
        layout.addStretch(1)
        self.topLeftGroupBox.setLayout(layout)          

        
class PlotCanvas(FigureCanvas):

    def __init__(self,county,state,start_place=750,start_worker=15):
        fig = Figure(figsize=(5, 4), dpi=110)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(None)
        self.fig_and_min(county,state,start_place=750,start_worker=15)


    def fig_and_min(self,county,state,start_place=750,start_worker=15):
        start_place, start_worker = int(start_place), int(start_worker) 
        start_place_init = start_place
        start_place = int(max(1,start_place - 500))
        stop_place = int(start_place + 800)
        start_worker_init = start_worker
        start_worker = int(max(1,start_worker - 10))
        stop_worker = int(start_worker + 15)
        l = []
        for places in range(int(start_place),int(stop_place),int((stop_place-start_place)/5)):
            for workers in range(int(start_worker),int(stop_worker),int((stop_worker-start_worker)/5)):
                returns = maxwaitingtime(county,state,int(places),int(workers),4)
                l.append([places,workers,returns.iloc[0,1][0]])

        l = pd.DataFrame(l,columns = ["Polling Places","Poll Workers","Max Waiting Time"])
        
        ax = self.figure.add_subplot(111)
        groups = l.groupby("Poll Workers")
        for name, group in groups:
            ax.plot(group["Polling Places"], group["Max Waiting Time"], marker="o", linestyle="", label=(str(name)+" Poll Workers"))

        ax.set_xlabel("Number of Polling Places in "+str(county))
        ax.set_ylabel("Max Wait Time")
        ax.set_title("Wait Time by Polling Places and Poll Workers in "+str(county))
        ax.axhline(y=20, color='purple', linestyle='--',label="Twenty Minute Wait Time")
        ax.axvline(x=start_place_init,label="Initial Polling Places",linestyle=":")
        ax.plot(range(start_place,stop_place,int((stop_place-start_place)/5)),[start_worker_init]*5,linestyle="",marker="*",c="#008B8B",label="Initial Poll Workers")
        ax.legend()
        ax.set_xticks(range(start_place,stop_place,int((stop_place-start_place)/5)))

        min_polling = l[l["Max Waiting Time"] <20]["Polling Places"].min()
        min_workers_conditional = l[l["Max Waiting Time"] <20][l["Polling Places"] == min_polling]["Poll Workers"].min()

        min_workers = l[l["Max Waiting Time"] <20]["Poll Workers"].min()
        min_polling_conditional = l[l["Max Waiting Time"] <20][l["Poll Workers"] == min_workers]["Polling Places"].min()
        self.draw()
        
if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    sys.exit(app.exec_()) 

