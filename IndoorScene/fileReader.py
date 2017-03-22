import os
import shutil

os.chdir('C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\validation')

for efile in os.listdir(os.curdir):
  filename=os.path.basename(efile)
  if "Corridor" in filename:
     shutil.copy(efile,'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\co')
  if "Elevator" in filename:
     shutil.copy(efile,'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\el')
  if "MeetingRoom" in filename:
     shutil.copy(efile,'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\mr')
  if "LargeOffice" in filename:
     shutil.copy(efile,'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\lo')
  if "LargeMeetingRoom" in filename:
     shutil.copy(efile,'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\lmr')
  if "PrinterArea" in filename:
     shutil.copy(efile,'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\pa')
  if "SmallOffice" in filename:
     shutil.copy(efile,'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\so')
  if "Toilet" in filename:
     shutil.copy(efile,'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\to')
  if "RecycleArea" in filename:
     shutil.copy(efile,'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\ra')
  
  