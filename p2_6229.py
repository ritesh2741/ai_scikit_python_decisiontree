import numpy as np
import csv
import time
import os
import pdb
import itertools
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix

class DecisionTreeData(dict):
    
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

def welcome_screen():
  print("\n ***************************************************************** \n")
  print("              DECISION TREE    ")
  print("\n ***************************************************************** \n")
  print("     1 : Learn a decision tree or load an existing tree\n")
  print("     2 : Testing accuracy of the decision tree\n")
  print("     3 : Applying the decision tree to new cases\n")
  print("     4 : Exit \n")
  print("\n ***************************************************************** \n")

def sub_menu():
  print("     3 : Applying the decision tree to new cases\n")
  print("     \t\t 3.1 : Enter a new case \n")
  print("     \t\t 3.2 : Exit \n")

def exit_message():
  print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\t Thank You ! ! ")
  print("\n\n\n\n\n\n\n\n\n\n\n\n")
  exit()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def load_tree():
  datafile_path = input('Please give a load file: ')
  loaded_file = os.getcwd() + '/'+ str(datafile_path)
  try:
    with open(loaded_file) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        #number of samples is read from first line first word
        n_samples = int(temp[0])

        #number of features available
        n_features = int(temp[1])

        #class names available in first line staring second word
        target_names = np.array(temp[2:])

        #create 2D data array of size n_samples * n_features
        data = np.empty((n_samples, n_features))
        
        #create an array of size n_samples * n_features
        target = np.empty((n_samples,), dtype=np.int)

        #iterate over remaining data and fill in data and target arrays
        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)
    
    print("\n\n\tData Loaded")
    time.sleep(2)
    return DecisionTreeData(data=data, target=target,
                 target_names=target_names,
                 feature_names=['sepal length (cm)', 'sepal width (cm)',
                                'petal length (cm)', 'petal width (cm)'])

  except  :
    print('\n\n\n\n\n\n\nFile Not Found / Invalid File. Please Try Again Please make sure your data set is in the directory that this program is in and is in valid format.\n\n')
    time.sleep(2)

def new_case(clf,values,target_names):

  val = [float(s) for s in values.split(',')]
  return(target_names[clf.predict([val])])

def accuracy(clf,target_names):

  test_data = load_tree()
  y_test = test_data.target.tolist()
  y_pred = []
  X = test_data.data
  for i in range (0, len(test_data.target)):
    y_pred.append(clf.predict([X[i]]))


  # # Split the data into a training set and a test set
  # classifier = svm.SVC(kernel='linear', C=0.01)
  # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

  # Run classifier, using a model that is too regularized (C too low) to see
  # the impact on the results
  # y_pred = classifier.fit(X_train, y_train).predict(X_test)


  #Compute Confusion Matrix
  cnf_matrix = confusion_matrix(y_test, y_pred)
  np.set_printoptions(precision=2)

  # Plot non-normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=target_names,title='Confusion matrix, without normalization')

  # Plot normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,title='Normalized confusion matrix')

  plt.show()

def main():
  load_count = 0
  while True:
    welcome_screen()
    option = input('Please choose an option:  ')
    if option == '1':
      iris = load_tree()
      try:
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(iris.data, iris.target)
        target_names = iris.target_names
        #Drawing Decision Tree Data in a dot file.
        with open(os.getcwd() + "/"+"figure.dot", 'w') as f:
          f = tree.export_graphviz(clf, out_file=f)
        load_count = 1
      except:
        continue

    elif option == '2':
      if load_count == 1:
        accuracy(clf,target_names)
      else:
        print('\n\n\n\n\n\n\n\t Please Load a file first \n\n')
        time.sleep(2)

    elif option == '3':
      while True:
        if load_count == 1:
          sub_menu()
          sub_menu_option = input('Please choose an option:  ')
          if sub_menu_option == '3.1'or sub_menu_option == '1':
            output = []
            ip = 1    
            while True:
              values = input("Please Enter new case separated by ','  ")
              result = new_case(clf,values,target_names)
              output.append("Case "+str(ip)+" => "+ str(result[0]))
              print("\n\tResult  => " + str(result[0]) + "\n\n")
              time.sleep(1)
              add = input ("Do you want to add more?").lower()
              ip = ip + 1
              if add != 'y':
                print ("Results are : ") 
                print (output) 
                print ("\n\n\n")
                break

          elif sub_menu_option == '3.2' or sub_menu_option == '2':
            exit_message()
          else:
            print( "\n\n\n\n\n Invalid Option Please Try again \n\n")
        else:
          print('\n\n\n\n\n\n\n\t Please Load a file first \n\n')
          time.sleep(2)   
          break
    elif option == '4':
      exit_message()
    else:
      print( "\n\n\n\n\n Invalid Option Please Try again \n\n")

if __name__ == "__main__":
  main()
