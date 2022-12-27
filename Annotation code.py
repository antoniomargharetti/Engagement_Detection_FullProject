import os
import csv
# import sys

path = os.path.join("C:\Engagement_Project", "project", "Labels", "AllLabels.csv")
temp = {}
with open(path, newline='') as f:
    rd = csv.DictReader(f)
    for row in rd:
        if row['Engagement'] == '0' or row['Engagement'] == '1' :
            temp[row['ClipID']] = '0'
        elif row['Engagement']=='2':
            temp[row['ClipID']] = '1'
        elif row['Engagement']=='3':
            temp[row['ClipID']] = '2'

with open('Final_annotated_labels.csv', 'w', newline='') as g:
    csvwriter = csv.writer(g)
    csvwriter.writerow(['ClipID', 'Engagement'])
    for i in temp:
        csvwriter.writerow([i, temp[i]])







