import csv
# This script makes a CSV file to store ASL finger position data
# First it creates the initial headers of the row 
# Then using a loop it will add new rows that have the data for each finger along with the actual sign
with open('ASLData.csv','w+') as f:
    writer=csv.writer(f)
    writer.writerow(['index','middle','ring','pinky','thumb','sign'])