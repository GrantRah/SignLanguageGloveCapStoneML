import csv
index,middle,ring,pinky,thumb,sign,=0,0,0,0,0,"a"
i = 0

# This script makes a CSV file to store ASL finger position data
# First it creates the initial headers of the row 
# Then using a loop it will add new rows that have the data for each finger along with the actual sign
# Running this portion of the code will overwrite any existing ASLData.csv file
with open('ASLData.csv','w+') as f:
    writer=csv.writer(f)
    writer.writerow(['index','middle','ring','pinky','thumb','sign'])
    while i < 20:
        writer.writerow([index,middle,ring,pinky,thumb,sign])
        i+=1
        index+=1

#The following code appends new data to the CSV file
# Each time it is run it will add one new row to the existing ASLData.csv file
with open('ASLData.csv','a') as f:
    writer=csv.writer(f)
    
    writer.writerow([2,middle,ring,pinky,thumb,sign])
        
        