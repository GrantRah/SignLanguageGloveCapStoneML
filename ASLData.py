#Important to note that panda could be used for this task
# If this doesn't work then I'll probably try pandas

import csv
import random
index,middle,ring,pinky,thumb,sign,=0,0,0,0,0,"a"
i = 0
#____________________________________________________________________________________________________________________
# This script makes a CSV file to store ASL finger position data
# First it creates the initial headers of the row 
# Then using a loop it will add new rows that have the data for each finger along with the actual sign
# Running this portion of the code will overwrite any existing ASLData.csv file
with open('ASLData.csv','w+') as f:
    writer=csv.writer(f)
    writer.writerow([ "index" , "middle","ring","pinky","thumb","sign"])
    #while i < 20:
        #writer.writerow([index,middle,ring,pinky,thumb,sign])
        #i+=1
        #index+=1
#____________________________________________________________________________________________________________________
#The following code appends new data to the CSV file
# Each time it is run it will add one new row to the existing ASLData.csv file
#with open('ASLData.csv','a') as f:
    #writer=csv.writer(f)
    
    #writer.writerow([2,middle,ring,pinky,thumb,sign])

#____________________________________________________________________________________________________________________        
 # Gonna use  rng to generate random data for testing purposes 
 # for the purpose of this test I will only use 3 signs: A,L and I due to there simplicity of their finger positions
 # Each sign will have a specific range of values for each finger
 # For a bent finger the range will be 25 - 45
 # For a straight finger the range will be 0 - 15
with open('ASLData.csv','a') as f:
    writer=csv.writer(f)
#while loop handles the number of rows to be added
    while i < 500: 
        sign = random.choice(['A','L','I'])
        if sign == 'A':
            index = random.randint(25,45)
            middle = random.randint(25,45)
            ring = random.randint(25,45)
            pinky = random.randint(25,45)
            thumb = random.randint(0,15)
        elif sign == 'L':
            index = random.randint(0,15)
            middle = random.randint(25,45)
            ring = random.randint(25,45)
            pinky = random.randint(25,45)
            thumb = random.randint(0,15)
        else: # sign == 'I'
            index = random.randint(25,45)
            middle = random.randint(25,45)
            ring = random.randint(25,45)
            pinky = random.randint(0,15)
            thumb = random.randint(0,15)
        writer.writerow([2,middle,ring,pinky,thumb,sign])
        i += 1