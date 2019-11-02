import csv

def lessen_data(path):
    points = []
    counter = 0
    add_data = True
    with open(path) as csv_file:
        readCSV = csv.reader(csv_file, delimiter=';')
        for row in readCSV:
            if counter == 250:
                add_data = not add_data
                counter = 0
            if add_data:
                points.append(row)
            counter+=1
    with open(path[:-4]+"-small.csv", 'w+',newline='') as output_file:
        writeCSV = csv.writer(output_file,delimiter=';')
        writeCSV.writerows(points)

lessen_data('boxes.csv')