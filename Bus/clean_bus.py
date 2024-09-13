import pandas as pd
##bus_vehicleNum
def clean_source(data):
    ##data = pd.read_csv('busgps20070223.txt', header=None)
    txt = pd.DataFrame({'busNum': data[1],
                        'vehicleNum': data[2],
                        'lng': data[6],
                        'lat': data[7],
                        'time': data[12]
                        })
    txt.to_csv('busgps.txt')
def split_byvehicleNum(data):
    ##group_names[2286]=NAN
    ##data = pd.read_csv('busgps.txt')
    group_names = data["vehicleNum"].unique()
    for group_name in group_names:
        data_group = data.groupby(by=['vehicleNum']).get_group(group_name).reset_index(drop=True)

        date_sortbytime = data_group.sort_values(by='time')
        txt = pd.DataFrame({'busNum': date_sortbytime['busNum'],
                            'vehicleNum': date_sortbytime['vehicleNum'],
                            'lng': date_sortbytime['lng'],
                            'lat': date_sortbytime['lat'],
                            'time': date_sortbytime['time']
                            })
        txt.to_csv('bus_vehicleNum/' + group_name + '.txt')

if __name__ == '__main__':
    data = pd.read_csv('busgps.txt')
    group_names = data["vehicleNum"].unique()
    for i in range(2287, 2461):
        data_group = data.groupby(by=['vehicleNum']).get_group(group_names[i]).reset_index(drop=True)
        date_sortbytime = data_group.sort_values(by='time')
        txt = pd.DataFrame({'busNum': date_sortbytime['busNum'],
                            'vehicleNum': date_sortbytime['vehicleNum'],
                            'lng': date_sortbytime['lng'],
                            'lat': date_sortbytime['lat'],
                            'time': date_sortbytime['time']
                            })
        txt.to_csv('bus_vehicleNum/' + group_names[i] + '.txt')