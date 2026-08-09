import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from time import sleep



while True:

    dates = []
    times = []
    co2_values = []
    temp_values = []
    rh_values = []

    # Read the CSV file
    with open("/media/envisense/DATALOG.CSV", 'r', encoding='latin1') as file:
        
        for i, line in enumerate(file):
            values = line.strip().split(',')
            
            if len(values) == 5:
                if i < 5 or values[1] == "00:00":
                    continue  # Skip the header row
                # Parse values as floats
                date_str, time_str, co2_str, temp_str, rh_str = values
                dates.append(date_str)
                
                # Convert time_str to a datetime object with a fixed date (e.g., '01-01-2000')
                time_datetime = datetime.strptime(time_str, '%H:%M')
                times.append(time_datetime.strftime('%H:%M'))
                
                co2_values.append(float(co2_str))
                temp_values.append(float(temp_str))
                rh_values.append(float(rh_str))

    # Initialize empty lists to store valid date values
    dates_as_datetime = []

    co2q1,co2q99 = np.percentile(co2_values,[1,99])
    tempq1,tempq99 = np.percentile(temp_values,[1,99])
    rhq1,rhq99 = np.percentile(rh_values,[1,99])

    mjd_values = []
    co2s = []
    temps = []
    rhs = []
    # Filter out invalid characters, convert dates and times to Matplotlib date format, and calculate MJD
    for i, (date, time, co2, temp, rh) in enumerate(zip(dates, times, co2_values, temp_values, rh_values)):
        if co2 < co2q99 and co2 > co2q1 and rh < rhq99 and rh > rhq1:
            cleaned_date = date.strip('\x00')
            # Attempt to parse the date and time
            date_time_str = f'{cleaned_date} {time}'
            date_time_as_datetime = datetime.strptime(date_time_str, '%d-%m-%Y %H:%M')
            dates_as_datetime.append(date_time_as_datetime)
            # Calculate the Modified Julian Date (MJD) for the valid date and time
            mjd_value = (date_time_as_datetime - datetime(1858, 11, 17)).total_seconds() / 86400.0
            mjd_values.append(mjd_value)
            co2s.append(co2)
            temps.append(temp)
            rhs.append(rh)
            
           
    # Create a figure and a grid of subplots with 3 rows and 1 column
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8), gridspec_kw={'hspace': 0})

    # Plot CO2 on the first subplot
    axs[0].plot(dates_as_datetime, co2s, label='CO2', color='tab:blue')
    axs[0].set_ylabel('CO2', color='tab:blue')
    axs[0].tick_params(axis='y', labelcolor='tab:blue')

    # Plot Temp on the second subplot
    axs[1].plot(dates_as_datetime, temps, label='Temp', color='tab:red')
    axs[1].set_ylabel('Temp', color='tab:red')
    axs[1].tick_params(axis='y', labelcolor='tab:red')

    # Plot RH on the third subplot
    axs[2].plot(dates_as_datetime, rhs, label='RH', color='tab:green')
    axs[2].set_ylabel('RH', color='tab:green')
    axs[2].tick_params(axis='y', labelcolor='tab:green')

    # Set x-axis format for all subplots
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M'))
        ax.tick_params(axis='x', rotation=45)

    # Create the top x-axis for MJD on the last subplot
    axs[-1].set_xlabel('Modified Julian Date (MJD)')
    axs[-1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[-1].xaxis.set_major_formatter(plt.ScalarFormatter())

    # Add a faint grid
    for ax in axs:
        ax.grid(linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjust spacing
    plt.tight_layout()

    # Display the plot
    plt.savefig('/var/www/html/room_stats/all_stats.jpg')




    # Get the last 100 data points
    dates_as_datetime = dates_as_datetime[-100:]
    co2s = co2s[-100:]
    temps = temps[-100:]
    rhs = rhs[-100:]

    # Create a figure and a grid of subplots with 3 rows and 1 column
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8), gridspec_kw={'hspace': 0})

    # Plot CO2 on the first subplot
    axs[0].plot(dates_as_datetime, co2s, label='CO2', color='tab:blue')
    axs[0].set_ylabel('CO2', color='tab:blue')
    axs[0].tick_params(axis='y', labelcolor='tab:blue')

    # Plot Temp on the second subplot
    axs[1].plot(dates_as_datetime, temps, label='Temp', color='tab:red')
    axs[1].set_ylabel('Temp', color='tab:red')
    axs[1].tick_params(axis='y', labelcolor='tab:red')

    # Plot RH on the third subplot
    axs[2].plot(dates_as_datetime, rhs, label='RH', color='tab:green')
    axs[2].set_ylabel('RH', color='tab:green')
    axs[2].tick_params(axis='y', labelcolor='tab:green')

    # Set x-axis format for all subplots
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M'))
        ax.tick_params(axis='x', rotation=45)

    # Create the top x-axis for MJD on the last subplot
    axs[-1].set_xlabel('Modified Julian Date (MJD)')
    axs[-1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[-1].xaxis.set_major_formatter(plt.ScalarFormatter())

    # Add a faint grid
    for ax in axs:
        ax.grid(linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjust spacing
    plt.tight_layout()

    # Display the plot
    plt.savefig('/var/www/html/room_stats/last_stats.jpg')





    # Get the last 100 data points
    dates_as_datetime = dates_as_datetime[-10:]
    co2s = co2s[-10:]
    temps = temps[-10:]
    rhs = rhs[-10:]
    # Create a figure and a grid of subplots with 3 rows and 1 column
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8), gridspec_kw={'hspace': 0})

    # Plot CO2 on the first subplot
    axs[0].plot(dates_as_datetime, co2s, label='CO2', color='tab:blue')
    axs[0].set_ylabel('CO2', color='tab:blue')
    axs[0].tick_params(axis='y', labelcolor='tab:blue')

    # Plot Temp on the second subplot
    axs[1].plot(dates_as_datetime, temps, label='Temp', color='tab:red')
    axs[1].set_ylabel('Temp', color='tab:red')
    axs[1].tick_params(axis='y', labelcolor='tab:red')

    # Plot RH on the third subplot
    axs[2].plot(dates_as_datetime, rhs, label='RH', color='tab:green')
    axs[2].set_ylabel('RH', color='tab:green')
    axs[2].tick_params(axis='y', labelcolor='tab:green')

    # Set x-axis format for all subplots
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M'))
        ax.tick_params(axis='x', rotation=45)

    # Create the top x-axis for MJD on the last subplot]
    axs[0].set_xlabel('Modified Julian Date (MJD)')
    axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[0].xaxis.set_major_formatter(plt.ScalarFormatter())

    # Add a faint grid
    for ax in axs:
        ax.grid(linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjust spacing
    plt.tight_layout()

    # Display the plot
    plt.savefig('/var/www/html/room_stats/vlast_stats.jpg')
    
    
    # Create a string with the current date and time
    current_date_time = "This was written at "+str(datetime.now().strftime("%d-%m-%Y %H:%M")) + "\n The last data-point is from "+ str(dates[-1]) + " " + str(times[-1])

    # Open the text file in write mode
    with open('/var/www/html/room_stats/datetime.txt', 'w+') as text_file:
        text_file.write(current_date_time)

    print(f"Line '{current_date_time}' has been written to /var/www/html/room_stats/datetime.txt.")


    sleep(1800) # Sleep for 3 seconds

