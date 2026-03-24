import subprocess
import time
import gc
import psutil
import os
import datetime
import threading
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

                   

def clean_ram():
    """Attempt to free up memory by running garbage collection."""
    gc.collect()
    logging.info("Garbage collection completed.")


def kill_zombie_processes():
    """Terminate any zombie processes to release system resources."""
    for proc in psutil.process_iter(['pid', 'status']):
        if proc.info['status'] == psutil.STATUS_ZOMBIE:
            try:
                os.kill(proc.info['pid'], 9)
                logging.info(f"Terminated zombie process with PID {proc.info['pid']}")
            except Exception as ex:
                logging.error(f"Failed to kill process {proc.info['pid']}: {ex}")



def run_selected_sync():
    """
    Fire-and-forget call to your shell sync script every 5 minutes.
    Adjust SYNC_SCRIPT if your path/name differs.
    """
    SYNC_SCRIPT = os.path.expanduser('~/bin/sync_selected_to_vps.sh')
    interval = 300  # 5 minutes

    while True:
        try:
            if not os.path.isfile(SYNC_SCRIPT):
                logging.error(f"Sync script not found: {SYNC_SCRIPT}")
            elif not os.access(SYNC_SCRIPT, os.X_OK):
                logging.error(f"Sync script is not executable: {SYNC_SCRIPT}")
            else:
                # Start it asynchronously (don’t wait)
                subprocess.Popen([SYNC_SCRIPT],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.STDOUT)
                logging.info(f"Started sync script: {SYNC_SCRIPT}")
                # Optional: RAM cleanup like your other loops
                clean_ram()
                kill_zombie_processes()
        except Exception as e:
            logging.error(f"Failed to launch sync script: {e}")

        # simple countdown, matching your style
        for remaining in range(interval, 0, -1):
            print(f"Next selected-file sync in {remaining} sec...", end="\r", flush=True)
            time.sleep(1)
        print("Kicking sync now!                          ")



def run_git_commit():
    """Runs the git auto-commit script in a loop."""
    while True:
        try:
            subprocess.run(["python", "git_auto_commit.py"], check=True)
            retry_time = 36000  # 10 hour
        except subprocess.CalledProcessError as e:
            retry_time = 300  # 5 minutes
            logging.error(f"Git auto-commit script crashed with exit code {e.returncode}. Restarting in {retry_time} seconds...")
        except Exception as e:
            retry_time = 300  # 5 minutes
            logging.error(f"Unexpected error in git auto-commit script: {e}. Restarting in {retry_time} seconds...")
        
        # Clean up RAM and kill zombie processes
        clean_ram()
        kill_zombie_processes()

        # Wait before restarting
        for remaining in range(retry_time, 0, -1):
            print(f"Restarting git auto-commit in {remaining} seconds...", end="\r", flush=True)
            time.sleep(1)
        print("Restarting now!                               ")


def run_email_alerts():
    """Runs the email alert system script once per hour."""
    while True:
        try:
            # Run the email alert system script
            subprocess.run(["python", "email_alert_system.py"], check=True)
            logging.info("Email alert check completed. Next check in 1 hour.")
            
            # Wait for 1 hour before running again (3600 seconds)
            wait_time = 3600
            
        except subprocess.CalledProcessError as e:
            wait_time = 300  # 5 minutes if there was an error
            logging.error(f"Email alert script crashed with exit code {e.returncode}. Restarting in {wait_time} seconds...")
        except Exception as e:
            wait_time = 300  # 5 minutes if there was an error
            logging.error(f"Unexpected error in email alert script: {e}. Restarting in {wait_time} seconds...")
        
        clean_ram()
        kill_zombie_processes()

        # Countdown timer before next run
        for remaining in range(wait_time, 0, -1):
            if remaining % 60 == 0:  # Only print every minute to reduce console spam
                print(f"Next email alert check in {remaining//60} minutes...", end="\r", flush=True)
            time.sleep(1)
        print("Running email alerts now!        ") 





def run_ingest():
    """Runs the weather ingest script in a loop."""
    while True:
        try:
            subprocess.run(["python", "server_weather_ingest.py"], check=True)
            retry_time = 30
        except subprocess.CalledProcessError as e:
            retry_time = 5
            logging.error(f"Ingest script crashed with exit code {e.returncode}. Restarting in {retry_time} seconds...")
        except Exception as e:
            retry_time = 5
            logging.error(f"Unexpected error in ingest script: {e}. Restarting in {retry_time} seconds...")
        
        clean_ram()
        kill_zombie_processes()

        for remaining in range(retry_time, 0, -1):
            print(f"Restarting weather ingest in {remaining} seconds...", end="\r", flush=True)
            time.sleep(1)
        print("Restarting now!                               ")


def plant_plot():
    """Runs the plant plot script in a loop."""
    while True:
        try:
            subprocess.run(["python", "../plant_station/data_plot.py"], check=True)
            retry_time = 120
        except subprocess.CalledProcessError as e:
            retry_time = 5
            logging.error(f"Plant plot script crashed with exit code {e.returncode}. Restarting in {retry_time} seconds...")
        except Exception as e:
            retry_time = 5
            logging.error(f"Unexpected error in plant plot script: {e}. Restarting in {retry_time} seconds...")
        
        clean_ram()
        kill_zombie_processes()

        for remaining in range(retry_time, 0, -1):
            print(f"Restarting plant plot in {remaining} seconds...", end="\r", flush=True)
            time.sleep(1)
        print("Restarting now!                               ")


def plant_ingest():
    """Runs the plant ingest script in a loop."""
    while True:
        try:
            subprocess.run(["python", "../plant_station/plant_data_ingest.py"], check=True)
            retry_time = 30
        except subprocess.CalledProcessError as e:
            retry_time = 5
            logging.error(f"Plant ingest script crashed with exit code {e.returncode}. Restarting in {retry_time} seconds...")
        except Exception as e:
            retry_time = 5
            logging.error(f"Unexpected error in plant ingest script: {e}. Restarting in {retry_time} seconds...")
        
        clean_ram()
        kill_zombie_processes()

        for remaining in range(retry_time, 0, -1):
            print(f"Restarting plant ingest in {remaining} seconds...", end="\r", flush=True)
            time.sleep(1)
        print("Restarting now!                               ")


def run_processing():
    """Runs the weather processing script in a loop."""
    while True:
        try:
            subprocess.run(["python", "server_weather_processing.py"], check=True)
            proc_retry_time = 120

            current_time = datetime.datetime.now()
            start_time = current_time + datetime.timedelta(seconds=proc_retry_time)

            logging.info(
                f"Weather processing complete!!!!!\n\n"
                f"\tCurrent time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"\tSleeping for {proc_retry_time / 60:.2f} mins...\n"
                f"\tWill start again at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"============================================================================================="
            )
        except subprocess.CalledProcessError as e:
            proc_retry_time = 5
            logging.error(f"Processing script crashed with exit code {e.returncode}. Restarting in {proc_retry_time} seconds...")
        except Exception as e:
            proc_retry_time = 5
            logging.error(f"Unexpected error in processing script: {e}. Restarting in {proc_retry_time} seconds...")
        
        clean_ram()
        kill_zombie_processes()

        for remaining in range(proc_retry_time, 0, -1):
            print(f"Restarting weather processing in {remaining} seconds...", end="\r", flush=True)
            time.sleep(1)
        print("Restarting now!                               ")


if __name__ == "__main__":
    # Create threads for each process
    threads = [
        threading.Thread(target=run_ingest),
        #threading.Thread(target=plant_plot),
        #threading.Thread(target=plant_ingest),
        threading.Thread(target=run_processing),
        threading.Thread(target=run_email_alerts),
        threading.Thread(target=run_git_commit),
        threading.Thread(target=run_selected_sync)
    ]


    # Start all threads
    for t in threads:
        t.start()

    # Join threads to keep the main thread alive
    for t in threads:
        t.join()
