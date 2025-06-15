#!/usr/bin/env python3
"""
Git Auto-Commit Script

This script automatically commits and pushes to a Git repository twice a day,
even if no files have changed. It creates empty commits with timestamps
when no changes are detected.
"""

import subprocess
import time
import os
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Repository path - adjust as needed
REPO_PATH = "/media/bigdata"

# Times to commit each day (24-hour format)
COMMIT_TIMES = ["22:00"]  # 10 AM and 10 PM

def run_git_command(cmd):
    """Execute a git command and return the output."""
    try:
        full_cmd = ["git"] + cmd
        logging.info(f"Executing: {' '.join(full_cmd)}")
        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=REPO_PATH
        )
        stdout, stderr = process.communicate()
        return_code = process.returncode
        
        stdout = stdout.decode('utf-8').strip()
        stderr = stderr.decode('utf-8').strip()
        
        if return_code != 0:
            logging.warning(f"Git command returned non-zero exit code {return_code}")
            logging.warning(f"Stderr: {stderr}")
        else:
            logging.info(f"Command succeeded: {stdout}")
        
        return return_code, stdout, stderr
    
    except Exception as e:
        logging.error(f"Error executing git command: {e}")
        return 1, "", str(e)

def check_for_changes():
    """Check if there are any changes in the repository."""
    return_code, stdout, stderr = run_git_command(["status", "--porcelain"])
    return stdout.strip() != ""

def get_current_branch():
    """Get the name of the current branch."""
    return_code, stdout, stderr = run_git_command(["branch", "--show-current"])
    return stdout.strip()

def create_commit(has_changes):
    """Create a commit with the current timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if has_changes:
        # Add all changes
        run_git_command(["add", "--all"])
        message = f"Auto-commit: Changes detected - {timestamp}"
    else:
        # Create an empty commit
        message = f"Auto-commit: No changes - {timestamp}"
    
    # Create the commit - using --allow-empty ensures we can commit without changes
    return_code, stdout, stderr = run_git_command(["commit", "--allow-empty", "-m", message])
    return return_code == 0

def push_to_remote():
    """Push commits to the remote repository."""
    branch = get_current_branch()
    return_code, stdout, stderr = run_git_command(["push", "origin", branch])
    return return_code == 0

def is_commit_time():
    """Check if it's time to make a commit based on the predefined schedule."""
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    
    # Check if current time matches any of the commit times
    for commit_time in COMMIT_TIMES:
        # Allow a 5-minute window for each commit time
        commit_hour, commit_minute = map(int, commit_time.split(":"))
        commit_datetime = now.replace(hour=commit_hour, minute=commit_minute, second=0, microsecond=0)
        
        time_diff = abs((now - commit_datetime).total_seconds())
        if time_diff <= 300:  # Within 5 minutes
            return True
    
    return False

def run_git_auto_commit():
    """Main function to handle automatic commits and pushes."""
    last_commit_time = datetime.now() - timedelta(hours=12)  # Initialize to ensure first check works
    
    while True:
        try:
            current_time = datetime.now()
            
            # Check if it's time to commit and we haven't committed recently
            if is_commit_time() and (current_time - last_commit_time).total_seconds() > 6 * 3600:
                logging.info("Git auto-commit: It's time to commit!")
                
                # Check for changes
                has_changes = check_for_changes()
                commit_status = "changes detected" if has_changes else "no changes"
                logging.info(f"Git auto-commit: Repository status - {commit_status}")
                
                # Create a commit
                if create_commit(has_changes):
                    logging.info("Git auto-commit: Commit created successfully")
                    
                    # Push to remote
                    if push_to_remote():
                        logging.info("Git auto-commit: Changes pushed to remote successfully")
                        last_commit_time = current_time  # Update last commit time
                    else:
                        logging.error("Git auto-commit: Failed to push changes to remote")
                else:
                    logging.error("Git auto-commit: Failed to create commit")
            
            # Sleep for a while before checking again
            for remaining in range(60, 0, -1):
                print(f"Next git auto-commit check in {remaining} seconds...", end="\r", flush=True)
                time.sleep(1)
            print("Checking git auto-commit now!                           ")
            
        except subprocess.CalledProcessError as e:
            retry_time = 300  # 5 minutes
            logging.error(f"Git auto-commit: Process error with code {e.returncode}. Restarting in {retry_time} seconds...")
            time.sleep(retry_time)
        except Exception as e:
            retry_time = 16000  # 5 minutes
            logging.error(f"Git auto-commit: Unexpected error: {e}. Restarting in {retry_time} seconds...")
            time.sleep(retry_time)

if __name__ == "__main__":
    # For standalone testing
    run_git_auto_commit()
