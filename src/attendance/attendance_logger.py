import csv
import os
from datetime import datetime

class AttendanceLogger:
    def __init__(self, attendance_file="./data/attendance.csv"):
        self.attendance_file = attendance_file
        self._initialize_file()

    def _initialize_file(self):
        if not os.path.exists(os.path.dirname(self.attendance_file)):
            os.makedirs(os.path.dirname(self.attendance_file))
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Name", "Status", "Method"])

    def log_attendance(self, name, status, method="N/A"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.attendance_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, name, status, method])
        print(f"Attendance logged: {timestamp}, {name}, {status}, {method}")
