import os

history_file = "history/recognitions.csv"

def view_all_history():
    if not os.path.exists(history_file):
        print("No history found yet.")
        return
    print("\n--- Full Recognition History ---")
    with open(history_file, "r") as f:
        for line in f:
            print(line.strip())

def view_alerts_only():
    if not os.path.exists(history_file):
        print("No history found yet.")
        return
    print("\n--- ALERTS: Unknown Persons Detected ---")
    with open(history_file, "r") as f:
        found_alerts = False
        for line in f:
            if "Unknown" in line:
                print(line.strip())
                found_alerts = True
        if not found_alerts:
            print("✅ No alerts found. All good!")

def main():
    while True:
        print("\nSearch History Options:")
        print("1. View Full History")
        print("2. View Alerts Only (Unknown Persons)")
        print("3. Exit")
        choice = input("Enter choice: ")

        if choice == "1":
            view_all_history()
        elif choice == "2":
            view_alerts_only()
        elif choice == "3":
            print("Exiting history viewer.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
