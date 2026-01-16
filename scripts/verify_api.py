import requests
import os

url = "http://localhost:8080/calibrate"
local_csv = "/workspaces/site-calibration/data/local_validated.csv"
global_csv = "/workspaces/site-calibration/data/global_validated.csv"

# Test 1: Successful calibration
print("Testing Test 1: Successful calibration...")
with open(local_csv, "rb") as l, open(global_csv, "rb") as g:
    files = {
        "local_csv": l,
        "global_csv": g
    }
    data = {"method": "tbc"}
    response = requests.post(url, files=files, data=data)
    
if response.status_code == 200:
    print("SUCCESS")
    result = response.json()
    print("Parameters:", result["parameters"])
    print("Report Preview:\n", result["report"][:200])
else:
    print(f"FAILED: {response.status_code} - {response.text}")

# Test 2: Validation - Less than 3 common points
print("\nTesting Test 2: Less than 3 common points...")
with open("small.csv", "w") as f:
    f.write("Point,Easting,Northing\nP1,100,100\n")

with open("small.csv", "rb") as l, open(global_csv, "rb") as g:
    files = {"local_csv": l, "global_csv": g}
    response = requests.post(url, files=files, data={"method": "tbc"})

print(f"Status Code: {response.status_code} (Expected 400)")
print("Detail:", response.json()["detail"] if response.status_code == 400 else response.text)

# Test 3: Validation - Collinear points
print("\nTesting Test 3: Collinear points...")
# Use Lat/Lon for global and collinear local
with open("collinear_local.csv", "w") as f:
    f.write("Point,Easting,Northing\nP1,0,0\nP2,1,1\nP3,2,2\n")
with open("collinear_global.csv", "w") as f:
    f.write("Point,Lat,Lon\nP1,-24.0, -69.0\nP2,-24.1, -69.1\nP3,-24.2, -69.2\n")

with open("collinear_local.csv", "rb") as l, open("collinear_global.csv", "rb") as g:
    files = {"local_csv": l, "global_csv": g}
    response = requests.post(url, files=files, data={"method": "tbc"})

print(f"Status Code: {response.status_code} (Expected 400)")
print("Detail:", response.json()["detail"] if response.status_code == 400 else response.text)

os.remove("small.csv")
os.remove("collinear_local.csv")
os.remove("collinear_global.csv")
