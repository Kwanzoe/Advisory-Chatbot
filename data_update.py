import json
import pandas as pd

# Data update code down here!


file = open("HealthIntents.json", "r")        # Opens a file stream for reading
json_object = json.load(file)
file.close()


file = open("HealthIntents.json", "w")      # Opens a file stream for writing
json.dump(json_object, file)
file.close()

