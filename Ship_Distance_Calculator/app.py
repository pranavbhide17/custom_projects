from flask import Flask, request, render_template_string
import math
from datetime import datetime
import pandas as pd

app = Flask(__name__)

def load_data(file_path):
    return pd.read_csv(file_path)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return R * c  # Distance in kilometers

def find_closest_points(data, poi_lat, poi_lon):
    data['timestamp_position'] = pd.to_datetime(data['timestamp_position'].str.strip("'"))

    results = []

    for vessel_name, group in data.groupby('name'):
        group['distance'] = group.apply(
            lambda row: haversine_distance(row['lat'], row['lon'], poi_lat, poi_lon), axis=1
        )

        closest_point = group.loc[group['distance'].idxmin()]

        results.append({
            'name': vessel_name,
            'timestamp': closest_point['timestamp_position'],
            'lat': closest_point['lat'],
            'lon': closest_point['lon'],
            'distance': closest_point['distance']
        })

    return pd.DataFrame(results)

@app.route('/', methods=['GET', 'POST'])
def index():
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Shipping Analytics</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
            }
            .container {
                display: flex;
                flex-direction: row;
                height: 100vh;
            }
            .left-panel {
                flex: 2;
                padding: 20px;
                background-color: #e8f4fc;
                overflow-y: auto;
            }
            .right-panel {
                flex: 3;
                padding: 20px;
                background-color: #d3d3d3;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            form {
                margin-bottom: 20px;
            }
            .map-container {
                width: 100%;
                height: 100%;
                background-color: white;
                border: 1px solid #ccc;
                display: flex;
                justify-content: center;
                align-items: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="left-panel">
                <h1>Upload CSV File and Enter POI</h1>
                <form action="/" method="post" enctype="multipart/form-data">
                    <label for="file">Select CSV File:</label>
                    <input type="file" name="file" id="file" required><br><br>

                    <label for="poi_lat">POI Latitude:</label>
                    <input type="text" name="poi_lat" id="poi_lat" required><br><br>

                    <label for="poi_lon">POI Longitude:</label>
                    <input type="text" name="poi_lon" id="poi_lon" required><br><br>

                    <button type="submit">Submit</button>
                </form>

                {% if tables %}
                <h2>Table of Latest Points</h2>
                {{ tables|safe }}
                {% endif %}
            </div>
            <div class="right-panel">
                <div class="map-container">
                    <p>Div for Showing the Map.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

    if request.method == 'POST':
        file = request.files['file']
        poi_lat = float(request.form['poi_lat'])
        poi_lon = float(request.form['poi_lon'])

        data = load_data(file)
        closest_points = find_closest_points(data, poi_lat, poi_lon)

        return render_template_string(html_template, tables=closest_points.to_html(classes='data'))

    return render_template_string(html_template, tables=None)

if __name__ == '__main__':
    app.run(debug=True)
