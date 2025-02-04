# Optimal Route Selection Using Genetic Algorithm

## Overview
This project aims to determine the best route between Kartal and Maltepe using a genetic algorithm implemented in MindSpore. Route selection is based on factors such as distance, average speed, traffic volume, and weather impact. The algorithm selects the optimal route and visualizes it using OpenStreetMap (OSM) data and Folium.

## Dataset
The dataset (`kartal_maltepe_routes.csv`) contains the following columns:
- `route_id`: Unique identifier for the route.
- `distance_km`: Distance of the route in kilometers.
- `avg_speed_kmh`: Average speed along the route in km/h.
- `traffic_volume`: Indicator of traffic congestion.
- `weather_impact`: Factor representing weather conditions affecting the route.
- `start_lat`, `start_lon`: Latitude and longitude of the start point.
- `end_lat`, `end_lon`: Latitude and longitude of the destination.

## Implementation
### Fitness Function
Each route is evaluated based on the following formula:

```
fitness = (speed * traffic * weather) / distance
```

Higher values indicate better routes.

### Genetic Algorithm Components
1. **Initialization**: A random population of routes is generated.
2. **Selection**: The top two routes are chosen based on fitness scores.
3. **Crossover**: Selected routes are combined to form new routes.
4. **Mutation**: Random modifications introduce diversity.
5. **Iteration**: The process is repeated for a defined number of generations to find the best route.

### Visualization
- The optimal route is plotted using `folium` on OpenStreetMap.
- The shortest path is calculated using `osmnx` and `networkx`.
- The final map is saved as `best_route_osm.html`.

## Execution
Ensure the dataset is in the same directory as the script. Run the script using:
```bash
python main.py
```
The script will:
1. Compute the optimal route using the genetic algorithm.
2. Print the best route and its fitness score.
3. Generate a map (`best_route_osm.html`) displaying the route.

### Sample Output
```
🚀 Optimal Route: Route_15
🔥 Best Fitness Score: 4.25
📌 Updated best route map created: best_route_osm.html
```

When the code is executed, an `best_route_osm.html` file will be created. This file can be opened in a browser to view the best route on a map.

## HTML Map Screenshot
Below is the expected screenshot of the generated HTML file:
![image](https://github.com/user-attachments/assets/f35ffa66-1152-4413-a057-b11f06ca2fed)


## Dependencies
- `mindspore`
- `numpy`
- `pandas`
- `folium`
- `osmnx`
- `networkx`

Install dependencies using:
```bash
pip install mindspore numpy pandas folium osmnx networkx
```

## Conclusion
This project demonstrates how genetic algorithms can be used for route optimization. Traffic, speed, and weather conditions are considered to determine the best route, which is then visualized using OpenStreetMap.
