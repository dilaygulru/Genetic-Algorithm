import mindspore as ms
import numpy as np
import pandas as pd
import random
import folium
import osmnx as ox
import networkx as nx
from mindspore import Tensor, nn, ops

csv_file_path = "kartal_maltepe_routes.csv"
route_data = pd.read_csv(csv_file_path)


def fitness_function(route):
    row = route_data[route_data["route_id"] == route]
    
    if row.empty:
        return Tensor(0, ms.float32)

    distance = Tensor(row["distance_km"].values[0], ms.float32)
    speed = Tensor(row["avg_speed_kmh"].values[0], ms.float32)
    traffic = Tensor(row["traffic_volume"].values[0], ms.float32)
    weather = Tensor(row["weather_impact"].values[0], ms.float32)

    return (speed * traffic * weather) / distance

def initialize_population(size):
    return np.random.choice(route_data["route_id"].unique(), size=size, replace=False)

def select_parents(population):
    scores = ops.stack([fitness_function(route) for route in population])
    sorted_indices = ops.argsort(scores, descending=True)
    return [population[int(idx.asnumpy())] for idx in sorted_indices[:2]]

def crossover(parent1, parent2):
    return parent1 if random.random() > 0.5 else parent2  

def mutate(child, mutation_rate=0.3):
    if random.random() < mutation_rate:
        child = np.random.choice(route_data["route_id"].unique())
    return child

class GeneticAlgorithm(nn.Cell):
    def __init__(self, pop_size, generations):
        super(GeneticAlgorithm, self).__init__()
        self.pop_size = pop_size
        self.generations = generations
        self.best_route = None
        self.best_score = float('-inf')

    def construct(self):
        population = initialize_population(self.pop_size)

        for _ in range(self.generations):
            parents = select_parents(population)

            child1 = mutate(crossover(parents[0], parents[1]))
            child2 = mutate(crossover(parents[1], parents[0]))

            population = np.array([parents[0], parents[1], child1, child2])

            best_idx = np.argmax([fitness_function(r).asnumpy() for r in population])
            best_route = population[best_idx]
            best_fitness = fitness_function(best_route).asnumpy()

            if best_fitness > self.best_score:
                self.best_score = best_fitness
                self.best_route = best_route

        return self.best_route

pop_size = 10
generations = 20

ga_model = GeneticAlgorithm(pop_size, generations)
best_route = ga_model.construct()  

if best_route is not None:
    print(f"🚀 Optimal Route: {best_route}")
    print(f"🔥 Best Fitness Score: {ga_model.best_score}")

best_route_data = route_data[route_data["route_id"] == best_route]

start_lat = best_route_data["start_lat"].values[0]
start_lon = best_route_data["start_lon"].values[0]
end_lat = best_route_data["end_lat"].values[0]
end_lon = best_route_data["end_lon"].values[0]

G = ox.graph_from_point((start_lat, start_lon), dist=15000, network_type='drive')

orig_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
dest_node = ox.distance.nearest_nodes(G, end_lon, end_lat)

route = nx.shortest_path(G, orig_node, dest_node, weight="length")

m = folium.Map(location=[start_lat, start_lon], zoom_start=13)

route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.7).add_to(m)

folium.Marker(location=[start_lat, start_lon], popup="Start Point", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(location=[end_lat, end_lon], popup="End Point", icon=folium.Icon(color="red")).add_to(m)

map_file_path = "best_route_osm.html"
m.save(map_file_path)

print(f"📌 Updated best route map created: {map_file_path}")
