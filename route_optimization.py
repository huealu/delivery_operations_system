import os
from dotenv import load_dotenv
import requests
import json
import urllib
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


######################## Load APIs ########################

# Load environment variables from .env file
load_dotenv()

# Access the Google Maps API key
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

if google_maps_api_key:
    print("Google Maps API Key loaded successfully.")
else:
    print("Error: Google Maps API Key is missing.")


######################## Create test data ########################

def create_mock_data():
   """Creates a small test data."""
   data = {}
   data['API_key'] = google_maps_api_key
   data['addresses'] = ['3610+Hacks+Cross+Rd+Memphis+TN', # start location
                       '1921+Elvis+Presley+Blvd+Memphis+TN',
                       '149+Union+Avenue+Memphis+TN'
                      ]
   return data


def create_data(addresses):
   """Pack a dictionary of data."""
   data = {}
   data['API_key'] = google_maps_api_key
   data['addresses'] = addresses
   return data


######################## Create data by calculating distance between addresses ########################

def create_distance_matrix(data):
  """Creates a distance matrix from the addresses."""
  addresses = data["addresses"]
  API_key = data["API_key"]
  
  # Set distance matrix to 100 due to Distance Matrix API only accepts 100 elements per request
  # Get rows in multiple requests.
  max_elements = 100
  num_addresses = len(addresses) 
  
  # Maximum number of rows that can be computed per request
  max_rows = max_elements // num_addresses
  
  # num_addresses = q * max_rows + r. 
  q, r = divmod(num_addresses, max_rows)
  dest_addresses = addresses
  distance_matrix = []
  
  # Send q requests, returning max_rows rows per request.
  for i in range(q):
    origin_addresses = addresses[i * max_rows: (i + 1) * max_rows]
    response = send_request(origin_addresses, dest_addresses, API_key)
    distance_matrix += build_distance_matrix(response)

  # Get the remaining remaining r rows, if necessary.
  if r > 0:
    origin_addresses = addresses[q * max_rows: q * max_rows + r]
    response = send_request(origin_addresses, dest_addresses, API_key)
    distance_matrix += build_distance_matrix(response)
  return distance_matrix


def send_request(origin_addresses, dest_addresses, API_key):
  """ Build and send request for the given origin and destination addresses."""
  def build_address_str(addresses):
    # Build a pipe-separated string of addresses
    address_str = ''
    for i in range(len(addresses) - 1):
      address_str += addresses[i] + '|'
    address_str += addresses[-1]
    return address_str
  
  # Set up request
  request = 'https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial'
  origin_address_str = build_address_str(origin_addresses)
  dest_address_str = build_address_str(dest_addresses)
  request = request + '&origins=' + origin_address_str + '&destinations=' + \
                       dest_address_str + '&key=' + API_key
  jsonResult = urllib.request.urlopen(request).read()
  response = json.loads(jsonResult)
  return response


def build_distance_matrix(response):
  """Builds a distance matrix from the response."""
  distance_matrix = []
  for row in response['rows']:
    row_list = [row['elements'][j]['distance']['value'] for j in range(len(row['elements']))]
    distance_matrix.append(row_list)
  return distance_matrix


def create_data_model(distance_matrix, number_vehicles=2):
    """Stores the data for the problem."""
    data = {}
    # Distance matrix: each entry represents the distance between locations
    data['distance_matrix'] = distance_matrix
    data['num_vehicles'] = number_vehicles  # Number of delivery vehicles
    data['depot'] = 0  # Starting and ending location for all routes
    return data


######################## Print Route Optimization results ###########################

def print_solution(manager, routing, solution):
    """Prints the solution on the console."""
    # Create a dictionary to save the path
    route = dict()
    route['vehicles'] = list()

    print('Objective: {}'.format(solution.ObjectiveValue()))
    total_distance = 0
    for vehicle_id in range(manager.GetNumberOfVehicles()):
        index = routing.Start(vehicle_id)
        plan_output = f'Route for vehicle {vehicle_id}:\n'

        # Create a dictionary for each vehicle route
        vehicle_route = dict()
        vehicle_route['id'] = vehicle_id

        # Create a list of all stops
        stops = list()

        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += f' {manager.IndexToNode(index)} ->'
            previous_index = index

            # Save all stops location
            stops.append(manager.IndexToNode(index))

            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

        plan_output += f' {manager.IndexToNode(index)}\n'
        plan_output += f'Distance of the route: {route_distance} units\n'

        # Save the information of vehicle route
        stops.append(manager.IndexToNode(index))
        vehicle_route['route'] = stops
        vehicle_route['distance'] = route_distance

        route['vehicles'].append(vehicle_route)

        # print the information
        print(plan_output)
        total_distance += route_distance

    print(f'Total distance of all routes: {total_distance} units')
    route['distance'] = total_distance

    return route


def encode_address(address, route):
    """Encode the address."""
    # Create a list of all stops addresses
    vehicles_route_address = list()
    for vehicle in route['vehicles']:
        # Create a list of all stops addresses for a vehicle
        stop_adresses = list()

        for stop in vehicle['route']:
            stop_adresses.append(address[stop])
        
        print(f"Vehicle {vehicle['id']} route: {stop_adresses}")

        # save the address of each vehicle route into vehicles_route_address
        vehicles_route_address.append(stop_adresses)
    
    return vehicles_route_address, route['distance']
   


######################## Create Route Optimization model ###########################

def route_optimization_model(addresses=None, number_vehicles=2):
    """Solves the Vehicle Routing Problem."""

    ################ Create data #################
    if addresses is None:
        # Use test data 
        data = create_mock_data()
        addresses = data['addresses']
    else:
        # Use user provided data 
        data = create_data(addresses)

    # Create distance matrix 
    if isinstance(data, dict):
        distance_matrix = create_distance_matrix(data)
    else:
        print("Error: Invalid data format!")
    
    print(distance_matrix) # Print distance matrix 

    # Instantiate the data problem
    data = create_data_model(distance_matrix, number_vehicles)
    
    ################# Create optimization model ###############################

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), 
                                           data['num_vehicles'], 
                                           data['depot'])
    
    # Create the Routing Model
    routing = pywrapcp.RoutingModel(manager)
    
    # Define cost function (distance between locations)
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes"""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add capacity constraint (e.g., for package weights)
    # In here we skip it for simplicity.
    
    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    # Print solution
    if solution:
        route = print_solution(manager, routing, solution)

        # decode the address
        vehicles_route_address, total_distance = encode_address(addresses, route)

        return vehicles_route_address, total_distance
    else:
        print("No solution found!")


if __name__ == "__main__":
    addresses = ['3610+Hacks+Cross+Rd+Memphis+TN', # depot
                     '1921+Elvis+Presley+Blvd+Memphis+TN',
                     '149+Union+Avenue+Memphis+TN',
                     '1034+Audubon+Drive+Memphis+TN',
                     '1532+Madison+Ave+Memphis+TN',
                     '706+Union+Ave+Memphis+TN',
                     '3641+Central+Ave+Memphis+TN',
                     '926+E+McLemore+Ave+Memphis+TN']
    number_vehicles = 1
    route = route_optimization_model(addresses, number_vehicles)

 