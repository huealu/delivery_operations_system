
from route_optimization import route_optimization_model
from flask import Flask, request, jsonify
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from flask_cors import CORS

app = Flask(__name__)

@app.route('/optimize-route', methods=['POST'])
def optimize_route():
    # Parse incoming data
    data = request.json
    addresses = data.get('addresses')

    # Example 
    def mock_optimization(addresses):
        return {
            "route": addresses[::-1],  # Reverse addresses for demo
            "totalDistance": len(addresses) * 5  # Mock distance
        }
    

    # Run optimization
    result = route_optimization_model(addresses)

    # Rematch addresses to the route
    for id, vehicle in result['vehicles'].values():
        route = result['vehicles'][id]['route']

        route_name = []

        for i in range(len(route)):
            route[i] = addresses[route[i]]
            route_name.append(route[i])
        
        result['route_name'] = route_name
    
    def set_up_result():
        return {
            "route": result['route_name'],
            "totalDistance": result['distance']
        }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
