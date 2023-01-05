import random

# 倍数法でOD需要を作成する関数(元のtripsデータをスカラー倍)
def make_trips_multiple(original_trips, num_zones, original_total_flow, total_flow):

    trips = {}

    for origin_node in original_trips.keys():
        trips_flow = {}
        for destination_node in original_trips[origin_node].keys():
            trips_flow[destination_node + num_zones] = original_trips[origin_node][destination_node] * (total_flow/original_total_flow)
        trips[origin_node] = trips_flow

    # print(original_trips)
    # print(trips)

    return trips


# 乱数でOD需要を作成する関数
def make_trips_random(num_zones, total_flow):

    rand = {}
    sum_rand = 0.0
    for origin_node in range(1, num_zones+1):
        rand[origin_node] = {}
        for destination_node in range(num_zones+1, num_zones*2+1):
            rand[origin_node][destination_node] = random.uniform(0.0, 1.0)
            sum_rand += rand[origin_node][destination_node]

    trips = {}
    for origin_node in range(1, num_zones+1):
        trips[origin_node] = {}
        for destination_node in range(num_zones+1, num_zones*2+1):
            trips[origin_node][destination_node] = total_flow * rand[origin_node][destination_node]/sum_rand

    return trips

def trips_float_to_int(trips, total_flow):

    trips_int = {}

    sum_flow = 0.0

    for origin_node in trips.keys():
        trips_int[origin_node] = {}
        for dest_node in trips[origin_node].keys():
            trips_int[origin_node][dest_node] = float(round(trips[origin_node][dest_node]))
            sum_flow += float(round(trips[origin_node][dest_node]))

    origin_node = list(trips_int.keys())[0]
    dest_node = list(trips_int[list(trips_int.keys())[0]].keys())[0]
    trips_int[origin_node][dest_node] += total_flow - sum_flow

    return trips_int


if __name__ == "__main__":

    import os
    import readNetwork as rn

    root = os.path.dirname(os.path.abspath('.'))
    root = os.path.join(root, '..', '_sampleData', 'Sample')

    # print(root)

    original_trips = rn.read_trips(os.path.join(root, 'Sample_trips.tntp'))
    original_total_flow = rn.read_total_flow(os.path.join(root, 'Sample_trips.tntp'))
    # print(original_trips)
    # print(original_tf)

    num_zones = rn.read_num_zones(os.path.join(root, 'Sample_net.tntp'))
    # print(num_zones)

    [vtflow_data, utflow_data] = rn.read_tflow(os.path.join(root, 'Scenario_0', 'Sample_tflow.tntp'))
    # print(vtflow_data)
    # print(utflow_data)
    
    for vehicle_num in vtflow_data.keys():
        vehicle_trips = make_trips_multiple(original_trips, num_zones, original_total_flow, vtflow_data[vehicle_num])
        # print(vehicle_trips)

        vehicle_trips  = make_trips_random(num_zones, vtflow_data[vehicle_num])
        # print(vehicle_trips)
        # os.makedirs(os.path.join(root, 'Scenario_0', 'virtual_net', 'vehicle', str(vehicle_num)), exist_ok=True)
        # rn.write_trips(os.path.join(root, 'Scenario_0', 'virtual_net', 'vehicle', str(vehicle_num), 'Sample_vir_trips.tntp'), vehicle_trips, num_zones*2, vtflow_data[vehicle_num])
        vehicle_trips = trips_float_to_int(vehicle_trips, vtflow_data[vehicle_num])
        print(vehicle_trips)

    for user_num in utflow_data.keys():
        user_trips = make_trips_multiple(original_trips, num_zones, original_total_flow, utflow_data[user_num])
        # print(user_trips)

        user_trips = make_trips_random(num_zones, utflow_data[user_num])
        # print(user_trips)
        # os.makedirs(os.path.join(root, 'Scenario_0', 'virtual_net', 'user', str(user_num)), exist_ok=True)
        # rn.write_trips(os.path.join(root, 'Scenario_0', 'virtual_net', 'user', str(user_num), 'Sample_vir_trips.tntp'), user_trips, num_zones*2, utflow_data[vehicle_num])
        user_trips = trips_float_to_int(user_trips, utflow_data[user_num])
        print(user_trips)
