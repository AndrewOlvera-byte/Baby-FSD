import carla

client = carla.Client("10.0.0.121", 2000)
client.set_timeout(10.0)

world = client.get_world()
print("Connected to CARLA. Current map:", world.get_map().name)
