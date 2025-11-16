import carla
import inspect
import agents.navigation.behavior_agent as ba

print("carla module at:", carla.__file__)
print("has Location:", hasattr(carla, "Location"))
print("BehaviorAgent from:", inspect.getsourcefile(ba.BehaviorAgent))