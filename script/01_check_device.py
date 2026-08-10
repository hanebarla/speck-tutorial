import samna
import sinabs
import sinabs.backend.dynapcnn.io as sio

print("samna:", getattr(samna, "__version__", "unknown"))
print("sinabs:", getattr(sinabs, "__version__", "unknown"))

device_map = sio.get_device_map()
print(device_map)