import airsim
import time
from lcp_solve import *

# do the simulation
(trajA, errA), (trajB, errB) = sim(startA, goalA, startB, goalB, cx, cy, cz)
# plotStatic(trajA, trajB)
# print(trajA[4:7, :])
# print(trajB[3, 0])


client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, "Drone0")
client.enableApiControl(True, "Drone1")
client.armDisarm(True, "Drone0")
client.armDisarm(True, "Drone1")

airsim.wait_key('Press any key to takeoff')
f1 = client.takeoffAsync(vehicle_name="Drone0")
f2 = client.takeoffAsync(vehicle_name="Drone1")

airsim.wait_key('Press any key to move vehicles')
client.moveToPositionAsync(0, 0, -5, 5, vehicle_name="Drone0")
client.moveToPositionAsync(0, 0, -11, 5, vehicle_name="Drone1")



airsim.wait_key('Press any key to move vehicles')
for i in range(trajA.shape[1]):
    client.moveByVelocityAsync(trajA[4, i], trajA[5, i], trajA[6, i], yaw_mode={'is_rate':True, 'yaw_or_rate':trajA[7, i]}, duration=0.5, vehicle_name='Drone0')
    client.moveByVelocityAsync(trajB[4, i], trajB[5, i], trajB[6, i], yaw_mode={'is_rate':True, 'yaw_or_rate':trajB[7, i]}, duration=0.5, vehicle_name='Drone1')
    time.sleep(0.5)


airsim.wait_key('Press any key to reset to original state')

client.armDisarm(False, "Drone0")
client.armDisarm(False, "Drone1")
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False, "Drone0")
client.enableApiControl(False, "Drone1")