import time
import json
import random
import math
from datetime import datetime

class TelemetrySimulator:
    def __init__(self):
        self.step = 0
        # Initialize baseline normal operation variables
        self.base_cpu = 15.0
        self.base_mem = 65.0  # 65% available memory
        self.base_disk = 12.0
        self.base_temp = 45.0
        self.total_drops = 0

    def generate_metrics(self, state="NORMAL"):
        """Generates realistic metrics based on the requested system state."""
        self.step += 1
        
        # Add subtle sine wave noise to mimic natural daily computing cycles
        time_flux = math.sin(self.step / 10.0) * 3.0
        random_noise = random.uniform(-1.5, 1.5)

        if state == "NORMAL":
            # Stable, healthy baseline computing behavior
            cpu = max(1.0, min(100.0, self.base_cpu + time_flux + random_noise))
            mem = max(1.0, min(100.0, self.base_mem - (time_flux * 0.2) + random_noise))
            disk = max(0.0, self.base_disk + (random_noise * 2))
            temp = max(30.0, self.base_temp + (time_flux * 0.5) + (random_noise * 0.2))
            # Healthy network drops happen rarely
            if random.random() > 0.98:
                self.total_drops += random.randint(1, 3)

        elif state == "DEGRADED":
            # Mimics a heavy background build job or minor memory leak
            cpu = max(1.0, min(100.0, 75.0 + (time_flux * 2) + random.uniform(-5, 5)))
            mem = max(1.0, min(100.0, 25.0 - random.uniform(0, 5))) # Available memory dropping
            disk = max(0.0, 180.0 + random.uniform(-20, 20)) # High disk read/write wait
            temp = max(30.0, 72.0 + random.uniform(-2, 3))   # Laptop heating up
            if random.random() > 0.85:
                self.total_drops += random.randint(2, 8)

        elif state == "OUTAGE":
            # Mimics a runaway rogue process or a complete system lockup
            cpu = random.uniform(98.0, 100.0)
            mem = random.uniform(1.0, 4.0)          # Less than 4% memory available
            disk = random.uniform(450.0, 600.0)     # Massive disk I/O blocking queue
            temp = random.uniform(88.0, 95.0)       # Thermal throttling threshold
            self.total_drops += random.randint(15, 50) # Network buffer dropping packets rapidly

        else:
            raise ValueError(f"Unknown state: {state}")

        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metrics": {
                "cpu_utilization_pct": round(cpu, 2),
                "memory_available_pct": round(mem, 2),
                "disk_io_wait_ms": round(disk, 2),
                "network_packet_drops": int(self.total_drops),
                "hardware_temperature_c": round(temp, 2)
            }
        }
        return payload

def main():
    simulator = TelemetrySimulator()
    
    # Configuration loop to easily test different anomaly profiles
    # Change this variable to "NORMAL", "DEGRADED", or "OUTAGE" to test the system
    current_sim_state = "NORMAL" 
    
    print(f"Current Simulation State Profile: [{current_sim_state}]")
    
    while True:
        try:
            telemetry_data = simulator.generate_metrics(state=current_sim_state)
            print(json.dumps(telemetry_data, indent=2))
            time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopping Simulation Daemon safely.")
            break

if __name__ == "__main__":
    main()
