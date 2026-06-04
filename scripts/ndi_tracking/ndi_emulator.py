import pyigtl
import time
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R

def start_emulator(port=18944):
    server = pyigtl.OpenIGTLinkServer(port=port)
    print(f"NDI Emulator gestart op poort {port}...")
    print("Wachten op verbinding van de GUI...")
    
    # Simuleer een pen die langzaam rondjes draait om een punt
    t = 0
    try:
        while True:
            if server.is_connected:
                # 1. Genereer posities
                x = 100.0 + 50.0 * np.cos(t)
                y = 100.0 + 50.0 * np.sin(t)
                z = -500.0 + 20.0 * np.sin(t*0.5)
                
                # 2. Genereer en normaliseer quaternion
                q = np.array([0.0, 0.0, np.sin(t/4), np.cos(t/4)])
                q = q / np.linalg.norm(q)
                
                # 3. Bouw de 4x4 transformatiematrix
                matrix = np.eye(4)
                matrix[:3, :3] = R.from_quat(q).as_matrix()
                matrix[:3, 3] = [x, y, z]
                
                # 4. Verstuur als veilig TRANSFORM bericht
                transform = pyigtl.TransformMessage(
                    device_name="PointerToTracker", 
                    matrix=matrix
                )
                server.send_message(transform)
                
                print(f"Versturen: Pos=[{x:.1f}, {y:.1f}, {z:.1f}]", end='\r')
                t += 0.05
            
            time.sleep(0.02) # 50Hz
            
    except KeyboardInterrupt:
        print("\nEmulator gestopt.")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18944
    start_emulator(port)