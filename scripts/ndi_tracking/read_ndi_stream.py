# To do: print x, y, z rotations as well. Because coordinate correction for tooltip needs rotation and translation. For now, only translation is printed. Rotation can be calculated from the 4x4 matrix in the TransformMessage, but it is not implemented in this code snippet.

import pyigtl
import time
import pandas as pd
import threading
import os
from datetime import datetime

HOST = "130.89.204.125"
PORT = 18944

class NDIDataCapture:
    def __init__(self):
        self.data = []
        self.running = True
        self.lock = threading.Lock()
        self.event_flag = False
        
    def ndi_reader_thread(self):
        """NDI data read thread"""
        try:
            client = pyigtl.OpenIGTLinkClient(HOST, PORT)
            print(f"Verbonden met NDI op {HOST}:{PORT}")
            
            while self.running:
                messages = client.get_latest_messages()
                
                for msg in messages:
                    if isinstance(msg, pyigtl.TransformMessage):
                        naam = getattr(msg, 'device_name', 'Onbekend')
                        pos = msg.matrix[:3, 3]
                    elif isinstance(msg, pyigtl.PositionMessage):
                        naam = getattr(msg, 'device_name', 'Onbekend')
                        pos = msg.positions[0] if getattr(msg, 'positions', None) else None
                    else:
                        continue
                    
                    if pos is None:
                        continue
                    
                    with self.lock:
                        event = 1 if self.event_flag else 0
                        self.data.append({
                            'timestamp': datetime.now(),
                            'device': naam,
                            'X': pos[0],
                            'Y': pos[1],
                            'Z': pos[2],
                            'event': event
                        })
                        self.event_flag = False
                    
                    print(f"Apparaat: {naam} | X: {pos[0]:.1f}, Y: {pos[1]:.1f}, Z: {pos[2]:.1f}")
                
                time.sleep(0.01)
        except Exception as e:
            print(f"NDI fout: {e}")
    
    def input_thread(self):
        """Thread for capturing Enter events"""
        while self.running:
            try:
                input()  # Wacht op Enter
                with self.lock:
                    self.event_flag = True
                print("[EVENT MARKED]")
            except Exception:
                break
    
    def start(self):
        """Start capture threads"""
        ndi_t = threading.Thread(target=self.ndi_reader_thread, daemon=True)
        input_t = threading.Thread(target=self.input_thread, daemon=True)
        
        ndi_t.start()
        input_t.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStoppend...")
            self.running = False
            ndi_t.join(timeout=2)
            input_t.join(timeout=2)
    
    def get_dataframe(self):
        """Return collected data as pandas DataFrame"""
        with self.lock:
            df = pd.DataFrame(self.data)
        return df
    
    def save_csv(self, filename="ndi_capture.csv"):
        """Save dataframe to CSV"""
        # Zorg dat de map bestaat
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        df = self.get_dataframe()
        df.to_csv(filename, index=False)
        print(f"✓ Opgeslagen: {filename}")
        return df

def main():
    capture = NDIDataCapture()
    print("Druk op Enter om events te markeren (Ctrl+C om te stoppen)")
    capture.start()
    
    # Na stoppen: toon dataframe en sla op in data/ndi_captures
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(current_dir, "..", "..", "data", "ndi_captures", "ndi_capture.csv")
    
    df = capture.save_csv(default_path)
    print(f"\n{len(df)} rijen verzameld")
    print(df.head(10))

if __name__ == "__main__":
    main()