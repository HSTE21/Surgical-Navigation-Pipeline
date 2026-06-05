import pyigtl
import time

HOST = "130.89.204.125"
PORT = 18944

def main():
    client = pyigtl.OpenIGTLinkClient(HOST, PORT)
    print(f"Verbinden met NDI (OpenIGTLink) op {HOST}:{PORT}...")
    
    try:
        while True:
            messages = client.get_latest_messages()
            
            for msg in messages:
                # Controleer of het een positie- of transformbericht is
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

                # Print direct naar de terminal
                print(f"Apparaat: {naam} | X: {pos[0]:.1f}, Y: {pos[1]:.1f}, Z: {pos[2]:.1f}")
            
            # Korte pauze om te voorkomen dat je computer vastloopt (100Hz)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nVerbinding verbroken.")
    except Exception as e:
        print(f"Fout: {e}")

if __name__ == "__main__":
    main()