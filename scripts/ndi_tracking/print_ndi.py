import pyigtl
import time
import sys

def main(host="127.0.0.1", port=18944):
    client = pyigtl.OpenIGTLinkClient(host, port)
    print(f"Verbinden met NDI (OpenIGTLink) op {host}:{port}...")
    print("Druk op Ctrl+C om te stoppen.\n")
    
    try:
        while True:
            messages = client.get_latest_messages()
            
            for msg in messages:
                # Controleer of het een positie- of transformbericht is
                if isinstance(msg, (pyigtl.TransformMessage, pyigtl.PositionMessage)):
                    # Haal naam en positie op
                    naam = getattr(msg, 'device_name', 'Onbekend')
                    pos = msg.position
                    
                    # Print direct naar de terminal
                    print(f"Apparaat: {naam} | X: {pos[0]:.1f}, Y: {pos[1]:.1f}, Z: {pos[2]:.1f}")
            
            # Korte pauze om te voorkomen dat je computer vastloopt (100Hz)
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nScript gestopt.")

if __name__ == "__main__":
    # Gebruik: python print_ndi.py [IP_ADRES]
    ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    main(ip)