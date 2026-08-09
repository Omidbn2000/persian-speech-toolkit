import sounddevice as sd
import numpy as np

def simple_amplitude_monitor():
    """Simple function to monitor amplitude"""
    print("Monitoring microphone input...")
    print("Speak or make noise to see amplitude")
    print("-" * 50)
    
    def callback(indata, frames, time, status):
        # Calculate RMS
        rms = np.sqrt(np.mean(indata ** 2))
        
        # Create simple bar display
        level = int(rms * 2000)  # Scale for visibility
        bar = '=' * min(level, 60)  # Limit bar length
        
        # Clear line and show amplitude
        print(f'\rAmplitude: {rms:.6f} [{" " * 60}]', end='')
        print(f'\rAmplitude: {rms:.6f} [{bar:<60}]', end='')
    
    with sd.InputStream(callback=callback):
        try:
            input("Press Enter to stop...")
        except KeyboardInterrupt:
            pass
    
    print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    simple_amplitude_monitor()