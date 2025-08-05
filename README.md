# Computer Vision Motion Tracking Suite

A comprehensive Python suite for real-time motion tracking and analysis using OpenCV, featuring three specialized applications for different use cases.

## Overview

This project contains three main scripts, each designed for specific motion tracking and analysis tasks:

1. **`vis.py`** - Human Pose Tracker with Hand/Finger Detection
2. **`vis_gen.py`** - General Object Motion Tracker
3. **`vis_gen_music.py`** - Motion-Controlled Dance Music Synthesizer

## Installation

### Prerequisites
- Python 3.7 or higher
- Working webcam
- Windows/Linux/macOS

### Setup
1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install opencv-python numpy scipy pygame
   ```

## Script 1: Human Pose Tracker (`vis.py`)

### Purpose
Advanced human body feature detection with specialized hand and finger tracking, designed to eliminate false positives and provide accurate pose analysis.

### Features
- **Multi-method Hand Detection**: Combines skin color detection, motion analysis, and edge detection
- **Finger Detection**: Advanced contour analysis with convex hull and protrusion detection
- **False Positive Filtering**: Intelligent poster/background object filtering
- **Real-time Calibration**: Adjust skin detection sensitivity on-the-fly
- **Debug Mode**: Visualize detection masks and intermediate processing steps
- **Body Part Tracking**: Face, upper body, and pose metrics
- **Movement Scoring**: 0-1 normalized movement values for all detected features

### Controls
- **'d'**: Toggle debug mode (shows skin mask, motion mask, detection info)
- **'+'/'-'**: Adjust skin detection sensitivity in real-time
- **'q'**: Quit program

### Usage
```bash
python vis.py
```

### Debug Mode Features
When debug mode is enabled, you'll see:
- **Skin Mask**: Top-right corner showing skin color detection
- **Motion Mask**: Bottom-right corner showing motion detection
- **Detection Info**: Current skin detection parameters and calibration instructions
- **Filter Status**: Information about which detection filters are active

### Technical Details
- **Skin Detection**: Uses HSV color space with multiple adaptive ranges
- **Motion Detection**: Frame differencing with morphological operations
- **Finger Detection**: Contour analysis with convex hull and peak detection
- **Poster Filtering**: Analyzes object properties to exclude large, uniform areas
- **Performance**: Optimized for real-time processing with frame skipping

## Script 2: General Motion Tracker (`vis_gen.py`)

### Purpose
Tracks the overall movement of any objects in the video feed, assigning unique IDs and movement scores on a 0-1 scale.

### Features
- **Object Tracking**: Assigns unique IDs to moving objects
- **Movement Scoring**: Calculates 0-1 movement scores based on trajectory analysis
- **Trajectory Visualization**: Shows object movement paths over time
- **Screen Size Control**: Dynamic resolution switching (5 preset sizes)
- **Fullscreen Mode**: Toggle between windowed and fullscreen display
- **Real-time Sensitivity**: Adjust motion detection sensitivity
- **Debug Mode**: Visualize motion masks and tracking information

### Controls
- **'d'**: Toggle debug mode
- **'t'**: Toggle trajectory display
- **'+'/'-'**: Adjust motion sensitivity
- **'1'-'5'**: Change screen size (320x240 to 1280x720)
- **'f'**: Toggle fullscreen mode
- **'r'**: Reset tracking data
- **'q'**: Quit program

### Usage
```bash
python vis_gen.py
```

### Debug Mode Features
- **Motion Mask**: Shows detected motion areas
- **Raw Motion**: Displays unfiltered motion detection
- **Test Patterns**: Visual indicators when no motion is detected
- **Statistics**: Object count, movement scores, and system parameters

### Technical Details
- **Motion Detection**: Frame differencing with adaptive thresholding
- **Object Tracking**: Distance-based object association with history
- **Movement Calculation**: Euclidean distance analysis with normalization
- **Resolution Adaptation**: Automatic parameter adjustment based on screen size
- **Camera Management**: Robust camera reconnection and error handling

## Script 3: Motion-Controlled Synthesizer (`vis_gen_music.py`)

### Purpose
Creates real-time dance music by mapping motion parameters to a sophisticated synthesizer, producing professional-quality electronic music.

### Features
- **Dance Music Generation**: Complete electronic music production system
- **Multi-instrument Synthesis**: Kick, snare, hi-hat, bass, lead, harmony, and pad
- **Musical Structure**: BPM tracking, chord progressions, key changes, and modes
- **Volume Control**: Individual volume adjustment for each instrument
- **Audio Processing**: Compression, limiting, and EQ for professional sound
- **Motion Mapping**: Translates movement to musical parameters
- **Real-time Control**: Adjust volumes and parameters during performance

### Controls
- **'d'**: Toggle debug mode
- **'t'**: Toggle trajectory display
- **'+'/'-'**: Adjust motion sensitivity
- **'1'-'5'**: Change screen size
- **'f'**: Toggle fullscreen mode
- **'a'**: Toggle audio on/off
- **'v'**: Cycle through volume parameters (kick, snare, hi-hat, bass, lead, harmony, pad)
- **Arrow Keys**: Adjust volume of selected parameter (up/down)
- **'r'**: Reset tracking
- **'q'**: Quit program

### Usage
```bash
python vis_gen_music.py
```

### Audio Features

#### Instruments
- **Kick Drum**: Pitch envelope with frequency sweep and click
- **Snare Drum**: Filtered noise with tone components
- **Hi-Hat**: High-frequency noise with metallic character
- **Bass Line**: Harmonic synthesis with saturation
- **Lead Melody**: Multi-harmonic with chorus effect
- **Harmony**: Chord-based synthesis
- **Pad**: Sustained chords with multiple harmonics

#### Musical Structure
- **BPM**: 128 beats per minute (dance tempo)
- **Chord Progressions**: I-V-vi-IV and other dance music patterns
- **Key Changes**: Automatic modulation every 8 measures
- **Modes**: Alternates between major and minor scales
- **Rhythm Patterns**: Four-on-the-floor kick, 2&4 snare, constant hi-hat

#### Motion-to-Music Mapping
- **Average Movement** → Lead frequency
- **Object Count** → Bass frequency
- **Maximum Movement** → Overall energy
- **Movement Variance** → Rhythm complexity

#### Audio Processing
- **Gentle Compression**: Threshold 0.3, ratio 2.0
- **Dance EQ**: Bass boost, kick boost, high frequency enhancement
- **Limiting**: Soft limiting at 0.8 with hard clip at 0.85
- **Stereo Mixing**: Slight panning for spatial depth

### Debug Mode Features
- **Audio Parameters**: BPM, beat count, energy, frequencies
- **Volume Display**: Current parameter and its volume level
- **Motion Mask**: Visual feedback for motion detection
- **System Statistics**: Performance and tracking information

## Common Features Across All Scripts

### Performance Optimization
- **Frame Skipping**: Reduces processing load
- **Resolution Scaling**: Automatic parameter adjustment
- **Buffer Management**: Optimized camera and audio buffers
- **Error Handling**: Robust camera reconnection

### Visual Feedback
- **Color Coding**: Movement-based color indicators
- **Bounding Boxes**: Object detection visualization
- **Trajectories**: Movement path visualization
- **Text Overlays**: Real-time statistics and parameters

### Configuration
- **Real-time Adjustment**: Most parameters can be changed during runtime
- **Preset Management**: Multiple screen sizes and sensitivity levels
- **Debug Tools**: Comprehensive visualization of internal processing

## Troubleshooting

### Common Issues

1. **Camera Not Detected**:
   - Check webcam connection and permissions
   - Try different camera indices (0, 1, 2)
   - Restart the application

2. **Poor Detection Quality**:
   - Ensure adequate lighting
   - Maintain appropriate distance from camera
   - Use contrasting clothing/background
   - Adjust sensitivity parameters

3. **Audio Issues** (vis_gen_music.py):
   - Check system audio settings
   - Ensure pygame is properly installed
   - Try different audio buffer sizes
   - Restart audio system with 'a' key

4. **Performance Problems**:
   - Reduce screen resolution
   - Lower motion sensitivity
   - Close other applications
   - Use smaller contour areas

### Installation Issues

1. **OpenCV Installation**:
   ```bash
   pip install --upgrade pip
   pip install opencv-python
   ```

2. **Pygame Audio Issues**:
   ```bash
   pip install pygame --upgrade
   ```

3. **Windows-specific Issues**:
   - Install Visual C++ build tools
   - Use virtual environment
   - Run as administrator if needed

## File Structure

```
Vis/
├── vis.py              # Human pose tracker with hand detection
├── vis_gen.py          # General object motion tracker
├── vis_gen_music.py    # Motion-controlled synthesizer
├── requirements.txt    # Python dependencies
└── README.md          # This documentation
```

## Dependencies

- **opencv-python**: Computer vision and camera capture
- **numpy**: Numerical computing and array operations
- **scipy**: Scientific computing (peak detection)
- **pygame**: Audio synthesis and playback
- **threading**: Multi-threaded audio processing
- **queue**: Inter-thread communication

## Technical Specifications

### System Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 100MB free space
- **Audio**: Working audio output (for music synthesis)

### Performance Targets
- **Frame Rate**: 30 FPS target
- **Latency**: <100ms audio latency
- **CPU Usage**: <50% on modern systems
- **Memory**: <500MB RAM usage

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## Version History

- **v1.0**: Initial release with vis.py (human pose tracking)
- **v2.0**: Added vis_gen.py (general motion tracking)
- **v3.0**: Added vis_gen_music.py (motion-controlled synthesizer)
- **v3.1**: Enhanced audio quality and volume controls