# JalanKu - Road Damage Detection Model

YOLOv8 model for detecting road damages including potholes and cracks.

## Model Information

- **Architecture**: YOLOv8
- **Training Epochs**: 300
- **Classes**: 4
  - Longitudinal
  - Transverse
  - Alligator
  - Pothole

## Performance

| Image | Detection Results |
|-------|------------------|
| images.jpg | 4 Potholes |
| images 1.jpg | 1 Alligator |
| MicrosoftTeams-image_32.jpg | 1 Pothole |

## Usage

### Python

```python
from ultralytics import YOLO

model = YOLO('model_jalanku.pt')

# Recommended inference settings
results = model.predict(
    image,
    conf=0.25,    # Confidence threshold
    iou=0.45,     # NMS IoU threshold
    max_det=50    # Max detections
)
```

### Recommended Configuration

```python
# For production use
results = model.predict(
    image,
    conf=0.3,      # Minimize false positives
    iou=0.45,
    augment=False  # Consistent results
)
```

## Post-Processing Filter

```python
def filter_detections(results, min_area=1000, max_area=100000):
    filtered = []
    for r in results:
        for box in r.boxes:
            area = (box.xywh[0][2] * box.xywh[0][3])
            if min_area < area < max_area:
                filtered.append(box)
    return filtered
```

## Flutter Integration (TFLite)

### Export to TFLite

```bash
yolo export model=model_jalanku.pt format=tflite
```

### Flutter Implementation

```dart
// Recommended settings for TFLite
final interpreter = await Interpreter.fromAsset('model_jalanku.tflite');

// Pre-processing
img = resize(img, 640, 640);
img = normalize(img, 0, 255);

// Post-processing with confidence filter
if (confidence > 0.3 && isValidSize(bbox)) {
  // Accept detection
}
```

## Handling False Positives

1. **Confidence Threshold**: Use 0.25-0.35
2. **IoU Threshold**: 0.45-0.5 for NMS
3. **Size Filter**: Minimum area 1000 px²
4. **Spatial Filter**: Only accept detections in road areas

## Model Files

- `model_jalanku.pt` - PyTorch model
- `result/` - Validation results

## Training Notes

- Model trained for 300 epochs
- Confidence < 0.1 produces false positives
- Recommended confidence: 0.25-0.3

## License

MIT License

## Author

Dannofc
