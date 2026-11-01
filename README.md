# LocalMatte

**A local, CPU-based video background removal tool built with Python.**

LocalMatte is a personal project exploring video segmentation and object tracking to remove video backgrounds and export the results as transparent PNG frames.

The goal is to build a tool that runs entirely on a local machine without relying on cloud APIs.

## Project Goals

The planned workflow is:

1. Upload a video.
2. Select a subject by clicking on the first frame.
3. Use Meta's SAM 2 model to segment and track the selected subject across the video.
4. Apply the generated masks to create transparent RGBA frames.
5. Export the result as a numbered PNG sequence.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Video reading and frame processing |
| PyTorch | Deep learning framework |
| Meta SAM 2 | Object segmentation and video tracking |
| Gradio | Local web interface |

## Development Status

This project is currently under active development. The complete background-removal workflow is not yet implemented.

### Completed

- [x] Python environment setup
- [x] CPU-based PyTorch installation and verification
- [x] OpenCV installation and verification
- [x] Basic video reading and frame extraction

### Planned

- [ ] SAM 2 static image segmentation
- [ ] SAM 2 video object tracking
- [ ] Transparent RGBA frame generation
- [ ] Numbered PNG sequence export
- [ ] Gradio user interface
- [ ] Documentation and demonstration

## Hardware Considerations

LocalMatte is being developed and tested on a CPU-only system without a dedicated GPU.

Performance and model compatibility on CPU are important considerations throughout development.

## Project Motivation

This project is an opportunity to gain practical experience with computer vision, deep learning, video processing, and Python application development while building a useful local-first tool.

## License

License not yet selected.